//! Quarantined memory allocation and memory-mapping primitives (`novaserve-tensor::alloc`).
//!
//! All `unsafe` blocks in this module carry explicit provenance and alignment invariants
//! verified by `cargo +nightly miri test -p novaserve-tensor`.

use crate::error::TensorError;
use std::alloc::{self, Layout};
use std::cell::Cell;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::ptr::NonNull;

pub const CACHE_LINE_ALIGN: usize = 64;

/// 64-byte cache-aligned owned byte buffer whose backing allocation is fully zero-initialized.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuf` exclusively owns its heap buffer allocated via `std::alloc`,
// with no interior mutability or thread-local state.
unsafe impl Send for AlignedBuf {}
// SAFETY: Shared `&AlignedBuf` references only expose immutable byte/f32 slices.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new_zeroed(len: usize) -> Result<Self, TensorError> {
        let cap = len
            .max(CACHE_LINE_ALIGN)
            .checked_next_multiple_of(CACHE_LINE_ALIGN)
            .ok_or(TensorError::AllocationFailed {
                size: len,
                align: CACHE_LINE_ALIGN,
            })?;
        let layout = Layout::from_size_align(cap, CACHE_LINE_ALIGN).map_err(|_| {
            TensorError::AllocationFailed {
                size: cap,
                align: CACHE_LINE_ALIGN,
            }
        })?;
        // SAFETY: `layout.size() >= 64 > 0` and `align == 64` is a valid power of two.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or(TensorError::AllocationFailed {
            size: cap,
            align: CACHE_LINE_ALIGN,
        })?;
        Ok(Self {
            ptr,
            len,
            cap,
            layout,
        })
    }

    pub fn from_bytes(src: &[u8]) -> Result<Self, TensorError> {
        let mut buf = Self::new_zeroed(src.len())?;
        buf.as_bytes_mut().copy_from_slice(src);
        Ok(buf)
    }

    pub fn from_f32_slice(src: &[f32]) -> Result<Self, TensorError> {
        let byte_len = src
            .len()
            .checked_mul(4)
            .ok_or(TensorError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
        let mut buf = Self::new_zeroed(byte_len)?;
        buf.as_f32_slice_mut()?.copy_from_slice(src);
        Ok(buf)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to `self.cap >= self.len` zero-initialized bytes
        // valid for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access to `self.len <= self.cap`
        // initialized bytes rooted at `self.ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_f32_slice(&self) -> Result<&[f32], TensorError> {
        if !self.len.is_multiple_of(4) {
            return Err(TensorError::InvalidDimension(format!(
                "buffer byte length {} is not a multiple of 4 for f32 view",
                self.len
            )));
        }
        // SAFETY: `self.ptr` is 64-byte aligned (multiple of `align_of::<f32>() == 4`),
        // all `self.len` bytes are initialized, and all 32-bit patterns are valid `f32`.
        let slice =
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<f32>(), self.len / 4) };
        Ok(slice)
    }

    #[inline]
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], TensorError> {
        if !self.len.is_multiple_of(4) {
            return Err(TensorError::InvalidDimension(format!(
                "buffer byte length {} is not a multiple of 4 for f32 view",
                self.len
            )));
        }
        // SAFETY: `&mut self` guarantees exclusive access, `self.ptr` is 64-byte aligned,
        // and all `self.len` bytes are initialized.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<f32>(), self.len / 4)
        };
        Ok(slice)
    }
}

impl Clone for AlignedBuf {
    fn clone(&self) -> Self {
        Self::from_bytes(self.as_bytes()).expect("AlignedBuf clone allocation failed")
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field("aligned_64", &self.addr().is_multiple_of(CACHE_LINE_ALIGN))
            .finish()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` in `AlignedBuf::new_zeroed`
        // and has not been deallocated yet.
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Pre-allocated 64-byte aligned bump arena reset in $O(1)$ between inference iterations
/// so forward passes perform **zero** heap allocations for intermediate activations.
///
/// Sub-allocations advance `cursor` monotonically and return strictly disjoint slices;
/// `reset(&mut self)` requires exclusive `&mut self`, statically proving at compile time
/// that no previously borrowed arena slices remain live.
pub struct AlignedScratchArena {
    ptr: NonNull<u8>,
    capacity: usize,
    layout: Layout,
    cursor: Cell<usize>,
    alloc_count: Cell<usize>,
}

// SAFETY: `AlignedScratchArena` owns its backing buffer and is only sent across threads
// by value (`!Sync` due to `Cell<usize>` prevents concurrent unsynchronized bump access).
unsafe impl Send for AlignedScratchArena {}

impl AlignedScratchArena {
    pub fn new(capacity_bytes: usize) -> Result<Self, TensorError> {
        let capacity = capacity_bytes
            .max(CACHE_LINE_ALIGN)
            .checked_next_multiple_of(CACHE_LINE_ALIGN)
            .ok_or(TensorError::AllocationFailed {
                size: capacity_bytes,
                align: CACHE_LINE_ALIGN,
            })?;
        let layout = Layout::from_size_align(capacity, CACHE_LINE_ALIGN).map_err(|_| {
            TensorError::AllocationFailed {
                size: capacity,
                align: CACHE_LINE_ALIGN,
            }
        })?;
        // SAFETY: `capacity >= 64 > 0` and `CACHE_LINE_ALIGN == 64` is a power of two.
        // Zero-initializing once ensures every byte in `[0..capacity)` is initialized forever.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or(TensorError::AllocationFailed {
            size: capacity,
            align: CACHE_LINE_ALIGN,
        })?;
        Ok(Self {
            ptr,
            capacity,
            layout,
            cursor: Cell::new(0),
            alloc_count: Cell::new(0),
        })
    }

    /// Resets the bump cursor to 0 in $O(1)$ with zero syscalls or deallocations.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor.set(0);
        self.alloc_count.set(0);
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.cursor.get()
    }

    #[inline]
    pub fn remaining_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.cursor.get())
    }

    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.alloc_count.get()
    }

    #[inline]
    pub fn base_addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Allocates a disjoint 64-byte aligned mutable byte slice from the arena.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_bytes(&self, byte_len: usize) -> Result<&mut [u8], TensorError> {
        let cur = self.cursor.get();
        let start =
            cur.checked_next_multiple_of(CACHE_LINE_ALIGN)
                .ok_or(TensorError::ArenaExhausted {
                    requested: byte_len,
                    available: self.capacity.saturating_sub(cur),
                    capacity: self.capacity,
                })?;
        let end = start
            .checked_add(byte_len)
            .ok_or(TensorError::ArenaExhausted {
                requested: byte_len,
                available: self.capacity.saturating_sub(start),
                capacity: self.capacity,
            })?;
        if end > self.capacity {
            return Err(TensorError::ArenaExhausted {
                requested: byte_len,
                available: self.capacity.saturating_sub(start),
                capacity: self.capacity,
            });
        }
        // Advance cursor past `end` (at least 1 byte if `byte_len == 0` to keep zero-sized pointers distinct)
        let next_cursor = if byte_len == 0 {
            (start + CACHE_LINE_ALIGN).min(self.capacity)
        } else {
            end
        };
        self.cursor.set(next_cursor);
        self.alloc_count.set(self.alloc_count.get() + 1);

        // SAFETY:
        // 1. `start` is a multiple of 64 and `start + byte_len <= self.capacity`.
        // 2. Monotonic `cursor` progression guarantees `[start .. start + byte_len)` is
        //    strictly disjoint from all other slices handed out before `&mut self.reset()`.
        // 3. `sub_ptr` is derived directly from `self.ptr` (root allocation provenance),
        //    and all bytes were zero-initialized at arena creation.
        unsafe {
            let sub_ptr = self.ptr.as_ptr().add(start);
            Ok(std::slice::from_raw_parts_mut(sub_ptr, byte_len))
        }
    }

    /// Allocates a disjoint 64-byte aligned `&mut [f32]` slice from the arena.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_f32(&self, count: usize) -> Result<&mut [f32], TensorError> {
        let byte_len =
            count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(TensorError::ArenaExhausted {
                    requested: usize::MAX,
                    available: self.remaining_bytes(),
                    capacity: self.capacity,
                })?;
        let bytes = self.alloc_bytes(byte_len)?;
        // SAFETY: `bytes` is 64-byte aligned (`64 % align_of::<f32>() == 0`), exclusively
        // borrowed over `count * 4` initialized bytes, and every 32-bit pattern is a valid `f32`.
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr().cast::<f32>(),
                count,
            ))
        }
    }

    /// Allocates a disjoint 64-byte aligned `&mut [f32]` slice zeroed to `0.0`.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_f32_zeroed(&self, count: usize) -> Result<&mut [f32], TensorError> {
        let out = self.alloc_f32(count)?;
        out.fill(0.0);
        Ok(out)
    }
}

impl Drop for AlignedScratchArena {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` in `AlignedScratchArena::new`
        // and `&mut self` guarantees no borrowed sub-slices exist.
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Zero-copy backing store for binary model files (`memmap2::Mmap` on native targets,
/// 64-byte `AlignedBuf` on `wasm32` / `miri` / in-memory loads).
pub enum ByteSource {
    #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
    Mmap(memmap2::Mmap),
    Aligned(AlignedBuf),
}

impl ByteSource {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        Ok(Self::Aligned(AlignedBuf::from_bytes(bytes)?))
    }

    pub fn from_aligned(buf: AlignedBuf) -> Self {
        Self::Aligned(buf)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_file(path: &Path) -> Result<Self, TensorError> {
        #[cfg(not(miri))]
        {
            let file = std::fs::File::open(path)?;
            let meta = file.metadata()?;
            if meta.len() == 0 {
                return Ok(Self::Aligned(AlignedBuf::new_zeroed(0)?));
            }
            // SAFETY: File is opened read-only for immutable weight inspection; model files
            // are static immutable artifacts during engine execution.
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            Ok(Self::Mmap(mmap))
        }
        #[cfg(miri)]
        {
            let data = std::fs::read(path)?;
            Self::from_bytes(&data)
        }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            #[cfg(all(not(target_arch = "wasm32"), not(miri)))]
            Self::Mmap(m) => m.as_ref(),
            Self::Aligned(b) => b.as_bytes(),
        }
    }
}

/// Helper to check whether an `f32` slice is safely reinterpret-castable from a byte slice.
#[inline]
pub fn try_cast_bytes_to_f32(bytes: &[u8]) -> Result<&[f32], TensorError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(TensorError::InvalidDimension(format!(
            "byte slice len {} not divisible by 4",
            bytes.len()
        )));
    }
    if bytes.is_empty() {
        return Ok(&[]);
    }
    let addr = bytes.as_ptr() as usize;
    if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(TensorError::AlignmentError {
            addr,
            required: std::mem::align_of::<f32>(),
        });
    }
    // SAFETY: Verified non-empty pointer is 4-byte aligned, `bytes.len()` is a multiple of 4,
    // and the lifetime matches `bytes`.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) })
}
