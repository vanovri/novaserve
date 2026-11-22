//! From-scratch `GGUF v3` binary parser and serializer (`100%` safe Rust outside `ByteSource`).
//!
//! Supports all 13 GGUF metadata value types (including nested arrays), `general.alignment`
//! padding calculation, tensor info validation, zero-copy `mmap` weight views, and lossless
//! round-trip serialization (`write_gguf_v3` / `serialize_gguf_v3`).

use crate::alloc::ByteSource;
use crate::dtype::DType;
use crate::error::TensorError;
use crate::tensor::TensorView;
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in LE
pub const GGUF_VERSION_V3: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const MAX_ARRAY_DEPTH: usize = 8;

/// All 13 canonical GGUF v3 metadata value variants.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufArray {
    pub elem_type: u32,
    pub elements: Vec<GgufValue>,
}

impl GgufValue {
    pub const fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U32(v) => Some(v),
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I32(v) if v >= 0 => Some(v as u32),
            Self::U16(v) => Some(v as u32),
            Self::U8(v) => Some(v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U64(v) => Some(v),
            Self::U32(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U8(v) => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            Self::Array(arr) if arr.elem_type == 8 => {
                let mut out = Vec::with_capacity(arr.elements.len());
                for el in &arr.elements {
                    out.push(el.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Raw GGML dimension order (`ne[0]` = innermost contiguous dimension).
    pub ggml_dims: Vec<u64>,
    /// Row-major shape (`ggml_dims` reversed, matching `[rows, cols]` convention).
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Relative byte offset from `tensor_data_offset`.
    pub rel_offset: usize,
    pub byte_len: usize,
}

pub struct GgufFile {
    source: ByteSource,
    alignment: usize,
    tensor_data_offset: usize,
    metadata: BTreeMap<String, GgufValue>,
    tensors: BTreeMap<String, GgufTensorInfo>,
}

struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    #[inline]
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], TensorError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| TensorError::CorruptGguf("cursor addition overflow".into()))?;
        if end > self.buf.len() {
            return Err(TensorError::CorruptGguf(format!(
                "unexpected EOF at pos={} reading {len} bytes (file_len={})",
                self.pos,
                self.buf.len()
            )));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, TensorError> {
        Ok(self.read_bytes(1)?[0])
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8, TensorError> {
        Ok(self.read_u8()? as i8)
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, TensorError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16, TensorError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, TensorError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, TensorError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, TensorError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64, TensorError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, TensorError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn read_f64(&mut self) -> Result<f64, TensorError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, TensorError> {
        let len_u64 = self.read_u64()?;
        let len = usize::try_from(len_u64)
            .map_err(|_| TensorError::CorruptGguf("string length overflows usize".into()))?;
        let raw = self.read_bytes(len)?;
        let s = std::str::from_utf8(raw)
            .map_err(|e| TensorError::CorruptGguf(format!("invalid UTF-8 string: {e}")))?;
        Ok(s.to_string())
    }

    fn read_value_of_type(&mut self, type_id: u32, depth: usize) -> Result<GgufValue, TensorError> {
        if depth > MAX_ARRAY_DEPTH {
            return Err(TensorError::CorruptGguf(format!(
                "GGUF array nesting exceeds max depth {MAX_ARRAY_DEPTH}"
            )));
        }
        match type_id {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => match self.read_u8()? {
                0 => Ok(GgufValue::Bool(false)),
                1 => Ok(GgufValue::Bool(true)),
                other => Err(TensorError::CorruptGguf(format!(
                    "invalid GGUF bool byte {other} (must be 0 or 1)"
                ))),
            },
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                if elem_type > 12 {
                    return Err(TensorError::CorruptGguf(format!(
                        "invalid GGUF array element type {elem_type}"
                    )));
                }
                let count_u64 = self.read_u64()?;
                let count = usize::try_from(count_u64).map_err(|_| {
                    TensorError::CorruptGguf("GGUF array count overflows usize".into())
                })?;
                // Every element consumes at least 1 byte in the stream: guard against OOM!
                if count > self.remaining() {
                    return Err(TensorError::CorruptGguf(format!(
                        "GGUF array claims {count} elements with only {} bytes remaining",
                        self.remaining()
                    )));
                }
                let mut elements = Vec::with_capacity(count);
                for _ in 0..count {
                    elements.push(self.read_value_of_type(elem_type, depth + 1)?);
                }
                Ok(GgufValue::Array(GgufArray {
                    elem_type,
                    elements,
                }))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(TensorError::CorruptGguf(format!(
                "unknown GGUF metadata value type {other}"
            ))),
        }
    }
}

#[inline]
fn align_up(offset: usize, alignment: usize) -> Result<usize, TensorError> {
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 65_536 {
        return Err(TensorError::CorruptGguf(format!(
            "invalid GGUF general.alignment {alignment}"
        )));
    }
    offset
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| TensorError::CorruptGguf("alignment padding overflow".into()))
}

impl GgufFile {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TensorError> {
        let source = ByteSource::open_file(path.as_ref())?;
        Self::from_source(source)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        let source = ByteSource::from_bytes(bytes)?;
        Self::from_source(source)
    }

    pub fn from_source(source: ByteSource) -> Result<Self, TensorError> {
        let buf = source.as_bytes();
        let mut reader = ByteReader::new(buf);

        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(TensorError::CorruptGguf(format!(
                "bad GGUF magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
            )));
        }

        let version = reader.read_u32()?;
        if version != GGUF_VERSION_V3 {
            return Err(TensorError::CorruptGguf(format!(
                "unsupported GGUF version {version}, expected {GGUF_VERSION_V3}"
            )));
        }

        let tensor_count_u64 = reader.read_u64()?;
        let kv_count_u64 = reader.read_u64()?;

        let tensor_count = usize::try_from(tensor_count_u64)
            .map_err(|_| TensorError::CorruptGguf("tensor_count overflows usize".into()))?;
        let kv_count = usize::try_from(kv_count_u64)
            .map_err(|_| TensorError::CorruptGguf("metadata_kv_count overflows usize".into()))?;

        // Minimum possible KV entry is 8 (empty key len) + 4 (type_id) + 1 (u8) = 13 bytes.
        if kv_count.saturating_mul(13) > reader.remaining() {
            return Err(TensorError::CorruptGguf(format!(
                "metadata_kv_count {kv_count} impossible for remaining {} bytes",
                reader.remaining()
            )));
        }
        // Minimum possible tensor info is 8 (name len) + 4 (n_dims) + 8 (dim0) + 4 (type) + 8 (offset) = 32 bytes.
        if tensor_count.saturating_mul(32) > reader.remaining() {
            return Err(TensorError::CorruptGguf(format!(
                "tensor_count {tensor_count} impossible for remaining {} bytes",
                reader.remaining()
            )));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = reader.read_string()?;
            if key.is_empty() {
                return Err(TensorError::CorruptGguf(
                    "empty metadata key in GGUF".into(),
                ));
            }
            if metadata.contains_key(&key) {
                return Err(TensorError::CorruptGguf(format!(
                    "duplicate metadata key '{key}'"
                )));
            }
            let val_type = reader.read_u32()?;
            let val = reader.read_value_of_type(val_type, 0)?;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => usize::try_from(*a)
                .map_err(|_| TensorError::CorruptGguf("alignment overflows usize".into()))?,
            Some(other) => {
                return Err(TensorError::CorruptGguf(format!(
                    "general.alignment must be u32, got {other:?}"
                )));
            }
            None => GGUF_DEFAULT_ALIGNMENT,
        };
        // Validate alignment immediately
        let _ = align_up(0, alignment)?;

        let mut raw_infos = Vec::with_capacity(tensor_count);
        let mut seen_tensors = std::collections::BTreeSet::new();
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            if name.is_empty() || !seen_tensors.insert(name.clone()) {
                return Err(TensorError::CorruptGguf(format!(
                    "empty or duplicate tensor name '{name}'"
                )));
            }
            let n_dims = reader.read_u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                return Err(TensorError::CorruptGguf(format!(
                    "tensor '{name}' invalid n_dimensions={n_dims}"
                )));
            }
            let mut ggml_dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ggml_dims.push(reader.read_u64()?);
            }
            let ggml_type = reader.read_u32()?;
            let dtype = DType::from_ggml_type_id(ggml_type).map_err(|e| {
                TensorError::CorruptGguf(format!("tensor '{name}' unsupported type: {e}"))
            })?;
            let byte_len = dtype.ggml_tensor_byte_size(&ggml_dims).map_err(|e| {
                TensorError::CorruptGguf(format!("tensor '{name}' invalid shape: {e}"))
            })?;
            let offset_u64 = reader.read_u64()?;
            let rel_offset = usize::try_from(offset_u64).map_err(|_| {
                TensorError::CorruptGguf(format!("tensor '{name}' offset overflow"))
            })?;
            if !rel_offset.is_multiple_of(alignment) {
                return Err(TensorError::CorruptGguf(format!(
                    "tensor '{name}' rel_offset {rel_offset} not aligned to {alignment}"
                )));
            }

            let mut shape: Vec<usize> = ggml_dims.iter().rev().map(|&d| d as usize).collect();
            if shape.is_empty() {
                shape.push(1);
            }

            raw_infos.push(GgufTensorInfo {
                name,
                ggml_dims,
                shape,
                dtype,
                rel_offset,
                byte_len,
            });
        }

        let tensor_data_offset = if tensor_count == 0 {
            reader.pos
        } else {
            align_up(reader.pos, alignment)?
        };
        if tensor_data_offset > buf.len() {
            return Err(TensorError::CorruptGguf(format!(
                "tensor_data_offset {tensor_data_offset} exceeds file len {}",
                buf.len()
            )));
        }

        // Verify zero out-of-bounds and zero overlapping tensor intervals
        let data_section_len = buf.len() - tensor_data_offset;
        raw_infos.sort_by_key(|info| (info.rel_offset, info.byte_len));
        let mut prev_end = 0usize;
        for info in &raw_infos {
            if info.rel_offset < prev_end {
                return Err(TensorError::CorruptGguf(format!(
                    "tensor '{}' at offset {} overlaps previous tensor ending at {prev_end}",
                    info.name, info.rel_offset
                )));
            }
            let end = info
                .rel_offset
                .checked_add(info.byte_len)
                .ok_or_else(|| TensorError::CorruptGguf("tensor end offset overflow".into()))?;
            if end > data_section_len {
                return Err(TensorError::CorruptGguf(format!(
                    "tensor '{}' span [{}..{end}) exceeds data section len {data_section_len}",
                    info.name, info.rel_offset
                )));
            }
            prev_end = end;
        }

        let mut tensors = BTreeMap::new();
        for info in raw_infos {
            tensors.insert(info.name.clone(), info);
        }

        Ok(Self {
            source,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
        })
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensor_infos(&self) -> &BTreeMap<String, GgufTensorInfo> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, TensorError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let start = self.tensor_data_offset + info.rel_offset;
        let end = start + info.byte_len;
        let slice = &self.source.as_bytes()[start..end];
        TensorView::from_bytes(slice, info.dtype, &info.shape)
    }
}

fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_gguf_value_payload(out: &mut Vec<u8>, val: &GgufValue) -> Result<(), TensorError> {
    match val {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => out.push(u8::from(*v)),
        GgufValue::String(s) => write_gguf_string(out, s),
        GgufValue::Array(arr) => {
            out.extend_from_slice(&arr.elem_type.to_le_bytes());
            out.extend_from_slice(&(arr.elements.len() as u64).to_le_bytes());
            for el in &arr.elements {
                if el.type_id() != arr.elem_type {
                    return Err(TensorError::CorruptGguf(format!(
                        "heterogeneous GGUF array element {:?} != {}",
                        el.type_id(),
                        arr.elem_type
                    )));
                }
                write_gguf_value_payload(out, el)?;
            }
        }
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
    }
    Ok(())
}

/// Serializes metadata and `TensorView`s into a spec-compliant `GGUF v3` binary image.
pub fn serialize_gguf_v3(
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[(&str, TensorView<'_>)],
) -> Result<Vec<u8>, TensorError> {
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(a)) => *a as usize,
        Some(GgufValue::U64(a)) => *a as usize,
        Some(_) => {
            return Err(TensorError::CorruptGguf(
                "general.alignment must be U32 or U64".into(),
            ));
        }
        None => GGUF_DEFAULT_ALIGNMENT,
    };
    let _ = align_up(0, alignment)?;

    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in metadata {
        if k.is_empty() {
            return Err(TensorError::CorruptGguf("empty metadata key".into()));
        }
        write_gguf_string(&mut out, k);
        out.extend_from_slice(&v.type_id().to_le_bytes());
        write_gguf_value_payload(&mut out, v)?;
    }

    let mut rel_offsets = Vec::with_capacity(tensors.len());
    let mut data_cursor = 0usize;
    let mut seen = std::collections::BTreeSet::new();

    for (name, view) in tensors {
        if name.is_empty() || !seen.insert(*name) {
            return Err(TensorError::CorruptGguf(format!(
                "empty or duplicate tensor name '{name}'"
            )));
        }
        if !view.is_contiguous() {
            return Err(TensorError::InvalidDimension(format!(
                "tensor '{name}' must be contiguous for GGUF export"
            )));
        }
        let ggml_type = view.dtype().to_ggml_type_id().ok_or_else(|| {
            TensorError::UnsupportedDType(format!(
                "dtype {:?} has no GGUF tensor type id",
                view.dtype()
            ))
        })?;
        let ggml_dims: Vec<u64> = view.dims().iter().rev().map(|&d| d as u64).collect();
        let expected_bytes = view.dtype().ggml_tensor_byte_size(&ggml_dims)?;
        if view.raw_bytes().len() != expected_bytes {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected_bytes],
                actual: vec![view.raw_bytes().len()],
                context: "serialize_gguf_v3",
            });
        }

        let aligned_rel = align_up(data_cursor, alignment)?;
        rel_offsets.push(aligned_rel);
        data_cursor = aligned_rel + expected_bytes;

        write_gguf_string(&mut out, name);
        out.extend_from_slice(&(ggml_dims.len() as u32).to_le_bytes());
        for d in &ggml_dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ggml_type.to_le_bytes());
        out.extend_from_slice(&(aligned_rel as u64).to_le_bytes());
    }

    if !tensors.is_empty() {
        let header_aligned = align_up(out.len(), alignment)?;
        out.resize(header_aligned, 0u8);

        let data_base = out.len();
        for ((_, view), &rel_off) in tensors.iter().zip(rel_offsets.iter()) {
            let target_pos = data_base + rel_off;
            if out.len() < target_pos {
                out.resize(target_pos, 0u8);
            }
            out.extend_from_slice(view.raw_bytes());
        }
    }

    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_gguf_v3<P: AsRef<Path>>(
    path: P,
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[(&str, TensorView<'_>)],
) -> Result<(), TensorError> {
    let bytes = serialize_gguf_v3(metadata, tensors)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
