use crate::alloc::{try_cast_bytes_to_f32, AlignedBuf};
use crate::dtype::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, DType};
use crate::error::TensorError;
use crate::quant::{
    dequantize_row_q4_0, dequantize_row_q8_0, quantize_matrix_q4_0, quantize_matrix_q8_0,
};

pub const MAX_RANK: usize = 4;

#[inline]
pub fn compute_contiguous_strides(
    dims: &[usize],
) -> Result<([usize; MAX_RANK], usize), TensorError> {
    if dims.is_empty() || dims.len() > MAX_RANK {
        return Err(TensorError::InvalidDimension(format!(
            "tensor rank must be in 1..={MAX_RANK}, got {}",
            dims.len()
        )));
    }
    let rank = dims.len();
    let mut shape_arr = [1usize; MAX_RANK];
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(TensorError::InvalidDimension(
                "zero-sized dimension is not permitted".into(),
            ));
        }
        shape_arr[i] = d;
    }

    let mut strides = [0usize; MAX_RANK];
    strides[rank - 1] = 1;
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(shape_arr[i + 1])
            .ok_or_else(|| TensorError::InvalidDimension("stride product overflow".into()))?;
    }
    let numel = strides[0]
        .checked_mul(shape_arr[0])
        .ok_or_else(|| TensorError::InvalidDimension("numel product overflow".into()))?;
    Ok((strides, numel))
}

/// Zero-copy up-to-4D view over raw byte or quantized storage.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    data: &'a [u8],
    dtype: DType,
    shape: [usize; MAX_RANK],
    strides: [usize; MAX_RANK],
    rank: usize,
    elem_offset: usize,
}

impl<'a> TensorView<'a> {
    pub fn from_bytes(data: &'a [u8], dtype: DType, dims: &[usize]) -> Result<Self, TensorError> {
        let (strides, numel) = compute_contiguous_strides(dims)?;
        let rank = dims.len();
        let mut shape = [1usize; MAX_RANK];
        shape[..rank].copy_from_slice(dims);

        let expected_bytes = if dtype.is_quantized() {
            let cols = dims[rank - 1];
            let rows = numel / cols;
            dtype.matrix_byte_size(rows, cols)?
        } else {
            numel
                .checked_mul(dtype.block_bytes())
                .ok_or_else(|| TensorError::InvalidDimension("byte size overflow".into()))?
        };

        if data.len() != expected_bytes {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected_bytes],
                actual: vec![data.len()],
                context: "TensorView::from_bytes",
            });
        }

        Ok(Self {
            data,
            dtype,
            shape,
            strides,
            rank,
            elem_offset: 0,
        })
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.dims().iter().copied().product()
    }

    #[inline]
    pub fn raw_bytes(&self) -> &'a [u8] {
        self.data
    }

    #[inline]
    pub fn check_aligned(&self, align: usize) -> Result<(), TensorError> {
        let addr = self.data.as_ptr() as usize;
        if !addr.is_multiple_of(align) {
            return Err(TensorError::AlignmentError {
                addr,
                required: align,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        if let Ok((expected, _)) = compute_contiguous_strides(self.dims()) {
            self.strides[..self.rank] == expected[..self.rank]
        } else {
            false
        }
    }

    /// Zero-copy slice along `axis` in `[start, end)`.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, TensorError> {
        if axis >= self.rank {
            return Err(TensorError::InvalidDimension(format!(
                "slice axis {axis} out of bounds for rank {}",
                self.rank
            )));
        }
        let dim_len = self.shape[axis];
        if start >= end || end > dim_len {
            return Err(TensorError::InvalidDimension(format!(
                "invalid slice [{start}..{end}) on axis {axis} of len {dim_len}"
            )));
        }
        if self.dtype.is_quantized() && axis == self.rank - 1 {
            return Err(TensorError::UnsupportedDType(
                "cannot sub-slice innermost quantized block axis without dequantizing".into(),
            ));
        }

        let mut next = *self;
        next.shape[axis] = end - start;

        if self.dtype.is_quantized() {
            // Outer-axis contiguous byte slice
            if !self.is_contiguous() || axis != 0 {
                return Err(TensorError::UnsupportedDType(
                    "quantized slice requires contiguous outer axis 0".into(),
                ));
            }
            let cols = self.shape[self.rank - 1];
            let rows_per_outer = self.strides[0] / cols;
            let row_bytes = self.dtype.row_byte_size(cols)?;
            let byte_stride = rows_per_outer * row_bytes;
            next.data = &self.data[start * byte_stride..end * byte_stride];
        } else {
            next.elem_offset = self
                .elem_offset
                .checked_add(start * self.strides[axis])
                .ok_or_else(|| TensorError::InvalidDimension("slice offset overflow".into()))?;
        }
        Ok(next)
    }

    /// Zero-copy reshape onto `new_dims` (requires contiguous layout).
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::InvalidDimension(
                "cannot zero-copy reshape a non-contiguous view".into(),
            ));
        }
        let (new_strides, new_numel) = compute_contiguous_strides(new_dims)?;
        if new_numel != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: vec![new_numel],
                context: "TensorView::reshape",
            });
        }
        if self.dtype.is_quantized() && new_dims[new_dims.len() - 1] != self.shape[self.rank - 1] {
            return Err(TensorError::InvalidDimension(
                "quantized reshape must preserve innermost row dimension".into(),
            ));
        }
        let mut shape = [1usize; MAX_RANK];
        shape[..new_dims.len()].copy_from_slice(new_dims);
        Ok(Self {
            data: self.data,
            dtype: self.dtype,
            shape,
            strides: new_strides,
            rank: new_dims.len(),
            elem_offset: self.elem_offset,
        })
    }

    /// Zero-copy transpose swapping `dim0` and `dim1` in $O(1)$ metadata operations.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        if dim0 >= self.rank || dim1 >= self.rank {
            return Err(TensorError::InvalidDimension(format!(
                "transpose({dim0}, {dim1}) out of bounds for rank {}",
                self.rank
            )));
        }
        if self.dtype.is_quantized() && dim0 != dim1 {
            return Err(TensorError::UnsupportedDType(
                "zero-copy strided transpose is unsupported on block-quantized tensors".into(),
            ));
        }
        let mut next = *self;
        next.shape.swap(dim0, dim1);
        next.strides.swap(dim0, dim1);
        Ok(next)
    }

    /// Reads a single element as `f32` respecting arbitrary strided views.
    pub fn get_f32(&self, indices: &[usize]) -> Result<f32, TensorError> {
        if indices.len() != self.rank {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: indices.to_vec(),
                context: "TensorView::get_f32",
            });
        }
        if self.dtype.is_quantized() {
            let mut row_idx = 0usize;
            for (&idx, &dim) in indices[..self.rank - 1]
                .iter()
                .zip(&self.shape[..self.rank - 1])
            {
                if idx >= dim {
                    return Err(TensorError::InvalidDimension("index out of bounds".into()));
                }
                row_idx = row_idx * dim + idx;
            }
            let col = indices[self.rank - 1];
            let cols = self.shape[self.rank - 1];
            if col >= cols {
                return Err(TensorError::InvalidDimension("column out of bounds".into()));
            }
            let row_bytes = self.dtype.row_byte_size(cols)?;
            let row_slice = &self.data[row_idx * row_bytes..(row_idx + 1) * row_bytes];
            let mut decoded = vec![0.0f32; cols];
            match self.dtype {
                DType::Q8_0 => dequantize_row_q8_0(row_slice, &mut decoded)?,
                DType::Q4_0 => dequantize_row_q4_0(row_slice, &mut decoded)?,
                _ => unreachable!(),
            }
            return Ok(decoded[col]);
        }

        let mut flat_idx = self.elem_offset;
        for (axis, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[axis] {
                return Err(TensorError::InvalidDimension(format!(
                    "index {idx} out of bounds for dim {}",
                    self.shape[axis]
                )));
            }
            flat_idx += idx * self.strides[axis];
        }

        Ok(self.read_scalar_at(flat_idx))
    }

    #[inline]
    fn read_scalar_at(&self, elem_idx: usize) -> f32 {
        match self.dtype {
            DType::F32 => {
                let off = elem_idx * 4;
                f32::from_le_bytes([
                    self.data[off],
                    self.data[off + 1],
                    self.data[off + 2],
                    self.data[off + 3],
                ])
            }
            DType::F16 => {
                let off = elem_idx * 2;
                f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]))
            }
            DType::BF16 => {
                let off = elem_idx * 2;
                bf16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]))
            }
            DType::I8 => (self.data[elem_idx] as i8) as f32,
            DType::U8 => self.data[elem_idx] as f32,
            DType::Q8_0 | DType::Q4_0 => unreachable!("handled in get_f32"),
        }
    }

    /// Zero-copy `&[f32]` view when `dtype == F32`, contiguous, and 4-byte aligned.
    pub fn as_f32_slice(&self) -> Result<&'a [f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::UnsupportedDType(format!(
                "as_f32_slice requires F32, got {:?}",
                self.dtype
            )));
        }
        if !self.is_contiguous() {
            return Err(TensorError::InvalidDimension(
                "as_f32_slice requires contiguous tensor view".into(),
            ));
        }
        let start_byte = self.elem_offset * 4;
        let end_byte = start_byte + self.numel() * 4;
        try_cast_bytes_to_f32(&self.data[start_byte..end_byte])
    }

    /// Materializes or dequantizes any tensor view (including strided/transposed/F16/BF16/Q8_0/Q4_0)
    /// into a contiguous `Vec<f32>`.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, TensorError> {
        let total = self.numel();
        let mut out = vec![0.0f32; total];
        if self.dtype.is_quantized() {
            let cols = self.shape[self.rank - 1];
            let rows = total / cols;
            let row_bytes = self.dtype.row_byte_size(cols)?;
            for r in 0..rows {
                let src = &self.data[r * row_bytes..(r + 1) * row_bytes];
                let dst = &mut out[r * cols..(r + 1) * cols];
                match self.dtype {
                    DType::Q8_0 => dequantize_row_q8_0(src, dst)?,
                    DType::Q4_0 => dequantize_row_q4_0(src, dst)?,
                    _ => unreachable!(),
                }
            }
            return Ok(out);
        }

        if self.is_contiguous() {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = self.read_scalar_at(self.elem_offset + i);
            }
            return Ok(out);
        }

        // Arbitrary strided N-D traversal
        let mut coords = [0usize; MAX_RANK];
        for slot in &mut out {
            let mut flat = self.elem_offset;
            for (&coord, &stride) in coords[..self.rank].iter().zip(&self.strides[..self.rank]) {
                flat += coord * stride;
            }
            *slot = self.read_scalar_at(flat);
            for d in (0..self.rank).rev() {
                coords[d] += 1;
                if coords[d] < self.shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
        Ok(out)
    }
}

/// Owned 64-byte cache-aligned tensor buffer (`TensorBuf`).
#[derive(Clone, Debug)]
pub struct TensorBuf {
    buf: AlignedBuf,
    dtype: DType,
    shape: [usize; MAX_RANK],
    rank: usize,
}

impl TensorBuf {
    pub fn zeros_f32(dims: &[usize]) -> Result<Self, TensorError> {
        let (_, numel) = compute_contiguous_strides(dims)?;
        let buf = AlignedBuf::new_zeroed(numel * 4)?;
        let mut shape = [1usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            buf,
            dtype: DType::F32,
            shape,
            rank: dims.len(),
        })
    }

    pub fn from_f32_slice(dims: &[usize], data: &[f32]) -> Result<Self, TensorError> {
        let (_, numel) = compute_contiguous_strides(dims)?;
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected: vec![numel],
                actual: vec![data.len()],
                context: "TensorBuf::from_f32_slice",
            });
        }
        let buf = AlignedBuf::from_f32_slice(data)?;
        let mut shape = [1usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            buf,
            dtype: DType::F32,
            shape,
            rank: dims.len(),
        })
    }

    pub fn from_f32_as_dtype(
        dims: &[usize],
        data: &[f32],
        target_dtype: DType,
    ) -> Result<Self, TensorError> {
        let (_, numel) = compute_contiguous_strides(dims)?;
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected: vec![numel],
                actual: vec![data.len()],
                context: "TensorBuf::from_f32_as_dtype",
            });
        }
        let cols = dims[dims.len() - 1];
        let rows = numel / cols;
        let buf = match target_dtype {
            DType::F32 => AlignedBuf::from_f32_slice(data)?,
            DType::F16 => {
                let mut raw = AlignedBuf::new_zeroed(numel * 2)?;
                let bytes = raw.as_bytes_mut();
                for (i, &v) in data.iter().enumerate() {
                    let b = f32_to_f16(v).to_le_bytes();
                    bytes[i * 2] = b[0];
                    bytes[i * 2 + 1] = b[1];
                }
                raw
            }
            DType::BF16 => {
                let mut raw = AlignedBuf::new_zeroed(numel * 2)?;
                let bytes = raw.as_bytes_mut();
                for (i, &v) in data.iter().enumerate() {
                    let b = f32_to_bf16(v).to_le_bytes();
                    bytes[i * 2] = b[0];
                    bytes[i * 2 + 1] = b[1];
                }
                raw
            }
            DType::Q8_0 => quantize_matrix_q8_0(data, rows, cols)?,
            DType::Q4_0 => quantize_matrix_q4_0(data, rows, cols)?,
            DType::I8 => {
                let mut raw = AlignedBuf::new_zeroed(numel)?;
                for (dst, &v) in raw.as_bytes_mut().iter_mut().zip(data.iter()) {
                    *dst = (v.round().clamp(-128.0, 127.0) as i8) as u8;
                }
                raw
            }
            DType::U8 => {
                let mut raw = AlignedBuf::new_zeroed(numel)?;
                for (dst, &v) in raw.as_bytes_mut().iter_mut().zip(data.iter()) {
                    *dst = v.round().clamp(0.0, 255.0) as u8;
                }
                raw
            }
        };
        let mut shape = [1usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            buf,
            dtype: target_dtype,
            shape,
            rank: dims.len(),
        })
    }

    pub fn from_raw_aligned(
        buf: AlignedBuf,
        dtype: DType,
        dims: &[usize],
    ) -> Result<Self, TensorError> {
        let _ = TensorView::from_bytes(buf.as_bytes(), dtype, dims)?;
        let mut shape = [1usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            buf,
            dtype,
            shape,
            rank: dims.len(),
        })
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.dims().iter().copied().product()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.buf.as_bytes()
    }

    #[inline]
    pub fn as_f32_slice(&self) -> Result<&[f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::UnsupportedDType(format!(
                "expected F32, got {:?}",
                self.dtype
            )));
        }
        self.buf.as_f32_slice()
    }

    #[inline]
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::UnsupportedDType(format!(
                "expected F32, got {:?}",
                self.dtype
            )));
        }
        self.buf.as_f32_slice_mut()
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        TensorView::from_bytes(self.buf.as_bytes(), self.dtype, self.dims())
            .expect("TensorBuf invariant guarantees valid TensorView")
    }
}
