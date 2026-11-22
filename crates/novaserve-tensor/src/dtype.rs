use crate::error::TensorError;
use serde::{Deserialize, Serialize};

pub const QK8_0: usize = 32;
pub const BYTES_Q8_0: usize = 34;

pub const QK4_0: usize = 32;
pub const BYTES_Q4_0: usize = 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    I8,
    U8,
}

impl DType {
    #[inline]
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_0)
    }

    #[inline]
    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q8_0 => QK8_0,
            Self::Q4_0 => QK4_0,
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::U8 => 1,
        }
    }

    #[inline]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => BYTES_Q8_0,
            Self::Q4_0 => BYTES_Q4_0,
            Self::I8 | Self::U8 => 1,
        }
    }

    /// Byte size for a single row of `cols` elements (zero-padding quantized tail blocks if `cols % 32 != 0`).
    #[inline]
    pub fn row_byte_size(self, cols: usize) -> Result<usize, TensorError> {
        let blk_elems = self.block_elements();
        let num_blocks = if cols == 0 {
            0
        } else {
            cols.div_ceil(blk_elems)
        };
        num_blocks
            .checked_mul(self.block_bytes())
            .ok_or_else(|| TensorError::InvalidDimension("row byte size overflow".into()))
    }

    /// Exact byte size for `rows` rows of `cols` elements.
    #[inline]
    pub fn matrix_byte_size(self, rows: usize, cols: usize) -> Result<usize, TensorError> {
        let row_bytes = self.row_byte_size(cols)?;
        rows.checked_mul(row_bytes)
            .ok_or_else(|| TensorError::InvalidDimension("matrix byte size overflow".into()))
    }

    /// Strict GGML tensor byte size requiring `inner_dim % block_elements == 0`.
    pub fn ggml_tensor_byte_size(self, ggml_dims: &[u64]) -> Result<usize, TensorError> {
        if ggml_dims.is_empty() {
            return Err(TensorError::InvalidDimension(
                "GGML tensor must have at least 1 dimension".into(),
            ));
        }
        let ne0 = usize::try_from(ggml_dims[0])
            .map_err(|_| TensorError::InvalidDimension("dimension exceeds usize".into()))?;
        let blk_elems = self.block_elements();
        if ne0 == 0 || !ne0.is_multiple_of(blk_elems) {
            return Err(TensorError::InvalidDimension(format!(
                "GGML inner dim {ne0} not divisible by block size {blk_elems} for {self:?}"
            )));
        }
        let mut outer_rows: usize = 1;
        for &d in &ggml_dims[1..] {
            let du = usize::try_from(d)
                .map_err(|_| TensorError::InvalidDimension("dimension exceeds usize".into()))?;
            if du == 0 {
                return Err(TensorError::InvalidDimension(
                    "GGML zero dimension is invalid".into(),
                ));
            }
            outer_rows = outer_rows.checked_mul(du).ok_or_else(|| {
                TensorError::InvalidDimension("GGML shape product overflow".into())
            })?;
        }
        let row_bytes = (ne0 / blk_elems)
            .checked_mul(self.block_bytes())
            .ok_or_else(|| TensorError::InvalidDimension("GGML row byte overflow".into()))?;
        outer_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| TensorError::InvalidDimension("GGML total byte overflow".into()))
    }

    pub fn safetensors_name(self) -> Result<&'static str, TensorError> {
        match self {
            Self::F32 => Ok("F32"),
            Self::F16 => Ok("F16"),
            Self::BF16 => Ok("BF16"),
            Self::I8 => Ok("I8"),
            Self::U8 => Ok("U8"),
            Self::Q8_0 | Self::Q4_0 => Err(TensorError::UnsupportedDType(format!(
                "{self:?} is a GGML quantized format; pack in GGUF or U8 safetensors"
            ))),
        }
    }

    pub fn from_safetensors_name(s: &str) -> Result<Self, TensorError> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I8" => Ok(Self::I8),
            "U8" => Ok(Self::U8),
            other => Err(TensorError::UnsupportedDType(other.to_string())),
        }
    }

    pub const fn to_ggml_type_id(self) -> Option<u32> {
        match self {
            Self::F32 => Some(0),
            Self::F16 => Some(1),
            Self::Q4_0 => Some(2),
            Self::Q8_0 => Some(8),
            Self::I8 => Some(24),
            Self::BF16 => Some(30),
            Self::U8 => None,
        }
    }

    pub fn from_ggml_type_id(id: u32) -> Result<Self, TensorError> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            24 => Ok(Self::I8),
            30 => Ok(Self::BF16),
            other => Err(TensorError::UnsupportedDType(format!("ggml_type={other}"))),
        }
    }
}

/// Pure-Rust IEEE-754 `binary16` (`f16`) -> `binary32` (`f32`) exact bit conversion.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            // Normalize subnormal f16 mantissa
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            f32::from_bits(sign | (e << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits(sign | 0x7F80_0000)
        } else {
            f32::from_bits(sign | 0x7FC0_0000 | (mant << 13))
        }
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Pure-Rust IEEE-754 `binary32` (`f32`) -> `binary16` (`f16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        return if mant == 0 {
            sign | 0x7C00
        } else {
            let payload = ((mant >> 13) as u16).max(1);
            sign | 0x7E00 | (payload & 0x01FF)
        };
    }

    let unbiased = exp - 127;
    if unbiased < -14 {
        if unbiased < -25 || exp == 0 {
            return sign;
        }
        let shift = (-14 - unbiased) as u32 + 13;
        let full_mant = 0x0080_0000u32 | mant;
        let q = full_mant >> shift;
        let rem = full_mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = (rem > halfway) || (rem == halfway && (q & 1) == 1);
        return sign | ((q + u32::from(round_up)) as u16);
    }

    if unbiased > 15 {
        return sign | 0x7C00;
    }

    let mut h_exp = (unbiased + 15) as u32;
    let q = mant >> 13;
    let rem = mant & 0x1FFF;
    let round_up = (rem > 0x1000) || (rem == 0x1000 && (q & 1) == 1);
    let mut h_mant = q + u32::from(round_up);
    if h_mant >= 0x0400 {
        h_mant = 0;
        h_exp += 1;
    }
    if h_exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((h_exp as u16) << 10) | (h_mant as u16)
}

/// Pure-Rust `bfloat16` -> `f32` conversion.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Pure-Rust `f32` -> `bfloat16` with round-to-nearest-even and NaN preservation.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // Quiet NaN
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFFu32.wrapping_add(lsb);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
