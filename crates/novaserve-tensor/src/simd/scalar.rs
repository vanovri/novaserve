use crate::dtype::{f16_to_f32, BYTES_Q4_0, BYTES_Q8_0, QK4_0, QK8_0};

#[inline]
pub fn vec_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        s0 = a[base].mul_add(b[base], s0);
        s1 = a[base + 1].mul_add(b[base + 1], s1);
        s2 = a[base + 2].mul_add(b[base + 2], s2);
        s3 = a[base + 3].mul_add(b[base + 3], s3);
    }

    let mut acc = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        acc = a[i].mul_add(b[i], acc);
    }
    acc
}

pub fn matmul_f32_row_tile_scalar(
    a_tile: &[f32],
    b: &[f32],
    tile_m: usize,
    k: usize,
    n: usize,
    out_tile: &mut [f32],
) {
    out_tile[..tile_m * n].fill(0.0);
    const KC: usize = 128;
    let mut p0 = 0;
    while p0 < k {
        let p_end = (p0 + KC).min(k);
        for i in 0..tile_m {
            let a_row = &a_tile[i * k..i * k + k];
            let c_row = &mut out_tile[i * n..(i + 1) * n];
            for (p, &a_ip) in a_row[p0..p_end].iter().enumerate() {
                let b_row = &b[(p0 + p) * n..(p0 + p + 1) * n];
                for j in 0..n {
                    c_row[j] = a_ip.mul_add(b_row[j], c_row[j]);
                }
            }
        }
        p0 = p_end;
    }
}

#[inline]
pub fn vec_dot_q8_0_q8_0_scalar(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    debug_assert_eq!(w_blocks.len(), num_blocks * BYTES_Q8_0);
    debug_assert_eq!(x_blocks.len(), num_blocks * BYTES_Q8_0);

    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q8_0;
        let x_off = b * BYTES_Q8_0;
        let dw = f16_to_f32(u16::from_le_bytes([w_blocks[w_off], w_blocks[w_off + 1]]));
        let dx = f16_to_f32(u16::from_le_bytes([x_blocks[x_off], x_blocks[x_off + 1]]));

        let mut isum: i32 = 0;
        for i in 0..QK8_0 {
            let qw = (w_blocks[w_off + 2 + i] as i8) as i32;
            let qx = (x_blocks[x_off + 2 + i] as i8) as i32;
            isum += qw * qx;
        }
        acc += (dw * dx) * (isum as f32);
    }
    acc
}

#[inline]
pub fn vec_dot_q4_0_q8_0_scalar(w_blocks: &[u8], x_blocks: &[u8], num_blocks: usize) -> f32 {
    debug_assert_eq!(w_blocks.len(), num_blocks * BYTES_Q4_0);
    debug_assert_eq!(x_blocks.len(), num_blocks * BYTES_Q8_0);
    let _ = QK4_0;

    let mut acc = 0.0f32;
    for b in 0..num_blocks {
        let w_off = b * BYTES_Q4_0;
        let x_off = b * BYTES_Q8_0;
        let dw = f16_to_f32(u16::from_le_bytes([w_blocks[w_off], w_blocks[w_off + 1]]));
        let dx = f16_to_f32(u16::from_le_bytes([x_blocks[x_off], x_blocks[x_off + 1]]));

        let mut isum: i32 = 0;
        for j in 0..16 {
            let byte = w_blocks[w_off + 2 + j];
            let q0 = ((byte & 0x0F) as i32) - 8;
            let q1 = ((byte >> 4) as i32) - 8;
            let x0 = (x_blocks[x_off + 2 + j] as i8) as i32;
            let x1 = (x_blocks[x_off + 2 + j + 16] as i8) as i32;
            isum += q0 * x0 + q1 * x1;
        }
        acc += (dw * dx) * (isum as f32);
    }
    acc
}

#[inline]
pub fn rmsnorm_row_scalar(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq = v.mul_add(v, sum_sq);
    }
    let inv_rms = 1.0 / (sum_sq / (dim as f32) + eps).sqrt();
    for i in 0..dim {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}
