//! Zero-copy `Safetensors` reader & serializer (`100%` safe Rust outside `ByteSource`).
//!
//! Enforces strict header-size bounds, non-overflowing shape products, exact byte-length
//! verification, and contiguous non-overlapping `[begin, end)` data offsets.

use crate::alloc::ByteSource;
use crate::dtype::DType;
use crate::error::TensorError;
use crate::tensor::{compute_contiguous_strides, TensorView};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct SafetensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

pub struct SafetensorsFile {
    source: ByteSource,
    data_offset: usize,
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, SafetensorEntry>,
}

impl SafetensorsFile {
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
        if buf.len() < 8 {
            return Err(TensorError::CorruptSafetensors(
                "buffer shorter than 8-byte header length".into(),
            ));
        }

        let header_len_u64 = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        if header_len_u64 == 0 {
            return Err(TensorError::CorruptSafetensors(
                "zero-length safetensors JSON header".into(),
            ));
        }
        let header_len = usize::try_from(header_len_u64).map_err(|_| {
            TensorError::CorruptSafetensors("header_len overflows platform usize".into())
        })?;
        if header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(TensorError::CorruptSafetensors(format!(
                "header_len {header_len} exceeds safety cap {MAX_SAFETENSORS_HEADER_BYTES}"
            )));
        }

        let data_offset = 8usize.checked_add(header_len).ok_or_else(|| {
            TensorError::CorruptSafetensors("8 + header_len arithmetic overflow".into())
        })?;
        if data_offset > buf.len() {
            return Err(TensorError::CorruptSafetensors(format!(
                "header extends past EOF (data_offset={data_offset}, file_len={})",
                buf.len()
            )));
        }

        let header_str = std::str::from_utf8(&buf[8..data_offset]).map_err(|e| {
            TensorError::CorruptSafetensors(format!("invalid UTF-8 in JSON header: {e}"))
        })?;
        let root: Value = serde_json::from_str(header_str).map_err(|e| {
            TensorError::CorruptSafetensors(format!("malformed safetensors JSON: {e}"))
        })?;
        let obj = root.as_object().ok_or_else(|| {
            TensorError::CorruptSafetensors("safetensors root JSON must be an object".into())
        })?;

        let data_section_len = buf.len() - data_offset;
        let mut metadata = BTreeMap::new();
        let mut entries = Vec::with_capacity(obj.len());

        for (key, val) in obj {
            if key == "__metadata__" {
                let meta_obj = val.as_object().ok_or_else(|| {
                    TensorError::CorruptSafetensors("__metadata__ must be a JSON object".into())
                })?;
                for (mk, mv) in meta_obj {
                    let s = mv.as_str().ok_or_else(|| {
                        TensorError::CorruptSafetensors(
                            "__metadata__ values must be JSON strings".into(),
                        )
                    })?;
                    metadata.insert(mk.clone(), s.to_string());
                }
                continue;
            }

            let t_obj = val.as_object().ok_or_else(|| {
                TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' descriptor must be an object"
                ))
            })?;
            let dtype_str = t_obj.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' missing string dtype"))
            })?;
            let dtype = DType::from_safetensors_name(dtype_str).map_err(|e| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' bad dtype: {e}"))
            })?;

            let shape_arr = t_obj
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    TensorError::CorruptSafetensors(format!("tensor '{key}' missing shape array"))
                })?;
            if shape_arr.is_empty() || shape_arr.len() > 4 {
                return Err(TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' unsupported rank {}",
                    shape_arr.len()
                )));
            }
            let mut shape = Vec::with_capacity(shape_arr.len());
            for dim_v in shape_arr {
                let d_u64 = dim_v.as_u64().ok_or_else(|| {
                    TensorError::CorruptSafetensors(format!(
                        "tensor '{key}' non-integer shape element"
                    ))
                })?;
                let d = usize::try_from(d_u64).map_err(|_| {
                    TensorError::CorruptSafetensors(format!("tensor '{key}' dim overflow"))
                })?;
                if d == 0 {
                    return Err(TensorError::CorruptSafetensors(format!(
                        "tensor '{key}' zero-sized dimension"
                    )));
                }
                shape.push(d);
            }

            let (_, numel) = compute_contiguous_strides(&shape).map_err(|e| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' stride overflow: {e}"))
            })?;
            let expected_bytes = numel.checked_mul(dtype.block_bytes()).ok_or_else(|| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' byte size overflow"))
            })?;

            let offsets_arr = t_obj
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    TensorError::CorruptSafetensors(format!("tensor '{key}' missing data_offsets"))
                })?;
            if offsets_arr.len() != 2 {
                return Err(TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' data_offsets must have length 2"
                )));
            }
            let begin_u64 = offsets_arr[0].as_u64().ok_or_else(|| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' non-u64 begin offset"))
            })?;
            let end_u64 = offsets_arr[1].as_u64().ok_or_else(|| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' non-u64 end offset"))
            })?;
            let begin = usize::try_from(begin_u64).map_err(|_| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' begin overflow"))
            })?;
            let end = usize::try_from(end_u64).map_err(|_| {
                TensorError::CorruptSafetensors(format!("tensor '{key}' end overflow"))
            })?;

            if begin > end {
                return Err(TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' inverted offsets [{begin}, {end}]"
                )));
            }
            if end - begin != expected_bytes {
                return Err(TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' byte span {} != expected {expected_bytes}",
                    end - begin
                )));
            }
            if end > data_section_len {
                return Err(TensorError::CorruptSafetensors(format!(
                    "tensor '{key}' end {end} exceeds data buffer len {data_section_len}"
                )));
            }

            entries.push(SafetensorEntry {
                name: key.clone(),
                dtype,
                shape,
                begin,
                end,
            });
        }

        // Enforce strict non-overlapping contiguous buffer coverage
        entries.sort_by_key(|e| (e.begin, e.end));
        let mut cursor = 0usize;
        for entry in &entries {
            if entry.begin != cursor {
                return Err(TensorError::CorruptSafetensors(format!(
                    "overlapping or gapped tensor '{}' at begin={}, expected {}",
                    entry.name, entry.begin, cursor
                )));
            }
            cursor = entry.end;
        }
        if cursor != data_section_len {
            return Err(TensorError::CorruptSafetensors(format!(
                "trailing unreferenced bytes in safetensors buffer: used {cursor}, total {data_section_len}"
            )));
        }

        let mut tensors = BTreeMap::new();
        for entry in entries {
            tensors.insert(entry.name.clone(), entry);
        }

        Ok(Self {
            source,
            data_offset,
            metadata,
            tensors,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, TensorError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound(name.to_string()))?;
        let raw =
            &self.source.as_bytes()[self.data_offset + entry.begin..self.data_offset + entry.end];
        TensorView::from_bytes(raw, entry.dtype, &entry.shape)
    }
}

/// Serializes named `TensorView`s and optional string metadata into canonical 8-byte-aligned
/// `Safetensors` binary bytes.
pub fn serialize_safetensors(
    tensors: &[(&str, TensorView<'_>)],
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<Vec<u8>, TensorError> {
    let mut header_map = Map::new();
    if let Some(meta) = metadata {
        if !meta.is_empty() {
            let mut m = Map::new();
            for (k, v) in meta {
                m.insert(k.clone(), Value::String(v.clone()));
            }
            header_map.insert("__metadata__".to_string(), Value::Object(m));
        }
    }

    let mut seen_names = std::collections::BTreeSet::new();
    let mut data_cursor = 0usize;
    for (name, view) in tensors {
        if *name == "__metadata__" || name.is_empty() {
            return Err(TensorError::CorruptSafetensors(format!(
                "reserved or empty tensor name '{name}'"
            )));
        }
        if !seen_names.insert(*name) {
            return Err(TensorError::CorruptSafetensors(format!(
                "duplicate tensor name '{name}'"
            )));
        }
        if !view.is_contiguous() {
            return Err(TensorError::InvalidDimension(format!(
                "tensor '{name}' must be contiguous to serialize into safetensors"
            )));
        }
        let dtype_str = view.dtype().safetensors_name()?;
        let byte_len = view.raw_bytes().len();
        let begin = data_cursor;
        let end = begin
            .checked_add(byte_len)
            .ok_or_else(|| TensorError::CorruptSafetensors("data offset overflow".into()))?;
        data_cursor = end;

        let mut desc = Map::new();
        desc.insert("dtype".to_string(), Value::String(dtype_str.to_string()));
        desc.insert(
            "shape".to_string(),
            Value::Array(view.dims().iter().map(|&d| Value::from(d as u64)).collect()),
        );
        desc.insert(
            "data_offsets".to_string(),
            Value::Array(vec![Value::from(begin as u64), Value::from(end as u64)]),
        );
        header_map.insert((*name).to_string(), Value::Object(desc));
    }

    let mut header_bytes = serde_json::to_vec(&Value::Object(header_map)).map_err(|e| {
        TensorError::CorruptSafetensors(format!("failed to serialize safetensors JSON: {e}"))
    })?;
    // Pad JSON header with ASCII spaces (`0x20`) to an 8-byte multiple so data starts 8-byte aligned!
    let pad = (8 - (header_bytes.len() % 8)) % 8;
    header_bytes.resize(header_bytes.len() + pad, b' ');

    let total_len = 8 + header_bytes.len() + data_cursor;
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for (_, view) in tensors {
        out.extend_from_slice(view.raw_bytes());
    }
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &[(&str, TensorView<'_>)],
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<(), TensorError> {
    let bytes = serialize_safetensors(tensors, metadata)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
