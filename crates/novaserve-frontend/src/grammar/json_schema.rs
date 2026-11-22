//! Draft-2020-12 JSON Schema -> Minimized `ByteDfa` Lowering Compiler + Structural Validator.

use crate::error::FrontendError;
use crate::grammar::dfa::{ByteDfa, ByteNfaBuilder, NfaFrag};
use serde_json::Value;
use std::collections::HashSet;

const MAX_SCHEMA_DEPTH: usize = 12;
const DEFAULT_UNBOUNDED_ARRAY_MAX: usize = 5;

/// Compiles a Draft-2020-12 JSON Schema (`serde_json::Value`) into an exact minimized `ByteDfa`.
pub fn compile_json_schema(schema: &Value) -> Result<ByteDfa, FrontendError> {
    let mut builder = ByteNfaBuilder::new();
    let ws_lead = opt_ws(&mut builder);
    let body = compile_subschema(schema, schema, &mut builder, 0)?;
    let ws_trail = opt_ws(&mut builder);
    let root = builder.concat(&[ws_lead, body, ws_trail]);
    let dfa = builder.build_minimized_dfa(root)?;
    if dfa.start_state() == crate::grammar::dfa::DEAD_STATE {
        return Err(FrontendError::SchemaCompile(
            "JSON schema compiled to an empty unsatisfiable language".into(),
        ));
    }
    Ok(dfa)
}

pub fn compile_json_schema_str(schema_json: &str) -> Result<ByteDfa, FrontendError> {
    let v: Value = serde_json::from_str(schema_json)?;
    compile_json_schema(&v)
}

fn opt_ws(b: &mut ByteNfaBuilder) -> NfaFrag {
    // Allow 0..=1 spaces/newlines around structural delimiters to permit both compact
    // and readable JSON while keeping state spaces tight.
    let ws1 = b.byte_ranges(&[(b'\t', b'\n'), (b' ', b' ')]);
    b.optional(ws1)
}

fn json_quoted_literal(b: &mut ByteNfaBuilder, literal: &str) -> NfaFrag {
    let encoded = serde_json::to_string(literal).expect("string serialization infallible");
    b.lit_bytes(encoded.as_bytes())
}

/// Single valid RFC 8259 + RFC 3629 JSON string character (unescaped safe UTF-8 OR legal escape).
fn json_string_char(b: &mut ByteNfaBuilder) -> NfaFrag {
    let s = b.add_state();
    let e = b.add_state();

    // 1. Safe unescaped printable ASCII: 0x20..=0x21, 0x23..=0x5B, 0x5D..=0x7E
    b.add_range(s, 0x20, 0x21, e);
    b.add_range(s, 0x23, 0x5B, e);
    b.add_range(s, 0x5D, 0x7E, e);

    // 2. Valid multi-byte RFC 3629 UTF-8 sequences
    let u2 = b.add_state();
    b.add_range(s, 0xC2, 0xDF, u2);
    b.add_range(u2, 0x80, 0xBF, e);

    let u3_e0 = b.add_state();
    let u3_tail = b.add_state();
    b.add_range(s, 0xE0, 0xE0, u3_e0);
    b.add_range(u3_e0, 0xA0, 0xBF, u3_tail);
    let u3_norm = b.add_state();
    b.add_range(s, 0xE1, 0xEC, u3_norm);
    b.add_range(s, 0xEE, 0xEF, u3_norm);
    b.add_range(u3_norm, 0x80, 0xBF, u3_tail);
    let u3_ed = b.add_state();
    b.add_range(s, 0xED, 0xED, u3_ed);
    b.add_range(u3_ed, 0x80, 0x9F, u3_tail);
    b.add_range(u3_tail, 0x80, 0xBF, e);

    let u4_tail2 = b.add_state();
    let u4_tail1 = b.add_state();
    let u4_f0 = b.add_state();
    b.add_range(s, 0xF0, 0xF0, u4_f0);
    b.add_range(u4_f0, 0x90, 0xBF, u4_tail2);
    let u4_mid = b.add_state();
    b.add_range(s, 0xF1, 0xF3, u4_mid);
    b.add_range(u4_mid, 0x80, 0xBF, u4_tail2);
    let u4_f4 = b.add_state();
    b.add_range(s, 0xF4, 0xF4, u4_f4);
    b.add_range(u4_f4, 0x80, 0x8F, u4_tail2);
    b.add_range(u4_tail2, 0x80, 0xBF, u4_tail1);
    b.add_range(u4_tail1, 0x80, 0xBF, e);

    // 3. Escape sequences: `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, plus non-surrogate `\uXXXX`
    let esc = b.add_state();
    b.add_range(s, b'\\', b'\\', esc);
    for &ch in b"\"\\/bfnrt" {
        b.add_range(esc, ch, ch, e);
    }

    // `\u` BMP non-surrogate hex (`[0-9a-ce-fA-CE-F] H H H` | `[dD] [0-7] H H`)
    let u_hex0 = b.add_state();
    b.add_range(esc, b'u', b'u', u_hex0);
    let h_rem3 = b.add_state();
    let h_rem2 = b.add_state();
    let h_rem1 = b.add_state();

    // First hex digit not 'd'/'D'
    for &(lo, hi) in &[
        (b'0', b'9'),
        (b'A', b'C'),
        (b'E', b'F'),
        (b'a', b'c'),
        (b'e', b'f'),
    ] {
        b.add_range(u_hex0, lo, hi, h_rem3);
    }
    // First hex digit 'd'/'D' -> second digit strictly '0'..'7' (excludes D800..DFFF surrogates!)
    let u_d_branch = b.add_state();
    b.add_range(u_hex0, b'd', b'd', u_d_branch);
    b.add_range(u_hex0, b'D', b'D', u_d_branch);
    b.add_range(u_d_branch, b'0', b'7', h_rem2);

    for &(src_st, dst_st) in &[(h_rem3, h_rem2), (h_rem2, h_rem1), (h_rem1, e)] {
        b.add_range(src_st, b'0', b'9', dst_st);
        b.add_range(src_st, b'A', b'F', dst_st);
        b.add_range(src_st, b'a', b'f', dst_st);
    }

    NfaFrag { start: s, out: e }
}

fn json_string_frag(
    b: &mut ByteNfaBuilder,
    min_len: Option<usize>,
    max_len: Option<usize>,
) -> NfaFrag {
    let q_open = b.lit_byte(b'"');
    let q_close = b.lit_byte(b'"');

    let min_l = min_len.unwrap_or(0).min(64);
    let body = match (min_l, max_len) {
        (0, None) => {
            let ch = json_string_char(b);
            b.zero_or_more(ch)
        }
        (m, None) => {
            let mut parts = Vec::with_capacity(m + 1);
            for _ in 0..m {
                parts.push(json_string_char(b));
            }
            let tail = json_string_char(b);
            parts.push(b.zero_or_more(tail));
            b.concat(&parts)
        }
        (m, Some(max_raw)) => {
            let n = max_raw.min(64).max(m);
            let mut parts = Vec::with_capacity(n);
            for _ in 0..m {
                parts.push(json_string_char(b));
            }
            for _ in m..n {
                let ch = json_string_char(b);
                parts.push(b.optional(ch));
            }
            b.concat(&parts)
        }
    };
    b.concat(&[q_open, body, q_close])
}

/// Exact RFC 8259 JSON integer `-?(0|[1-9][0-9]{0,11})` (fits cleanly in `i64`).
fn json_integer_frag(b: &mut ByteNfaBuilder) -> NfaFrag {
    let minus = b.lit_byte(b'-');
    let opt_minus = b.optional(minus);
    let zero = b.lit_byte(b'0');

    let nz_first = b.byte_ranges(&[(b'1', b'9')]);
    let mut digits = Vec::with_capacity(12);
    digits.push(nz_first);
    for _ in 0..11 {
        let d = b.byte_ranges(&[(b'0', b'9')]);
        digits.push(b.optional(d));
    }
    let nonzero_int = b.concat(&digits);
    let mag = b.alt(&[zero, nonzero_int]);
    b.concat(&[opt_minus, mag])
}

/// Exact RFC 8259 JSON number `-?(0|[1-9][0-9]{0,9})(\.[0-9]{1,6})?([eE][+-]?[0-9]{1,2})?`.
fn json_number_frag(b: &mut ByteNfaBuilder) -> NfaFrag {
    let int_part = json_integer_frag(b);

    // Optional fraction `\.[0-9]{1,6}`
    let dot = b.lit_byte(b'.');
    let f0 = b.byte_ranges(&[(b'0', b'9')]);
    let mut frac_parts = vec![dot, f0];
    for _ in 0..5 {
        let fd = b.byte_ranges(&[(b'0', b'9')]);
        frac_parts.push(b.optional(fd));
    }
    let frac = b.concat(&frac_parts);
    let opt_frac = b.optional(frac);

    // Optional bounded exponent `[eE][+-]?[0-9]{1,2}` (prevents f64 Inf overflow!)
    let exp_ch = b.byte_ranges(&[(b'E', b'E'), (b'e', b'e')]);
    let sign = b.byte_ranges(&[(b'+', b'+'), (b'-', b'-')]);
    let opt_sign = b.optional(sign);
    let e0 = b.byte_ranges(&[(b'0', b'9')]);
    let e1_raw = b.byte_ranges(&[(b'0', b'9')]);
    let e1 = b.optional(e1_raw);
    let exp_seq = b.concat(&[exp_ch, opt_sign, e0, e1]);
    let opt_exp = b.optional(exp_seq);

    b.concat(&[int_part, opt_frac, opt_exp])
}

fn json_boolean_frag(b: &mut ByteNfaBuilder) -> NfaFrag {
    let t = b.lit_bytes(b"true");
    let f = b.lit_bytes(b"false");
    b.alt(&[t, f])
}

fn json_null_frag(b: &mut ByteNfaBuilder) -> NfaFrag {
    b.lit_bytes(b"null")
}

fn json_const_value_frag(b: &mut ByteNfaBuilder, val: &Value) -> Result<NfaFrag, FrontendError> {
    let serialized = serde_json::to_string(val)?;
    Ok(b.lit_bytes(serialized.as_bytes()))
}

fn resolve_ref<'a>(root: &'a Value, reference: &str) -> Result<&'a Value, FrontendError> {
    let ptr = reference.strip_prefix('#').ok_or_else(|| {
        FrontendError::SchemaCompile(format!(
            "only local JSON pointer $ref supported: {reference}"
        ))
    })?;
    root.pointer(ptr).ok_or_else(|| {
        FrontendError::SchemaCompile(format!("unresolved JSON schema $ref: {reference}"))
    })
}

fn compile_subschema(
    root: &Value,
    schema: &Value,
    b: &mut ByteNfaBuilder,
    depth: usize,
) -> Result<NfaFrag, FrontendError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(FrontendError::SchemaCompile(format!(
            "JSON schema nesting exceeds max depth {MAX_SCHEMA_DEPTH}"
        )));
    }

    let obj = schema
        .as_object()
        .ok_or_else(|| FrontendError::SchemaCompile("JSON schema node must be an object".into()))?;

    if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
        let target = resolve_ref(root, ref_str)?;
        return compile_subschema(root, target, b, depth + 1);
    }

    if let Some(const_val) = obj.get("const") {
        return json_const_value_frag(b, const_val);
    }

    if let Some(enum_arr) = obj.get("enum").and_then(Value::as_array) {
        if enum_arr.is_empty() {
            return Err(FrontendError::SchemaCompile("empty 'enum' array".into()));
        }
        let mut choices = Vec::with_capacity(enum_arr.len());
        for item in enum_arr {
            choices.push(json_const_value_frag(b, item)?);
        }
        return Ok(b.alt(&choices));
    }

    if let Some(any_of) = obj
        .get("anyOf")
        .or_else(|| obj.get("oneOf"))
        .and_then(Value::as_array)
    {
        if any_of.is_empty() {
            return Err(FrontendError::SchemaCompile(
                "empty 'anyOf'/'oneOf' array".into(),
            ));
        }
        let mut branches = Vec::with_capacity(any_of.len());
        for sub in any_of {
            branches.push(compile_subschema(root, sub, b, depth + 1)?);
        }
        return Ok(b.alt(&branches));
    }

    // Handle `"type": ["string", "null"]` union array or single `"type": "..."`
    if let Some(type_val) = obj.get("type") {
        if let Some(type_arr) = type_val.as_array() {
            let mut branches = Vec::with_capacity(type_arr.len());
            for tv in type_arr {
                let ts = tv.as_str().ok_or_else(|| {
                    FrontendError::SchemaCompile("non-string element in 'type' array".into())
                })?;
                branches.push(compile_single_type(root, obj, ts, b, depth)?);
            }
            return Ok(b.alt(&branches));
        } else if let Some(ts) = type_val.as_str() {
            return compile_single_type(root, obj, ts, b, depth);
        } else {
            return Err(FrontendError::SchemaCompile(
                "'type' must be string or string array".into(),
            ));
        }
    }

    // Fallback inference if `"properties"` or `"items"` is present without explicit `"type"`
    if obj.contains_key("properties") {
        return compile_single_type(root, obj, "object", b, depth);
    }
    if obj.contains_key("items") {
        return compile_single_type(root, obj, "array", b, depth);
    }

    // Unconstrained primitive value fallback
    let s_frag = json_string_frag(b, None, Some(16));
    let n_frag = json_number_frag(b);
    let b_frag = json_boolean_frag(b);
    let null_frag = json_null_frag(b);
    Ok(b.alt(&[s_frag, n_frag, b_frag, null_frag]))
}

fn compile_single_type(
    root: &Value,
    obj: &serde_json::Map<String, Value>,
    type_name: &str,
    b: &mut ByteNfaBuilder,
    depth: usize,
) -> Result<NfaFrag, FrontendError> {
    match type_name {
        "null" => Ok(json_null_frag(b)),
        "boolean" => Ok(json_boolean_frag(b)),
        "integer" => Ok(json_integer_frag(b)),
        "number" => Ok(json_number_frag(b)),
        "string" => {
            let min_len = obj
                .get("minLength")
                .and_then(Value::as_u64)
                .map(|v| v as usize);
            let max_len = obj
                .get("maxLength")
                .and_then(Value::as_u64)
                .map(|v| v as usize);
            Ok(json_string_frag(b, min_len, max_len))
        }
        "array" => compile_array_schema(root, obj, b, depth),
        "object" => compile_object_schema(root, obj, b, depth),
        other => Err(FrontendError::SchemaCompile(format!(
            "unsupported JSON schema type '{other}'"
        ))),
    }
}

fn compile_array_schema(
    root: &Value,
    obj: &serde_json::Map<String, Value>,
    b: &mut ByteNfaBuilder,
    depth: usize,
) -> Result<NfaFrag, FrontendError> {
    let min_items = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
    let max_items = obj
        .get("maxItems")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or_else(|| min_items.max(DEFAULT_UNBOUNDED_ARRAY_MAX));

    if max_items < min_items {
        return Err(FrontendError::SchemaCompile(format!(
            "array maxItems ({max_items}) < minItems ({min_items})"
        )));
    }

    let default_item = serde_json::json!({ "type": "string" });
    let item_schema = obj.get("items").unwrap_or(&default_item);

    let open_br = b.lit_byte(b'[');
    let close_br = b.lit_byte(b']');

    if max_items == 0 {
        let ws = opt_ws(b);
        return Ok(b.concat(&[open_br, ws, close_br]));
    }

    // Build non-empty sequence of `[1 ..= max_items]` items
    let ws_in = opt_ws(b);
    let first_item = compile_subschema(root, item_schema, b, depth + 1)?;
    let mut seq = vec![ws_in, first_item];

    let req_tail = min_items.saturating_sub(1);
    let opt_tail = max_items - min_items.max(1);

    for _ in 0..req_tail {
        let ws_pre = opt_ws(b);
        let comma = b.lit_byte(b',');
        let ws_post = opt_ws(b);
        let elem = compile_subschema(root, item_schema, b, depth + 1)?;
        seq.push(b.concat(&[ws_pre, comma, ws_post, elem]));
    }
    for _ in 0..opt_tail {
        let ws_pre = opt_ws(b);
        let comma = b.lit_byte(b',');
        let ws_post = opt_ws(b);
        let elem = compile_subschema(root, item_schema, b, depth + 1)?;
        let comma_elem = b.concat(&[ws_pre, comma, ws_post, elem]);
        seq.push(b.optional(comma_elem));
    }

    let ws_out = opt_ws(b);
    seq.push(ws_out);
    let non_empty_body = b.concat(&seq);
    let non_empty_arr = b.concat(&[open_br, non_empty_body, close_br]);

    if min_items == 0 {
        let ob = b.lit_byte(b'[');
        let ws = opt_ws(b);
        let cb = b.lit_byte(b']');
        let empty_arr = b.concat(&[ob, ws, cb]);
        Ok(b.alt(&[empty_arr, non_empty_arr]))
    } else {
        Ok(non_empty_arr)
    }
}

fn compile_object_schema(
    root: &Value,
    obj: &serde_json::Map<String, Value>,
    b: &mut ByteNfaBuilder,
    depth: usize,
) -> Result<NfaFrag, FrontendError> {
    let required_set: HashSet<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let props_map = obj.get("properties").and_then(Value::as_object);
    if let Some(pm) = props_map {
        for &req_key in &required_set {
            if !pm.contains_key(req_key) {
                return Err(FrontendError::SchemaCompile(format!(
                    "required property '{req_key}' missing from 'properties'"
                )));
            }
        }
    } else if !required_set.is_empty() {
        return Err(FrontendError::SchemaCompile(
            "object has 'required' fields but no 'properties' map".into(),
        ));
    }

    let open_brace = b.lit_byte(b'{');
    let ws_open = opt_ws(b);
    let start_hdr = b.concat(&[open_brace, ws_open]);

    let Some(pm) = props_map else {
        let ws_close = opt_ws(b);
        let close_brace = b.lit_byte(b'}');
        return Ok(b.concat(&[start_hdr, ws_close, close_brace]));
    };

    if pm.is_empty() {
        let ws_close = opt_ws(b);
        let close_brace = b.lit_byte(b'}');
        return Ok(b.concat(&[start_hdr, ws_close, close_brace]));
    }

    // Linear 2P-state `(has_emitted == false, has_emitted == true)` spine
    let mut cur_false: Option<usize> = Some(start_hdr.out);
    let mut cur_true: Option<usize> = None;

    for (prop_name, prop_schema) in pm {
        let is_req = required_set.contains(prop_name.as_str());
        let next_true = b.add_state();

        if let Some(cf) = cur_false {
            // Transition without leading comma: `"key" : <val>`
            let k_lit = json_quoted_literal(b, prop_name);
            let ws1 = opt_ws(b);
            let colon = b.lit_byte(b':');
            let ws2 = opt_ws(b);
            let val_f = compile_subschema(root, prop_schema, b, depth + 1)?;
            let kv = b.concat(&[k_lit, ws1, colon, ws2, val_f]);
            b.add_eps(cf, kv.start);
            b.add_eps(kv.out, next_true);
        }

        if let Some(ct) = cur_true {
            // Transition WITH leading comma: `, "key" : <val>`
            let ws0 = opt_ws(b);
            let comma = b.lit_byte(b',');
            let ws_after_comma = opt_ws(b);
            let k_lit = json_quoted_literal(b, prop_name);
            let ws1 = opt_ws(b);
            let colon = b.lit_byte(b':');
            let ws2 = opt_ws(b);
            let val_f = compile_subschema(root, prop_schema, b, depth + 1)?;
            let ckv = b.concat(&[ws0, comma, ws_after_comma, k_lit, ws1, colon, ws2, val_f]);
            b.add_eps(ct, ckv.start);
            b.add_eps(ckv.out, next_true);
        }

        if is_req {
            cur_false = None;
            cur_true = Some(next_true);
        } else {
            let next_false = if let Some(cf) = cur_false {
                let nf = b.add_state();
                b.add_eps(cf, nf);
                Some(nf)
            } else {
                None
            };
            if let Some(ct) = cur_true {
                b.add_eps(ct, next_true);
            }
            cur_false = next_false;
            cur_true = Some(next_true);
        }
    }

    let pre_close = b.add_state();
    if let Some(cf) = cur_false {
        b.add_eps(cf, pre_close);
    }
    if let Some(ct) = cur_true {
        b.add_eps(ct, pre_close);
    }

    let ws_end = opt_ws(b);
    let close_brace = b.lit_byte(b'}');
    let end_tail = b.concat(&[ws_end, close_brace]);
    b.add_eps(pre_close, end_tail.start);

    Ok(NfaFrag {
        start: start_hdr.start,
        out: end_tail.out,
    })
}

/// Validates a parsed `serde_json::Value` against a Draft-2020-12 JSON Schema subset,
/// verifying types, `enum`, `const`, `required`, `properties`, `additionalProperties: false`,
/// `items`, `minItems`, `maxItems`, `minLength`, `maxLength`, and `anyOf`/`oneOf`.
pub fn validate_json_value(
    instance: &Value,
    root_schema: &Value,
    schema: &Value,
) -> Result<(), String> {
    let Some(obj) = schema.as_object() else {
        return Err("schema is not an object".into());
    };

    if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
        let ptr = ref_str
            .strip_prefix('#')
            .ok_or_else(|| format!("bad $ref {ref_str}"))?;
        let target = root_schema
            .pointer(ptr)
            .ok_or_else(|| format!("missing $ref target {ref_str}"))?;
        return validate_json_value(instance, root_schema, target);
    }

    if let Some(const_v) = obj.get("const") {
        if instance != const_v {
            return Err(format!(
                "const mismatch: got {instance}, expected {const_v}"
            ));
        }
    }

    if let Some(enum_arr) = obj.get("enum").and_then(Value::as_array) {
        if !enum_arr.iter().any(|cand| cand == instance) {
            return Err(format!("value {instance} not in enum {enum_arr:?}"));
        }
    }

    if let Some(any_of) = obj
        .get("anyOf")
        .or_else(|| obj.get("oneOf"))
        .and_then(Value::as_array)
    {
        if !any_of
            .iter()
            .any(|sub| validate_json_value(instance, root_schema, sub).is_ok())
        {
            return Err(format!("value {instance} matched none of anyOf/oneOf"));
        }
    }

    if let Some(type_val) = obj.get("type") {
        if let Some(t_arr) = type_val.as_array() {
            let ok = t_arr
                .iter()
                .filter_map(Value::as_str)
                .any(|t| check_type(instance, t));
            if !ok {
                return Err(format!("value {instance} failed union type {t_arr:?}"));
            }
        } else if let Some(ts) = type_val.as_str() {
            if !check_type(instance, ts) {
                return Err(format!("value {instance} expected type '{ts}'"));
            }
        }
    }

    match instance {
        Value::String(s) => {
            let char_len = s.chars().count();
            if let Some(min_l) = obj.get("minLength").and_then(Value::as_u64) {
                if char_len < min_l as usize {
                    return Err(format!("string len {char_len} < minLength {min_l}"));
                }
            }
            if let Some(max_l) = obj.get("maxLength").and_then(Value::as_u64) {
                if char_len > max_l as usize {
                    return Err(format!("string len {char_len} > maxLength {max_l}"));
                }
            }
        }
        Value::Array(arr) => {
            if let Some(min_i) = obj.get("minItems").and_then(Value::as_u64) {
                if arr.len() < min_i as usize {
                    return Err(format!("array len {} < minItems {min_i}", arr.len()));
                }
            }
            if let Some(max_i) = obj.get("maxItems").and_then(Value::as_u64) {
                if arr.len() > max_i as usize {
                    return Err(format!("array len {} > maxItems {max_i}", arr.len()));
                }
            }
            if let Some(item_schema) = obj.get("items") {
                for (idx, item) in arr.iter().enumerate() {
                    validate_json_value(item, root_schema, item_schema)
                        .map_err(|e| format!("items[{idx}]: {e}"))?;
                }
            }
        }
        Value::Object(map) => {
            if let Some(req_arr) = obj.get("required").and_then(Value::as_array) {
                for req in req_arr.iter().filter_map(Value::as_str) {
                    if !map.contains_key(req) {
                        return Err(format!("missing required field '{req}'"));
                    }
                }
            }
            let disallow_extra =
                obj.get("additionalProperties").and_then(Value::as_bool) == Some(false);
            if let Some(props) = obj.get("properties").and_then(Value::as_object) {
                for (k, v) in map {
                    if let Some(sub_s) = props.get(k) {
                        validate_json_value(v, root_schema, sub_s)
                            .map_err(|e| format!("field '{k}': {e}"))?;
                    } else if disallow_extra {
                        return Err(format!("disallowed additionalProperty '{k}'"));
                    }
                }
            }
        }
        _ => {}
    }

    Ok(())
}

fn check_type(v: &Value, expected: &str) -> bool {
    match expected {
        "null" => v.is_null(),
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "integer" => v.is_i64() || v.is_u64(),
        "number" => v.is_number(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        _ => false,
    }
}
