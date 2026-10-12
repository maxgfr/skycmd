//! GGUF `skycmd-s2` export (SPEC-S2 §5) and the matching reader used by the
//! round-trip tests, the golden generator and `s2-eval --gguf`.
//!
//! Ternary layout (`weight_format = ternary`): a ternary matrix `X.weight`
//! `[rows, cols]` is stored as `X.tq` (I8, ggml dims `[ceil(cols / 4), rows]`)
//! holding 4 codes per byte, code `j` of a group in bits `2j..2j+1` (the first
//! weight in the least significant bits), `00` = 0, `01` = +1, `10` = -1, each
//! row zero-padded to a multiple of 4 weights; and `X.scale` (F16 `[rows]`).
//! `token_embd.weight` becomes `token_embd.q8` (I8 `[V, d]`, `q = round(x / s)`
//! with `s = max|row| / 127`) and `token_embd.scale` (F16 `[V]`).

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use mlx_rs::Array;
use serde_json::Value;
use sky_gguf::{f16_to_f32, f32_to_f16, GgmlType, Gguf, GgufWriter, OwnedValue};

use crate::{
    model::host_f32,
    num::{NUM_FEATURES, NUM_ID},
    params::ParamMap,
    s2::{
        config::{S2Config, ARCH},
        model::{is_ternary, param_shapes, ternary_codes, S2Parts},
    },
    sequence::QTYPES,
};

/// Storage of the weight matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    F16,
    Ternary,
}

impl WeightFormat {
    pub fn name(self) -> &'static str {
        match self {
            WeightFormat::F32 => "f32",
            WeightFormat::F16 => "f16",
            WeightFormat::Ternary => "ternary",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "f32" => WeightFormat::F32,
            "f16" => WeightFormat::F16,
            "ternary" => WeightFormat::Ternary,
            _ => bail!("unknown weight format {s:?} (f32, f16 or ternary)"),
        })
    }
}

// ----------------------------------------------------------------------------- tokenizer keys

mod token_type {
    pub const NORMAL: i32 = 1;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
}

/// GPT-2 pre-tokenizer split pattern (as accepted by sky-infer).
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn normalizer_is_nfc(n: &Value) -> Result<bool> {
    match n["type"].as_str() {
        Some("NFC") => Ok(true),
        Some("Sequence") => {
            let mut nfc = false;
            for item in n["normalizers"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                nfc |= normalizer_is_nfc(item)?;
            }
            Ok(nfc)
        }
        other => bail!("unsupported normalizer {other:?}"),
    }
}

/// `(add_prefix_space, use_regex)` of the pre-tokenizer.
fn pre_tokenizer(p: &Value) -> Result<(bool, bool)> {
    match p["type"].as_str() {
        Some("ByteLevel") => Ok((
            p["add_prefix_space"].as_bool().unwrap_or(false),
            p["use_regex"].as_bool().unwrap_or(true),
        )),
        Some("Sequence") => match p["pretokenizers"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            [one] => pre_tokenizer(one),
            [split, bl] if split["type"] == "Split" => {
                let pat = split["pattern"]
                    .get("Regex")
                    .or_else(|| split["pattern"].get("String"));
                let (prefix, regex) = pre_tokenizer(bl)?;
                if pat.and_then(Value::as_str) == Some(GPT2_PATTERN)
                    && split["behavior"] == "Isolated"
                    && !regex
                {
                    Ok((prefix, true))
                } else {
                    bail!("Split pre-tokenizer with a non GPT-2 pattern")
                }
            }
            _ => bail!("unsupported pre-tokenizer sequence"),
        },
        other => bail!("unsupported pre-tokenizer {other:?}"),
    }
}

/// `jinja_str_or_json(x)` of the llama.cpp converter.
fn str_or_json(var: &str) -> String {
    format!("{{% if {var} is string %}}{{{{ {var} }}}}{{% else %}}{{{{ {var} | tojson }}}}{{% endif %}}")
}

/// The `systemone` chat template of the s1 GGUF files (Laya layout).
fn systemone_template(cls: &str, sep: &str, mask: &str) -> String {
    let description = str_or_json("o.description");
    let option = format!(
        "{{% if type == 'choice' %}}{{{{ o.key }}}}{{% if o.description %}}: {description}{{% endif %}}\
         {{% elif type == 'score' %}}level {{{{ o.key }}}}: {description}\
         {{% else %}}{{{{ o.key }}}}: {{% if o.description %}}{description}\
         {{% elif o.key == 'true' %}}yes, the statement holds\
         {{% else %}}no, the statement does not hold{{% endif %}}{{% endif %}}"
    );
    format!(
        "{cls}{{{{ type }}}} question: {}{sep}{{% for o in options %}}{mask} {option}{{% endfor %}}{sep}{}{sep}",
        str_or_json("instructions"),
        str_or_json("state")
    )
}

/// Writes the `tokenizer.ggml.*` and `skycmd.tokenizer.*` keys exactly as
/// `sky_infer::convert::write_tokenizer` does for s1.
pub fn write_tokenizer(w: &mut GgufWriter, tokenizer_json: &str, vocab_size: usize) -> Result<()> {
    let j: Value = serde_json::from_str(tokenizer_json).context("parsing tokenizer.json")?;
    let model = &j["model"];
    if model["type"] != "BPE" {
        bail!("tokenizer model must be BPE");
    }
    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| anyhow!("model.vocab is not an object"))?;
    let added: Vec<&Value> = j["added_tokens"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let max_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .chain(added.iter().filter_map(|a| a["id"].as_u64()))
        .max()
        .unwrap_or(0) as usize;
    let n = vocab_size.max(max_id + 1);
    let mut tokens = vec![String::new(); n];
    let mut types = vec![token_type::UNUSED; n];
    for (s, id) in vocab {
        let id = id.as_u64().ok_or_else(|| anyhow!("bad id for {s}"))? as usize;
        tokens[id] = s.clone();
        types[id] = token_type::NORMAL;
    }
    let (mut lstrip, mut rstrip) = (Vec::new(), Vec::new());
    for a in &added {
        let id = a["id"]
            .as_u64()
            .ok_or_else(|| anyhow!("added token without id"))? as usize;
        tokens[id] = a["content"].as_str().unwrap_or_default().to_string();
        types[id] = if a["special"].as_bool().unwrap_or(false) {
            token_type::CONTROL
        } else {
            token_type::USER_DEFINED
        };
        if a["lstrip"].as_bool().unwrap_or(false) {
            lstrip.push(OwnedValue::I32(id as i32));
        }
        if a["rstrip"].as_bool().unwrap_or(false) {
            rstrip.push(OwnedValue::I32(id as i32));
        }
    }
    for (i, t) in tokens.iter_mut().enumerate() {
        if t.is_empty() {
            *t = format!("[PAD{i}]");
        }
    }
    let mut merges = Vec::new();
    for m in model["merges"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        merges.push(match m {
            Value::String(s) => s.clone(),
            Value::Array(p) if p.len() == 2 => format!(
                "{} {}",
                p[0].as_str().ok_or_else(|| anyhow!("bad merge"))?,
                p[1].as_str().ok_or_else(|| anyhow!("bad merge"))?
            ),
            _ => bail!("bad merge entry"),
        });
    }
    let nfc = match &j["normalizer"] {
        Value::Null => false,
        n => normalizer_is_nfc(n)?,
    };
    let (add_prefix_space, use_regex) = match &j["pre_tokenizer"] {
        Value::Null => bail!("tokenizer without a pre-tokenizer"),
        p => pre_tokenizer(p)?,
    };
    w.add_kv("tokenizer.ggml.model", "gpt2");
    w.add_kv("tokenizer.ggml.pre", "gpt-2");
    w.add_kv("tokenizer.ggml.tokens", OwnedValue::strings(&tokens));
    w.add_kv(
        "tokenizer.ggml.token_type",
        OwnedValue::Array(types.into_iter().map(OwnedValue::I32).collect()),
    );
    w.add_kv("tokenizer.ggml.merges", OwnedValue::strings(&merges));
    w.add_kv("tokenizer.ggml.token_type_count", OwnedValue::U32(3));
    w.add_kv("tokenizer.ggml.add_bos_token", true);
    w.add_kv("tokenizer.ggml.add_eos_token", true);
    w.add_kv("tokenizer.ggml.add_sep_token", true);
    let id_of = |name: &str| -> Option<u32> {
        added
            .iter()
            .find(|a| a["content"] == name)
            .and_then(|a| a["id"].as_u64())
            .or_else(|| vocab.get(name).and_then(Value::as_u64))
            .map(|v| v as u32)
    };
    let (cls, sep, mask, pad, unk) = (
        id_of("[CLS]"),
        id_of("[SEP]"),
        id_of("[MASK]"),
        id_of("[PAD]"),
        id_of("[UNK]"),
    );
    for (key, id) in [
        ("tokenizer.ggml.bos_token_id", cls),
        ("tokenizer.ggml.eos_token_id", sep),
        ("tokenizer.ggml.unknown_token_id", unk),
        ("tokenizer.ggml.seperator_token_id", sep),
        ("tokenizer.ggml.padding_token_id", pad),
        ("tokenizer.ggml.cls_token_id", cls),
        ("tokenizer.ggml.mask_token_id", mask),
    ] {
        if let Some(id) = id {
            w.add_kv(key, OwnedValue::U32(id));
        }
    }
    let name = |id: Option<u32>, d: &str| {
        id.map(|i| tokens[i as usize].clone())
            .unwrap_or_else(|| d.to_string())
    };
    w.add_kv(
        "tokenizer.chat_template.systemone",
        systemone_template(
            &name(cls, "[CLS]"),
            &name(sep, "[SEP]"),
            &name(mask, "[MASK]"),
        ),
    );
    w.add_kv(
        "tokenizer.chat_templates",
        OwnedValue::strings(&["systemone"]),
    );
    if nfc {
        w.add_kv("skycmd.tokenizer.normalizer", "nfc");
    }
    if !lstrip.is_empty() {
        w.add_kv("skycmd.tokenizer.lstrip", OwnedValue::Array(lstrip));
    }
    if !rstrip.is_empty() {
        w.add_kv("skycmd.tokenizer.rstrip", OwnedValue::Array(rstrip));
    }
    if model["ignore_merges"].as_bool().unwrap_or(false) {
        w.add_kv("skycmd.tokenizer.ignore_merges", true);
    }
    if !use_regex {
        w.add_kv("skycmd.tokenizer.use_regex", false);
    }
    w.add_kv("tokenizer.ggml.add_space_prefix", add_prefix_space);
    Ok(())
}

// ----------------------------------------------------------------------------- ternary packing

/// Packs ternary codes `[rows, cols]` into `[rows, ceil(cols / 4)]` bytes.
pub fn pack_ternary(codes: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let bpr = cols.div_ceil(4);
    let mut out = vec![0u8; rows * bpr];
    for r in 0..rows {
        for c in 0..cols {
            let bits = match codes[r * cols + c] {
                1 => 0b01u8,
                -1 => 0b10,
                _ => 0,
            };
            out[r * bpr + c / 4] |= bits << (2 * (c % 4));
        }
    }
    out
}

/// Inverse of [`pack_ternary`].
pub fn unpack_ternary(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>> {
    let bpr = cols.div_ceil(4);
    if bytes.len() != rows * bpr {
        bail!(
            "ternary tensor has {} bytes, {} expected",
            bytes.len(),
            rows * bpr
        );
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(match (bytes[r * bpr + c / 4] >> (2 * (c % 4))) & 3 {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => bail!("invalid ternary code 11"),
            });
        }
    }
    Ok(out)
}

fn f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

// ----------------------------------------------------------------------------- writer

/// Builds the GGUF bytes of a decision model.
pub fn to_gguf(
    p: &ParamMap,
    cfg: &S2Config,
    tokenizer_json: &str,
    format: WeightFormat,
    name: &str,
) -> Result<Vec<u8>> {
    let mut w = GgufWriter::new();
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: usize| OwnedValue::U32(v as u32);
    w.add_kv("general.architecture", ARCH);
    w.add_kv("general.type", "model");
    w.add_kv("general.name", name);
    w.add_kv(k("vocab_size"), u(cfg.vocab_size));
    w.add_kv(k("embedding_length"), u(cfg.hidden_size));
    w.add_kv(k("feed_forward_length"), u(cfg.intermediate_size));
    w.add_kv(k("head_count"), u(cfg.n_heads()));
    w.add_kv(k("state_block_count"), u(cfg.state_layers));
    w.add_kv(k("static_block_count"), u(cfg.static_layers));
    w.add_kv(k("fusion_block_count"), u(cfg.fusion_layers));
    // SPEC-S3: written only when on, so s2 files stay byte-identical.
    if cfg.fuse_numbers {
        w.add_kv(k("fuse_numbers"), true);
    }
    if cfg.state_loops != 1 {
        w.add_kv(k("state_loops"), u(cfg.state_loops));
    }
    if cfg.temporal_carry {
        w.add_kv(k("temporal_carry"), true);
    }
    // SPEC-S3 §9: the fixed up-projection is the seed alone (no tensor).
    if let Some(es) = &cfg.ffn_es {
        w.add_kv(k("ffn_es_seed"), OwnedValue::U64(es.seed));
        w.add_kv(k("ffn_es_units"), u(es.units(cfg.hidden_size)));
        w.add_kv(k("ffn_es_k"), u(es.k(cfg.hidden_size)));
        w.add_kv(k("ffn_es_fan_in"), u(es.fan_in));
    }
    w.add_kv(
        k("exits"),
        OwnedValue::Array(cfg.exits.iter().map(|&e| u(e)).collect()),
    );
    w.add_kv(k("max_state_tokens"), u(cfg.max_state_tokens));
    w.add_kv(k("max_head_tokens"), u(cfg.max_head_tokens));
    w.add_kv(k("num_features"), u(NUM_FEATURES));
    w.add_kv(k("num_token_id"), u(NUM_ID as usize));
    w.add_kv(k("rope_freq_base"), OwnedValue::F32(cfg.rope_theta));
    w.add_kv(k("layer_norm_epsilon"), OwnedValue::F32(cfg.norm_eps));
    for (q, t) in QTYPES.iter().enumerate() {
        w.add_kv(
            k(&format!("exit_threshold.{t}")),
            OwnedValue::F32(cfg.exit_threshold[q]),
        );
    }
    for (key, t) in &cfg.temperature {
        w.add_kv(k(&format!("temperature.{key}")), OwnedValue::F32(*t));
    }
    w.add_kv(k("weight_format"), format.name());
    write_tokenizer(&mut w, tokenizer_json, cfg.vocab_size)?;

    for (name, shape) in param_shapes(cfg, S2Parts::DECISION) {
        let a = p
            .get(name.as_str())
            .ok_or_else(|| anyhow!("missing parameter {name}"))?;
        if a.shape() != shape.as_slice() {
            bail!("{name}: shape {:?}, expected {shape:?}", a.shape());
        }
        let values = host_f32(a)?;
        let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
        let base = name.strip_suffix(".weight").unwrap_or(&name);
        match format {
            WeightFormat::Ternary if is_ternary(&name) => {
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                let (codes, scales) = ternary_codes(&values, rows, cols);
                w.add_tensor(
                    format!("{base}.tq"),
                    &[cols.div_ceil(4) as u64, rows as u64],
                    GgmlType::I8,
                    pack_ternary(&codes, rows, cols),
                )?;
                w.add_tensor(
                    format!("{base}.scale"),
                    &[rows as u64],
                    GgmlType::F16,
                    f16_bytes(&scales),
                )?;
            }
            WeightFormat::Ternary if name == "token_embd.weight" => {
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                let mut q = Vec::with_capacity(values.len());
                let mut scales = Vec::with_capacity(rows);
                for r in 0..rows {
                    let row = &values[r * cols..(r + 1) * cols];
                    let s = row.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
                    let s = if s > 0.0 { s } else { 1.0 };
                    scales.push(s);
                    q.extend(
                        row.iter()
                            .map(|x| (x / s).round().clamp(-127.0, 127.0) as i8 as u8),
                    );
                }
                w.add_tensor("token_embd.q8", &dims, GgmlType::I8, q)?;
                w.add_tensor(
                    "token_embd.scale",
                    &[rows as u64],
                    GgmlType::F16,
                    f16_bytes(&scales),
                )?;
            }
            WeightFormat::F16 if shape.len() == 2 => {
                w.add_tensor_f32(name.as_str(), &dims, GgmlType::F16, &values)?;
            }
            _ => {
                w.add_tensor_f32(name.as_str(), &dims, GgmlType::F32, &values)?;
            }
        }
    }
    Ok(w.to_bytes()?)
}

// ----------------------------------------------------------------------------- reader

fn u32_key(g: &Gguf<'_>, key: &str) -> Result<usize> {
    g.get_u64(&format!("{ARCH}.{key}"))
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("missing key {ARCH}.{key}"))
}

/// The model configuration stored in a `skycmd-s2` GGUF.
pub fn config_from_gguf(g: &Gguf<'_>) -> Result<S2Config> {
    if g.architecture() != Some(ARCH) {
        bail!("not a {ARCH} file: architecture {:?}", g.architecture());
    }
    let f = |key: &str| g.get_f64(&format!("{ARCH}.{key}")).map(|v| v as f32);
    let exits = g
        .get_array(&format!("{ARCH}.exits"))
        .and_then(|a| a.to_i64_vec())
        .ok_or_else(|| anyhow!("missing exits"))?
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let mut temperature = BTreeMap::new();
    let prefix = format!("{ARCH}.temperature.");
    for (key, v) in &g.kvs {
        if let (Some(rest), Some(t)) = (key.strip_prefix(prefix.as_str()), v.as_f64()) {
            temperature.insert(rest.to_string(), t as f32);
        }
    }
    let mut exit_threshold = [0.9f32; 3];
    for (q, t) in QTYPES.iter().enumerate() {
        if let Some(v) = f(&format!("exit_threshold.{t}")) {
            exit_threshold[q] = v;
        }
    }
    let hidden_size = u32_key(g, "embedding_length")?;
    let head_count = g
        .get_u64(&format!("{ARCH}.head_count"))
        .map(|v| v as usize)
        .filter(|&h| h * crate::config::HEAD_DIM as usize != hidden_size);
    let fuse_numbers = match g.get(&format!("{ARCH}.fuse_numbers")) {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| anyhow!("{ARCH}.fuse_numbers is not a bool"))?,
    };
    let temporal_carry = match g.get(&format!("{ARCH}.temporal_carry")) {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| anyhow!("{ARCH}.temporal_carry is not a bool"))?,
    };
    let state_loops = g
        .get_u64(&format!("{ARCH}.state_loops"))
        .map_or(1, |v| v as usize);
    let ffn_es = match g.get_u64(&format!("{ARCH}.ffn_es_seed")) {
        None => None,
        Some(seed) => {
            let units = u32_key(g, "ffn_es_units")?;
            if units == 0 || !units.is_multiple_of(hidden_size) {
                bail!("{ARCH}.ffn_es_units {units} is not a multiple of {hidden_size}");
            }
            Some(crate::s2::config::EsFfn {
                seed,
                expand: units / hidden_size,
                k_frac: u32_key(g, "ffn_es_k")? as f32 / units as f32,
                fan_in: u32_key(g, "ffn_es_fan_in")?,
            })
        }
    };
    let cfg = S2Config {
        ffn_es,
        name: g.get_str("general.name").unwrap_or("s2").to_string(),
        head_count,
        fuse_numbers,
        state_loops,
        temporal_carry,
        vocab_size: u32_key(g, "vocab_size")?,
        hidden_size,
        intermediate_size: u32_key(g, "feed_forward_length")?,
        state_layers: u32_key(g, "state_block_count")?,
        static_layers: u32_key(g, "static_block_count")?,
        fusion_layers: u32_key(g, "fusion_block_count")?,
        exits,
        max_state_tokens: u32_key(g, "max_state_tokens")?,
        max_head_tokens: u32_key(g, "max_head_tokens")?,
        rope_theta: f("rope_freq_base").unwrap_or(10_000.0),
        norm_eps: f("layer_norm_epsilon").unwrap_or(1e-5),
        exit_threshold,
        temperature,
        bf16: false,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// Reads every parameter back as f32 (ternary matrices as `code · f16 scale`,
/// int8 embeddings as `q · f16 scale`): exactly the weights an engine computes with.
pub fn params_from_gguf(g: &Gguf<'_>, cfg: &S2Config) -> Result<ParamMap> {
    let mut map = ParamMap::new();
    let scales = |name: &str, rows: usize| -> Result<Vec<f32>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        if info.dtype != GgmlType::F16 || info.n_elements() != rows as u64 {
            bail!("{name}: expected F16 [{rows}]");
        }
        Ok(g.tensor_bytes(info)
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    };
    for (name, shape) in param_shapes(cfg, S2Parts::DECISION) {
        let base = name.strip_suffix(".weight").unwrap_or(&name);
        let values: Vec<f32> = if let Some(info) = g.tensor(&format!("{base}.tq")) {
            let (rows, cols) = (shape[0] as usize, shape[1] as usize);
            let codes = unpack_ternary(g.tensor_bytes(info), rows, cols)?;
            let s = scales(&format!("{base}.scale"), rows)?;
            codes
                .iter()
                .enumerate()
                .map(|(i, &c)| c as f32 * s[i / cols])
                .collect()
        } else if let (Some(info), true) = (g.tensor("token_embd.q8"), name == "token_embd.weight")
        {
            let (rows, cols) = (shape[0] as usize, shape[1] as usize);
            let s = scales("token_embd.scale", rows)?;
            g.tensor_bytes(info)
                .iter()
                .enumerate()
                .map(|(i, &b)| b as i8 as f32 * s[i / cols])
                .collect()
        } else {
            let info = g
                .tensor(&name)
                .ok_or_else(|| anyhow!("missing tensor {name}"))?;
            let got: Vec<i32> = info.shape().iter().map(|&d| d as i32).collect();
            if got != shape {
                bail!("{name}: shape {got:?}, expected {shape:?}");
            }
            g.tensor_f32(&name)?
        };
        map.insert(name.as_str().into(), Array::from_slice(&values, &shape));
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        s2::{data::tests::s2_tokenizer_json, model::init},
        test_util,
    };

    #[test]
    fn ternary_packing_round_trip() {
        let codes: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 0, 1, -1, 1, 1, 0, 0, 0];
        let packed = pack_ternary(&codes, 2, 7);
        assert_eq!(packed.len(), 4);
        // Row 0: [1, -1, 0, 1] -> 01 | 10<<2 | 00<<4 | 01<<6.
        assert_eq!(packed[0], 0b01_00_10_01);
        assert_eq!(unpack_ternary(&packed, 2, 7).unwrap(), codes);
        assert!(unpack_ternary(&[0xff], 1, 4).is_err());
    }

    #[test]
    fn es_ffn_keys_round_trip() {
        let _g = test_util::lock();
        let tok = s2_tokenizer_json();
        let es = crate::s2::config::EsFfn {
            seed: 0xfeed_beef_0123_4567,
            expand: 8,
            k_frac: 0.05,
            fan_in: 8,
        };
        let cfg = S2Config {
            ffn_es: Some(es),
            ..S2Config::tiny(262)
        };
        let p = init(&cfg, S2Parts::DECISION, 2).unwrap();
        let b = to_gguf(&p, &cfg, &tok, WeightFormat::Ternary, "es").unwrap();
        let g = Gguf::parse(&b).unwrap();
        assert_eq!(g.get_u64("skycmd-s2.ffn_es_seed"), Some(es.seed));
        assert_eq!(g.get_u64("skycmd-s2.ffn_es_units"), Some(1024));
        assert_eq!(g.get_u64("skycmd-s2.ffn_es_k"), Some(51));
        assert!(g.tensor("blk.0.ffn_up.tq").is_none() && g.tensor("blk.0.ffn_down.tq").is_some());
        let back = config_from_gguf(&g).unwrap();
        let bes = back.ffn_es.unwrap();
        assert_eq!(
            (bes.seed, bes.units(128), bes.k(128), bes.fan_in),
            (es.seed, 1024, 51, 8)
        );
        assert_eq!(params_from_gguf(&g, &back).unwrap().len(), p.len());
        // Off by default: no key at all.
        let p2 = init(&S2Config::tiny(262), S2Parts::DECISION, 1).unwrap();
        let g2b = to_gguf(&p2, &S2Config::tiny(262), &tok, WeightFormat::F32, "s2").unwrap();
        let g2 = Gguf::parse(&g2b).unwrap();
        assert!(g2.get("skycmd-s2.ffn_es_seed").is_none());
        assert!(config_from_gguf(&g2).unwrap().ffn_es.is_none());
    }

    #[test]
    fn s3_keys_round_trip() {
        let _g = test_util::lock();
        let tok = s2_tokenizer_json();
        // s2: none of the new keys or tensors.
        let cfg2 = S2Config::tiny(262);
        let p2 = init(&cfg2, S2Parts::DECISION, 1).unwrap();
        let g2b = to_gguf(&p2, &cfg2, &tok, WeightFormat::Ternary, "s2").unwrap();
        let g2 = Gguf::parse(&g2b).unwrap();
        assert!(
            g2.get("skycmd-s2.fuse_numbers").is_none() && g2.get("skycmd-s2.state_loops").is_none()
        );
        assert_eq!(g2.get_u64("skycmd-s2.head_count"), Some(2));
        assert!(g2.tensor("loop_embd.weight").is_none());
        let back = config_from_gguf(&g2).unwrap();
        assert_eq!(
            (back.fuse_numbers, back.state_loops, back.head_count),
            (false, 1, None)
        );
        assert!(!back.temporal_carry && g2.get("skycmd-s2.temporal_carry").is_none());
        // Temporal carry: key and the two vectors.
        let tcfg = S2Config {
            temporal_carry: true,
            ..S2Config::tiny(262)
        };
        let tp = init(&tcfg, S2Parts::DECISION, 4).unwrap();
        let tb = to_gguf(&tp, &tcfg, &tok, WeightFormat::Ternary, "t").unwrap();
        let tg = Gguf::parse(&tb).unwrap();
        assert_eq!(tg.get_bool("skycmd-s2.temporal_carry"), Some(true));
        assert_eq!(tg.tensor("carry_gate.weight").unwrap().dtype, GgmlType::F32);
        assert_eq!(tg.tensor("carry_norm.weight").unwrap().shape(), vec![128]);
        let tback = config_from_gguf(&tg).unwrap();
        assert!(tback.temporal_carry);
        assert_eq!(params_from_gguf(&tg, &tback).unwrap().len(), tp.len());
        // s3: fused, 1 block x 4 loops, one 96-wide head.
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s3-pico.json");
        let mut cfg = S2Config::load(path).unwrap();
        cfg.vocab_size = 262;
        cfg.temperature.insert("e3.noul".into(), 0.6);
        cfg.exit_threshold = [0.85, 1.0, 0.7];
        let p = init(&cfg, S2Parts::DECISION, 2).unwrap();
        for format in [WeightFormat::F32, WeightFormat::F16, WeightFormat::Ternary] {
            let bytes = to_gguf(&p, &cfg, &tok, format, "s3-pico").unwrap();
            let g = Gguf::parse(&bytes).unwrap();
            assert_eq!(g.get_bool("skycmd-s2.fuse_numbers"), Some(true));
            assert_eq!(g.get_u64("skycmd-s2.state_loops"), Some(4));
            assert_eq!(g.get_u64("skycmd-s2.state_block_count"), Some(1));
            assert_eq!(g.get_u64("skycmd-s2.head_count"), Some(1));
            assert_eq!(g.get_u64("skycmd-s2.embedding_length"), Some(96));
            let ex = g
                .get_array("skycmd-s2.exits")
                .unwrap()
                .to_i64_vec()
                .unwrap();
            assert_eq!(ex, vec![1, 2, 3, 4]);
            let le = g.tensor("loop_embd.weight").unwrap();
            assert_eq!(le.shape(), vec![4, 96]);
            assert_eq!(
                g.tensor("blk.0.attn_norm_loop.weight").unwrap().dtype,
                GgmlType::F32
            );
            assert!(g.tensor("blk.0.attn_norm.weight").is_none());
            if format == WeightFormat::Ternary {
                assert_eq!(le.dtype, GgmlType::F32);
                assert!(g.tensor("blk.0.attn_qkv.tq").is_some());
            }
            let back = config_from_gguf(&g).unwrap();
            assert_eq!(
                (
                    back.fuse_numbers,
                    back.state_loops,
                    back.head_count,
                    back.head_dim()
                ),
                (true, 4, Some(1), 96)
            );
            assert_eq!(back.exits, cfg.exits);
            assert_eq!(back.exit_threshold, cfg.exit_threshold);
            assert_eq!(back.temperature, cfg.temperature);
            let q = params_from_gguf(&g, &back).unwrap();
            assert_eq!(q.len(), p.len());
            for name in ["loop_embd.weight", "blk.0.attn_norm_loop.weight"] {
                let (a, b) = (host_f32(&p[name]).unwrap(), host_f32(&q[name]).unwrap());
                let tol = if format == WeightFormat::F16 {
                    1e-3
                } else {
                    0.0
                };
                assert!(
                    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn export_round_trip_names_and_shapes() {
        let _g = test_util::lock();
        let mut cfg = S2Config::tiny(262);
        cfg.temperature.insert("e1.choice.3_5".into(), 1.7);
        cfg.exit_threshold = [0.8, 1.0, 0.95];
        let p = init(&cfg, S2Parts::DECISION, 3).unwrap();
        let tok = s2_tokenizer_json();
        for format in [WeightFormat::F32, WeightFormat::F16, WeightFormat::Ternary] {
            let bytes = to_gguf(&p, &cfg, &tok, format, "s2-tiny").unwrap();
            let g = Gguf::parse(&bytes).unwrap();
            assert_eq!(g.architecture(), Some(ARCH));
            assert_eq!(g.get_str("skycmd-s2.weight_format"), Some(format.name()));
            assert_eq!(g.get_u64("skycmd-s2.num_token_id"), Some(5));
            assert_eq!(g.get_u64("skycmd-s2.num_features"), Some(17));
            assert_eq!(g.get_u64("tokenizer.ggml.mask_token_id"), Some(4));
            assert_eq!(g.get_array("tokenizer.ggml.tokens").unwrap().len(), 262);
            let back = config_from_gguf(&g).unwrap();
            assert_eq!(back.exits, cfg.exits);
            assert_eq!(back.temperature, cfg.temperature);
            assert_eq!(back.exit_threshold, cfg.exit_threshold);
            assert_eq!(
                (back.hidden_size, back.intermediate_size, back.vocab_size),
                (128, 90, 262)
            );
            let q = params_from_gguf(&g, &back).unwrap();
            assert_eq!(q.len(), p.len());
            for (name, shape) in param_shapes(&cfg, S2Parts::DECISION) {
                assert_eq!(q[name.as_str()].shape(), shape.as_slice(), "{name}");
                let (a, b) = (
                    host_f32(&p[name.as_str()]).unwrap(),
                    host_f32(&q[name.as_str()]).unwrap(),
                );
                let base = name.strip_suffix(".weight").unwrap_or(&name).to_string();
                match format {
                    WeightFormat::Ternary if is_ternary(&name) => {
                        assert!(
                            g.tensor(&format!("{base}.tq")).is_some() && g.tensor(&name).is_none()
                        );
                        assert_eq!(g.tensor(&format!("{base}.tq")).unwrap().dtype, GgmlType::I8);
                        assert_eq!(
                            g.tensor(&format!("{base}.scale")).unwrap().dtype,
                            GgmlType::F16
                        );
                        // Every value is one of {-γ, 0, γ} of its row and matches fake-quant.
                        let fq = host_f32(&crate::s2::model::ternarize(&p[name.as_str()]).unwrap())
                            .unwrap();
                        for (x, y) in b.iter().zip(&fq) {
                            assert!(
                                (x - y).abs() <= 1e-3 * y.abs().max(1e-3),
                                "{name}: {x} vs {y}"
                            );
                        }
                    }
                    WeightFormat::Ternary if name == "token_embd.weight" => {
                        assert!(
                            g.tensor("token_embd.q8").is_some()
                                && g.tensor("token_embd.weight").is_none()
                        );
                        for (x, y) in a.iter().zip(&b) {
                            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
                        }
                    }
                    WeightFormat::F16 if shape.len() == 2 => {
                        assert_eq!(g.tensor(&name).unwrap().dtype, GgmlType::F16);
                    }
                    _ => {
                        assert_eq!(g.tensor(&name).unwrap().dtype, GgmlType::F32, "{name}");
                        assert_eq!(a, b, "{name}");
                    }
                }
            }
        }
    }
}
