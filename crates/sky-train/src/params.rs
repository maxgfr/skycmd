//! Flat parameter maps keyed by the SPEC safetensors names, initialization,
//! parameter counting, optimizer groups and safetensors I/O.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    rc::Rc,
};

use anyhow::{anyhow, Result};
use mlx_rs::{random, Array, Dtype};

use crate::config::ModelConfig;

/// Model parameters keyed by their serialized name (`encoder.layers.0.attn.Wqkv.weight`, ...).
pub type ParamMap = HashMap<Rc<str>, Array>;

/// Which parts of the model a parameter map holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parts {
    /// Laya decision head (`head.*`, `type_emb`, `scorer.*`).
    pub decision: bool,
    /// MLM prediction head used only for pretraining (`mlm.*`, never exported).
    pub mlm: bool,
}

impl Parts {
    pub const PRETRAIN: Parts = Parts {
        decision: false,
        mlm: true,
    };
    pub const DECISION: Parts = Parts {
        decision: true,
        mlm: false,
    };
}

/// How a parameter is initialized.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Init {
    /// Normal truncated at ±2σ (HF ModernBERT `init_weight`).
    TruncNormal(f32),
    /// `U(-a, a)` (PyTorch linear / xavier defaults of the Laya head).
    Uniform(f32),
    /// Plain normal (PyTorch `nn.Embedding`).
    Normal(f32),
    Ones,
    Zeros,
}

/// Optimizer group of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    /// Encoder hidden matrices (Wqkv, Wo, Wi, mlp Wo): Muon.
    Muon,
    /// Other matrices (head, scorer, MLM dense): AdamW with weight decay.
    AdamDecay,
    /// Embeddings, norms and biases: AdamW without weight decay.
    AdamNoDecay,
}

/// Optimizer group of parameter `name` with `ndim` dimensions.
///
/// s1 names (`encoder.layers.*`) and s2 names (`blk.*`, `static.blk.*`) are
/// both recognized: the encoder hidden matrices go to Muon; the s2 embeddings
/// (`token_embd`, `num_proj`, `type_embd`, the per-loop `loop_embd`) and the s6-cast input
/// projection and horizon queries (`patch_proj`, `cast_query`) are never decayed.
pub fn group_of(name: &str, ndim: usize) -> Group {
    let hidden = name.starts_with("encoder.layers.")
        && (name.ends_with(".attn.Wqkv.weight")
            || name.ends_with(".attn.Wo.weight")
            || name.ends_with(".mlp.Wi.weight")
            || name.ends_with(".mlp.Wo.weight"));
    let s2_embedding = matches!(
        name,
        "token_embd.weight"
            | "type_embd.weight"
            | "num_proj.weight"
            | "loop_embd.weight"
            | "patch_proj.weight"
            | "cast_query.weight"
    );
    if hidden || crate::s2::model::is_hidden_matrix(name) {
        Group::Muon
    } else if ndim == 2
        && !name.contains("embeddings")
        && name != "type_emb.weight"
        && !s2_embedding
    {
        Group::AdamDecay
    } else {
        Group::AdamNoDecay
    }
}

/// Every parameter of the model as (name, shape, init), in a stable order.
fn specs(cfg: &ModelConfig, parts: Parts) -> Vec<(String, Vec<i32>, Init)> {
    let d = cfg.hidden_size as i32;
    let i = cfg.intermediate_size as i32;
    let v = cfg.vocab_size as i32;
    let std = 0.02f32;
    let std_out = std / (2.0 * cfg.num_hidden_layers as f32).sqrt();
    let mut out = vec![
        (
            "encoder.embeddings.tok_embeddings.weight".to_string(),
            vec![v, d],
            Init::TruncNormal(std),
        ),
        (
            "encoder.embeddings.norm.weight".to_string(),
            vec![d],
            Init::Ones,
        ),
    ];
    for l in 0..cfg.num_hidden_layers {
        let p = format!("encoder.layers.{l}");
        if l > 0 {
            out.push((format!("{p}.attn_norm.weight"), vec![d], Init::Ones));
        }
        out.push((
            format!("{p}.attn.Wqkv.weight"),
            vec![3 * d, d],
            Init::TruncNormal(std),
        ));
        out.push((
            format!("{p}.attn.Wo.weight"),
            vec![d, d],
            Init::TruncNormal(std_out),
        ));
        out.push((format!("{p}.mlp_norm.weight"), vec![d], Init::Ones));
        out.push((
            format!("{p}.mlp.Wi.weight"),
            vec![2 * i, d],
            Init::TruncNormal(std),
        ));
        out.push((
            format!("{p}.mlp.Wo.weight"),
            vec![d, i],
            Init::TruncNormal(std_out),
        ));
    }
    out.push(("encoder.final_norm.weight".to_string(), vec![d], Init::Ones));
    if parts.decision {
        let ff = 4 * d;
        let lin = |fan_in: i32| Init::Uniform(1.0 / (fan_in as f32).sqrt());
        for h in 0..cfg.rl_agent.head_layers {
            let p = format!("head.layers.{h}");
            let xavier = Init::Uniform((6.0 / (d + 3 * d) as f32).sqrt());
            out.push((format!("{p}.norm1.weight"), vec![d], Init::Ones));
            out.push((format!("{p}.norm1.bias"), vec![d], Init::Zeros));
            out.push((
                format!("{p}.self_attn.in_proj_weight"),
                vec![3 * d, d],
                xavier,
            ));
            out.push((
                format!("{p}.self_attn.in_proj_bias"),
                vec![3 * d],
                Init::Zeros,
            ));
            out.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d], lin(d)));
            out.push((format!("{p}.self_attn.out_proj.bias"), vec![d], Init::Zeros));
            out.push((format!("{p}.norm2.weight"), vec![d], Init::Ones));
            out.push((format!("{p}.norm2.bias"), vec![d], Init::Zeros));
            out.push((format!("{p}.linear1.weight"), vec![ff, d], lin(d)));
            out.push((format!("{p}.linear1.bias"), vec![ff], Init::Zeros));
            out.push((format!("{p}.linear2.weight"), vec![d, ff], lin(ff)));
            out.push((format!("{p}.linear2.bias"), vec![d], Init::Zeros));
        }
        out.push(("type_emb.weight".to_string(), vec![3, d], Init::Normal(1.0)));
        out.push(("scorer.0.weight".to_string(), vec![d], Init::Ones));
        out.push(("scorer.0.bias".to_string(), vec![d], Init::Zeros));
        out.push(("scorer.1.weight".to_string(), vec![d, d], lin(d)));
        out.push(("scorer.1.bias".to_string(), vec![d], Init::Zeros));
        out.push(("scorer.3.weight".to_string(), vec![1, d], lin(d)));
        out.push(("scorer.3.bias".to_string(), vec![1], Init::Zeros));
    }
    if parts.mlm {
        out.push((
            "mlm.dense.weight".to_string(),
            vec![d, d],
            Init::TruncNormal(std_out),
        ));
        out.push(("mlm.norm.weight".to_string(), vec![d], Init::Ones));
        out.push(("mlm.decoder.bias".to_string(), vec![v], Init::Zeros));
    }
    out
}

/// Expected number of scalars of a model with the given parts.
pub fn expected_count(cfg: &ModelConfig, parts: Parts) -> usize {
    specs(cfg, parts)
        .iter()
        .map(|(_, s, _)| s.iter().product::<i32>() as usize)
        .sum()
}

/// Initializes a fresh parameter map (f32) deterministically from `seed`.
pub fn init(cfg: &ModelConfig, parts: Parts, seed: u64) -> Result<ParamMap> {
    let mut key = random::key(seed)?;
    let mut map = ParamMap::new();
    for (name, shape, init) in specs(cfg, parts) {
        let (k, next) = random::split(&key, 2)?;
        key = next;
        let a = match init {
            Init::TruncNormal(s) => {
                random::truncated_normal::<f32, f32>(-2.0, 2.0, &shape[..], &k)?
                    .multiply(Array::from_f32(s))?
            }
            Init::Uniform(a) => random::uniform::<f32, f32>(-a, a, &shape[..], &k)?,
            Init::Normal(s) => random::normal::<f32>(&shape[..], 0.0, s, &k)?,
            Init::Ones => mlx_rs::ops::ones::<f32>(&shape)?,
            Init::Zeros => mlx_rs::ops::zeros::<f32>(&shape)?,
        };
        map.insert(name.into(), a);
    }
    eval(&map)?;
    Ok(map)
}

/// Total number of scalars in a parameter map.
pub fn count(map: &ParamMap) -> usize {
    map.values().map(|a| a.size()).sum()
}

/// Number of scalars of the parameters whose name starts with `prefix`.
pub fn count_prefix(map: &ParamMap, prefix: &str) -> usize {
    map.iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(_, a)| a.size())
        .sum()
}

/// Evaluates every array of the map.
pub fn eval(map: &ParamMap) -> Result<()> {
    mlx_rs::transforms::eval(map.values())?;
    Ok(())
}

/// Deep copy (new buffers) of a parameter map.
pub fn deep_clone(map: &ParamMap) -> ParamMap {
    map.iter()
        .map(|(k, v)| (k.clone(), v.deep_clone()))
        .collect()
}

/// Saves arrays to a safetensors file, cast to `dtype` if given.
pub fn save(path: &Path, map: &ParamMap, dtype: Option<Dtype>) -> Result<()> {
    let sorted: BTreeMap<&str, Array> = map
        .iter()
        .map(|(k, v)| -> Result<_> {
            let a = match dtype {
                Some(t) if v.dtype() != t => v.as_dtype(t)?,
                _ => v.clone(),
            };
            Ok((k.as_ref(), a))
        })
        .collect::<Result<_>>()?;
    mlx_rs::transforms::eval(sorted.values())?;
    Array::save_safetensors(sorted.iter().map(|(k, v)| (*k, v)), None, path)
        .map_err(|e| anyhow!("saving {}: {e}", path.display()))?;
    Ok(())
}

/// Loads every array of a safetensors file (as f32 for floating tensors).
pub fn load(path: &Path) -> Result<ParamMap> {
    let raw =
        Array::load_safetensors(path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
    let mut map = ParamMap::new();
    for (k, v) in raw {
        let v = match v.dtype() {
            Dtype::Float16 | Dtype::Bfloat16 => v.as_dtype(Dtype::Float32)?,
            _ => v,
        };
        map.insert(k.into(), v);
    }
    eval(&map)?;
    Ok(map)
}

/// Copies every parameter of `src` whose name exists in `dst` (shapes must match).
/// Returns the number of copied tensors.
pub fn load_matching(dst: &mut ParamMap, src: &ParamMap) -> Result<usize> {
    let mut n = 0;
    for (k, v) in dst.iter_mut() {
        if let Some(s) = src.get(k) {
            if s.shape() != v.shape() {
                return Err(anyhow!(
                    "shape mismatch for {k}: {:?} vs {:?}",
                    s.shape(),
                    v.shape()
                ));
            }
            *v = s.clone();
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    fn ladder(name: &str) -> ModelConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../configs/{name}.json"));
        ModelConfig::load(path).unwrap()
    }

    #[test]
    fn exact_parameter_counts() {
        let _g = test_util::lock();
        // (config, encoder, decision model total, MLM head extra)
        let expected = [
            ("s1-pico", 147_776, 202_305, 5_184),
            ("s1-nano", 918_656, 1_134_209, 18_560),
            ("s1-micro", 4_984_064, 6_630_913, 69_888),
        ];
        for (name, enc, total, mlm) in expected {
            let cfg = ladder(name);
            assert_eq!(expected_count(&cfg, Parts::DECISION), total, "{name}");
            let p = init(&cfg, Parts::DECISION, 0).unwrap();
            assert_eq!(count(&p), total, "{name}");
            assert_eq!(count_prefix(&p, "encoder."), enc, "{name}");
            let m = init(&cfg, Parts::PRETRAIN, 0).unwrap();
            assert_eq!(count(&m), enc + mlm, "{name}");
            println!("{name}: encoder {enc}, decision model {total}, mlm head {mlm}");
        }
    }

    #[test]
    fn names_follow_spec() {
        let _g = test_util::lock();
        let cfg = ladder("s1-micro");
        let p = init(&cfg, Parts::DECISION, 0).unwrap();
        for name in [
            "encoder.embeddings.tok_embeddings.weight",
            "encoder.embeddings.norm.weight",
            "encoder.layers.1.attn_norm.weight",
            "encoder.layers.5.attn.Wqkv.weight",
            "encoder.layers.0.attn.Wo.weight",
            "encoder.layers.0.mlp_norm.weight",
            "encoder.layers.0.mlp.Wi.weight",
            "encoder.layers.0.mlp.Wo.weight",
            "encoder.final_norm.weight",
            "head.layers.1.norm1.bias",
            "head.layers.1.self_attn.in_proj_weight",
            "head.layers.1.self_attn.in_proj_bias",
            "head.layers.0.self_attn.out_proj.weight",
            "head.layers.0.linear2.bias",
            "type_emb.weight",
            "scorer.0.bias",
            "scorer.1.weight",
            "scorer.3.weight",
        ] {
            assert!(p.contains_key(name), "{name}");
        }
        assert!(!p.contains_key("encoder.layers.0.attn_norm.weight"));
        assert_eq!(p["encoder.layers.0.mlp.Wi.weight"].shape(), &[1024, 256]);
        assert_eq!(p["scorer.3.weight"].shape(), &[1, 256]);
        assert_eq!(p["type_emb.weight"].shape(), &[3, 256]);
    }

    #[test]
    fn groups() {
        assert_eq!(
            group_of("encoder.layers.3.attn.Wqkv.weight", 2),
            Group::Muon
        );
        assert_eq!(group_of("encoder.layers.3.mlp.Wo.weight", 2), Group::Muon);
        assert_eq!(
            group_of("encoder.embeddings.tok_embeddings.weight", 2),
            Group::AdamNoDecay
        );
        assert_eq!(group_of("type_emb.weight", 2), Group::AdamNoDecay);
        assert_eq!(
            group_of("head.layers.0.linear1.weight", 2),
            Group::AdamDecay
        );
        assert_eq!(group_of("scorer.1.bias", 1), Group::AdamNoDecay);
        assert_eq!(group_of("encoder.final_norm.weight", 1), Group::AdamNoDecay);
        // s2 names.
        assert_eq!(group_of("blk.2.attn_qkv.weight", 2), Group::Muon);
        assert_eq!(group_of("static.blk.0.ffn_down.weight", 2), Group::Muon);
        assert_eq!(group_of("fusion.0.attn_kv.weight", 2), Group::AdamDecay);
        assert_eq!(group_of("cls.weight", 2), Group::AdamDecay);
        assert_eq!(group_of("token_embd.weight", 2), Group::AdamNoDecay);
        assert_eq!(group_of("num_proj.weight", 2), Group::AdamNoDecay);
        assert_eq!(group_of("type_embd.weight", 2), Group::AdamNoDecay);
        // The per-loop embeddings are embeddings too, not matrices to decay.
        assert_eq!(group_of("loop_embd.weight", 2), Group::AdamNoDecay);
        assert_eq!(group_of("blk.1.attn_norm.weight", 1), Group::AdamNoDecay);
    }
}
