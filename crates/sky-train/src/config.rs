//! Model configuration: the ModernBERT `encoder/config.json` keys plus a Laya
//! `rl_agent_config.json`-style block (`rl_agent`).

use std::{collections::BTreeMap, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Head dimension of every attention layer (encoder and decision head).
pub const HEAD_DIM: i32 = 64;

/// Fixed special token ids of the skycmd tokenizers.
pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const CLS_ID: u32 = 2;
pub const SEP_ID: u32 = 3;
pub const MASK_ID: u32 = 4;
/// Number of special tokens at the start of the vocabulary.
pub const N_SPECIAL: u32 = 5;

/// Laya runtime block (the `rl_agent_config.json` keys we own).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    #[serde(default = "default_temperature")]
    pub temperature: [f32; 3],
    /// Keys use the Laya `temp_bucket` format, e.g. `"choice:3-5"`.
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f32>,
}

fn default_temperature() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// Full model configuration (encoder + decision head).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Ladder name, e.g. `s1-pico`.
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
    #[serde(default = "default_eps")]
    pub norm_eps: f32,
    /// Laya decision block.
    pub rl_agent: AgentConfig,
    /// Every other ModernBERT key, kept verbatim for `encoder/config.json`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
    /// Run matmuls and activations in bfloat16 (f32 master weights, f32 losses).
    /// A runtime setting, never serialized.
    #[serde(skip)]
    pub bf16: bool,
}

fn default_eps() -> f32 {
    1e-5
}

impl ModelConfig {
    /// Loads and validates a config file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Checks the invariants the model code relies on.
    pub fn validate(&self) -> Result<()> {
        if !self.hidden_size.is_multiple_of(HEAD_DIM as usize) {
            bail!(
                "hidden_size {} is not a multiple of {HEAD_DIM}",
                self.hidden_size
            );
        }
        if self.num_attention_heads != self.n_heads() {
            bail!(
                "num_attention_heads {} must be hidden_size / {HEAD_DIM} = {}",
                self.num_attention_heads,
                self.n_heads()
            );
        }
        if self.global_attn_every_n_layers == 0 {
            bail!("global_attn_every_n_layers must be >= 1");
        }
        if self.vocab_size <= N_SPECIAL as usize {
            bail!("vocab_size too small");
        }
        Ok(())
    }

    /// Number of attention heads (head_dim is always 64).
    pub fn n_heads(&self) -> usize {
        self.hidden_size / HEAD_DIM as usize
    }

    /// True when encoder layer `l` uses global attention.
    pub fn is_global(&self, l: usize) -> bool {
        l.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// RoPE base of encoder layer `l`.
    pub fn rope_theta(&self, l: usize) -> f32 {
        if self.is_global(l) {
            self.global_rope_theta
        } else {
            self.local_rope_theta
        }
    }

    /// The HF ModernBERT `encoder/config.json` of this model.
    pub fn encoder_config_json(&self) -> Value {
        let mut m = Map::new();
        let defaults = json!({
            "architectures": ["ModernBertForMaskedLM"],
            "model_type": "modernbert",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "causal_mask": false,
            "classifier_activation": "gelu",
            "classifier_bias": false,
            "classifier_dropout": 0.0,
            "classifier_pooling": "mean",
            "decoder_bias": true,
            "deterministic_flash_attn": false,
            "dtype": "float32",
            "embedding_dropout": 0.0,
            "gradient_checkpointing": false,
            "hidden_activation": "gelu",
            "initializer_cutoff_factor": 2.0,
            "initializer_range": 0.02,
            "is_causal": false,
            "mlp_bias": false,
            "mlp_dropout": 0.0,
            "norm_bias": false,
            "position_embedding_type": "sans_pos",
            "repad_logits_with_grad": false,
            "sparse_pred_ignore_index": -100,
            "sparse_prediction": false,
            "pad_token_id": PAD_ID,
            "cls_token_id": CLS_ID,
            "sep_token_id": SEP_ID,
            "bos_token_id": CLS_ID,
            "eos_token_id": SEP_ID,
        });
        if let Value::Object(d) = defaults {
            m.extend(d);
        }
        m.extend(self.extra.clone());
        let own = json!({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "global_attn_every_n_layers": self.global_attn_every_n_layers,
            "local_attention": self.local_attention,
            "global_rope_theta": self.global_rope_theta,
            "local_rope_theta": self.local_rope_theta,
            "norm_eps": self.norm_eps,
            "layer_norm_eps": self.norm_eps,
            "max_position_embeddings": self.rl_agent.max_len,
        });
        if let Value::Object(o) = own {
            m.extend(o);
        }
        Value::Object(m)
    }

    /// A tiny configuration used by unit tests.
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            name: "tiny".into(),
            vocab_size,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            intermediate_size: 32,
            global_attn_every_n_layers: 2,
            local_attention: 4,
            global_rope_theta: 160_000.0,
            local_rope_theta: 10_000.0,
            norm_eps: 1e-5,
            rl_agent: AgentConfig {
                head_layers: 1,
                max_len: 64,
                head_max_len: 32,
                temperature: [1.0; 3],
                temperature_by_options: BTreeMap::new(),
            },
            extra: Map::new(),
            bf16: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder(name: &str) -> ModelConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs")
            .join(format!("{name}.json"));
        ModelConfig::load(path).unwrap()
    }

    #[test]
    fn ladder_configs_match_spec_table() {
        // name, d, L, heads, intermediate, head_layers, vocab
        let table = [
            ("s1-pico", 64, 2, 1, 128, 1, 1024),
            ("s1-nano", 128, 4, 2, 256, 1, 2048),
            ("s1-micro", 256, 6, 4, 512, 2, 4096),
        ];
        for (name, d, l, h, i, hl, v) in table {
            let c = ladder(name);
            assert_eq!(c.name, name);
            assert_eq!(
                (c.hidden_size, c.num_hidden_layers, c.num_attention_heads),
                (d, l, h)
            );
            assert_eq!(
                (c.intermediate_size, c.rl_agent.head_layers, c.vocab_size),
                (i, hl, v)
            );
            assert_eq!((c.global_attn_every_n_layers, c.local_attention), (3, 128));
            assert_eq!(
                (c.global_rope_theta, c.local_rope_theta),
                (160_000.0, 10_000.0)
            );
            assert_eq!((c.rl_agent.max_len, c.rl_agent.head_max_len), (256, 96));
        }
    }

    #[test]
    fn encoder_config_has_hf_keys() {
        let c = ladder("s1-pico");
        let e = c.encoder_config_json();
        assert_eq!(e["model_type"], "modernbert");
        assert_eq!(e["hidden_size"], 64);
        assert_eq!(e["local_rope_theta"], 10000.0);
        assert_eq!(e["norm_bias"], false);
        assert!(e.get("rl_agent").is_none());
        assert!(e.get("name").is_none());
    }
}
