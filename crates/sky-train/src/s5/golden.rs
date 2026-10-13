//! `sky-train s5-golden`: reference outputs of the iso model for sky-infer. A small random iso
//! model (byte-level tokenizer, 3 layers, local window 8) is written for each position scheme as
//! `iso-shared.gguf` / `iso-reset.gguf`, with `iso.json` holding the cases (s1-rendered options,
//! as `sky-eval` passes them) and the mlx logits. sky-infer and the wasm build must match them.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use sky_gguf::GgmlType;

use crate::config::{ModelConfig, SpecialTokens};
use crate::params::{self, Parts};
use crate::s5::data::tokenize;
use crate::s5::export::to_gguf;
use crate::s5::iso::{init_head, Positions};
use crate::s5::record::S5Record;
use crate::s5::train::predict_iso;
use crate::sequence::{byte_tokenizer_json, DecisionRecord, Tok};

#[derive(Debug, Clone, clap::Args)]
pub struct GoldenArgs {
    #[arg(long, default_value = "crates/sky-infer/tests/golden")]
    pub out: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub seed: u64,
}

/// The configuration of the golden model.
pub fn golden_config() -> ModelConfig {
    let mut c = ModelConfig::tiny(261);
    c.name = "iso-golden".into();
    c.num_hidden_layers = 3;
    c.global_attn_every_n_layers = 3;
    c.local_attention = 8;
    c.intermediate_size = 96;
    c.rl_agent.max_len = 160;
    c.rl_agent.head_max_len = 64;
    c
}

fn case(t: &str, ins: &str, opts: &[&str], state: &str) -> DecisionRecord {
    DecisionRecord {
        src: "golden".into(),
        t: t.into(),
        ins: ins.into(),
        opts: opts.iter().map(|s| s.to_string()).collect(),
        state: state.into(),
        target: vec![1.0; opts.len()],
        split: "test".into(),
    }
}

/// The golden cases (rendered like the s1 records).
pub fn cases() -> Vec<DecisionRecord> {
    let long: String = (0..12)
        .map(|i| format!("step {i}: altitude {}, speed {}. ", 10 + 3 * i, i % 5))
        .collect();
    vec![
        case(
            "choice",
            "Flap now or wait?",
            &["flap: push the bird up", "wait: let the bird fall"],
            "alt 12 vy -3 dx 40 gt 20 gb 35",
        ),
        case(
            "choice",
            "Which way?",
            &["up", "down", "left", "right"],
            "x 4 y 7",
        ),
        case(
            "score",
            "How severe is it?",
            &["level 0: harmless", "level 1: minor", "level 2: serious"],
            "a server is down",
        ),
        case(
            "noul",
            "The alert is real.",
            &["false: benign activity", "true: malicious activity"],
            &long,
        ),
        case("noul", "It will rain.", &["false", "true"], "clouds"),
        case(
            "choice",
            "Je suis fatigué",
            &["dormir: pour se reposer", "faire la fete: pour kiffer"],
            "je suis ko",
        ),
    ]
}

pub fn golden(a: &GoldenArgs) -> Result<()> {
    let cfg = golden_config();
    let tok_json = byte_tokenizer_json();
    let tmp = std::env::temp_dir().join(format!("sky-s5-golden-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)?;
    let tok_path = tmp.join("tokenizer.json");
    std::fs::write(&tok_path, &tok_json)?;
    let tok = Tok::load(&tok_path)?;
    let sp = SpecialTokens::default();
    let mut p = params::init(
        &cfg,
        Parts {
            decision: false,
            mlm: false,
        },
        a.seed,
    )?;
    p.extend(init_head(&cfg, a.seed + 1)?);
    let recs: Vec<S5Record> = cases()
        .iter()
        .enumerate()
        .map(|(i, d)| S5Record::from_decision(d, &format!("g{i}"), ""))
        .collect();
    let items = tokenize(&tok, &sp, recs)?;
    let refs: Vec<_> = items.iter().collect();
    std::fs::create_dir_all(&a.out)?;
    let mut logits = serde_json::Map::new();
    for (pos, name) in [(Positions::Shared, "shared"), (Positions::Reset, "reset")] {
        let bytes = to_gguf(&p, &cfg, pos, &[], &tok_json, GgmlType::F32, "iso-golden")?;
        std::fs::write(a.out.join(format!("iso-{name}.gguf")), &bytes)?;
        let preds = mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            predict_iso(&p, &cfg, pos, &refs, 1 << 20)
        })?;
        logits.insert(
            name.into(),
            json!(preds.iter().map(|p| p.logits.clone()).collect::<Vec<_>>()),
        );
    }
    let doc = json!({
        "meta": {"seed": a.seed, "tolerance": 1e-4, "generator": "sky-train s5-golden"},
        "cases": cases(),
        "logits": logits,
    });
    std::fs::write(
        a.out.join("iso.json"),
        serde_json::to_string_pretty(&doc)? + "\n",
    )?;
    std::fs::remove_dir_all(&tmp).ok();
    println!(
        "wrote iso-shared.gguf, iso-reset.gguf and iso.json to {}",
        a.out.display()
    );
    Ok(())
}
