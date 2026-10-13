//! The s5 stages. Relative paths resolve in the store (the working directory of every step).

pub struct Step {
    pub name: String,
    /// One of our binaries (built from the checkout) or a program on the PATH.
    pub prog: &'static str,
    pub args: Vec<String>,
    /// Holds the GPU lock while it runs.
    pub gpu: bool,
    pub env: Vec<(String, String)>,
}

impl Step {
    pub fn new(name: &str, prog: &'static str, args: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            prog,
            args: args.iter().map(|s| s.to_string()).collect(),
            gpu: false,
            env: Vec::new(),
        }
    }

    pub fn gpu(mut self) -> Self {
        self.gpu = true;
        self
    }
}

pub struct Stage {
    pub name: &'static str,
    pub about: &'static str,
    pub steps: Vec<Step>,
}

/// A pinned Hub or GitHub download (`sky-data fetch-model`).
fn fetch(name: &str, repo: &str, rev: &str, out: &str, kind: &str, include: &str) -> Step {
    let mut args = vec![
        "--cache",
        "data/hf",
        "fetch-model",
        repo,
        "--rev",
        rev,
        "--out",
        out,
    ];
    if !kind.is_empty() {
        args.push(kind);
    }
    if !include.is_empty() {
        args.extend(["--include", include]);
    }
    Step::new(name, "sky-data", &args)
}

/// Encoders, the reranker initialisation and the Apache-2.0 teachers (Phase 0).
/// Revisions are pinned to the commits checked on 2026-10-13.
fn fetch_models() -> Vec<Step> {
    const HF_MODEL: &str = "";
    let weights = "config.json,tokenizer*.json,special_tokens_map.json,pytorch_model.bin";
    vec![
        // The golden reference of sky-infer (tests/golden.rs, tests/tokenizer.rs).
        fetch(
            "oscar-1-17m",
            "mgoeckel/oscar-1-17m",
            "a443c18107534cb3148941de601e15a89a2c6da3",
            "models-src/oscar",
            HF_MODEL,
            "",
        ),
        fetch(
            "ettin-17m",
            "jhu-clsp/ettin-encoder-17m",
            "987607455c61e7a5bbc85f7758e0512ea6d0ae4c",
            "models-src/ettin-17m",
            HF_MODEL,
            weights,
        ),
        fetch(
            "ettin-32m",
            "jhu-clsp/ettin-encoder-32m",
            "1b8ba06455dd44f80fc9c1ca9e22806157a57379",
            "models-src/ettin-32m",
            HF_MODEL,
            weights,
        ),
        fetch(
            "ettin-reranker-17m",
            "cross-encoder/ettin-reranker-17m-v1",
            "9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6",
            "models-src/ettin-reranker-17m",
            HF_MODEL,
            "config.json,tokenizer*.json,model.safetensors,README.md",
        ),
        fetch(
            "mmbert-small",
            "jhu-clsp/mmBERT-small",
            "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
            "models-src/mmbert-small",
            HF_MODEL,
            weights,
        ),
        fetch(
            "laya-typed-decisions",
            "convaiinnovations/laya-typed-decisions",
            "e929ae5cf69bc34259cd2f95c9e91145b818b1f0",
            "models-src/laya-typed-decisions",
            HF_MODEL,
            "",
        ),
        fetch(
            "laya-multilingual",
            "convaiinnovations/laya-multilingual",
            "1720e3e3357cfe1e281542e223f8273b0890ca34",
            "models-src/laya-multilingual",
            HF_MODEL,
            "",
        ),
        fetch(
            "oscar-1-150m",
            "mgoeckel/oscar-1-150m",
            "1015773b7b1e3dcab42029db39efeffaed31cfce",
            "models-src/oscar-150m",
            HF_MODEL,
            "",
        ),
        fetch(
            "oscar-1-400m",
            "mgoeckel/oscar-1-400m",
            "d0a87f7adcf5ad3dbb8616eafe6a1c2d2e667f06",
            "models-src/oscar-400m",
            HF_MODEL,
            "",
        ),
    ]
}

/// The typed-decisions benchmark (train and test), the zero-shot evaluation sets and the
/// 2609.33209 coherence battery (evaluation only, never redistributed).
fn fetch_evals() -> Vec<Step> {
    vec![
        fetch(
            "typed-decisions",
            "LocalLLaMA/typed-decisions",
            "c806f6c69d734f29436f25536138547031a673d6",
            "data/s5/src/typed-decisions",
            "--dataset",
            "",
        ),
        fetch(
            "btzsc",
            "btzsc/btzsc",
            "fef2a2ac62b69c58670047dddf045c53d7c3cb5e",
            "data/s5/src/btzsc",
            "--dataset",
            "README.md,wikitoxic_*/*,biasframes_offensive/*,biasframes_sex/*,biasframes_intent/*,\
             amazonpolarity/*,yahootopics/*",
        ),
        fetch(
            "sib200-fr",
            "Davlan/sib200",
            "38977a667f6fc264d5c26ec57a01e16db040b358",
            "data/s5/src/sib200",
            "--dataset",
            "README.md,data/fra_Latn/*,data/eng_Latn/*",
        ),
        fetch(
            "mtop-fr",
            "mteb/mtop_intent",
            "9a200bf94b01afa0438a39c54f6a3aceaf78856f",
            "data/s5/src/mtop",
            "--dataset",
            "README.md,fr/*,en/*",
        ),
        fetch(
            "hwu64",
            "DeepPavlov/hwu64",
            "0dd289ccdeb185ec065d1ebcf5de1c443cd1620f",
            "data/s5/src/hwu64",
            "--dataset",
            "README.md,data/*,intents/intents-00000-of-00001.parquet",
        ),
        fetch(
            "coherence-battery",
            "bro789/typed-decision-coherence",
            "8898ef7bc3c151c7b4827a6afbc294907a1c7a64",
            "data/s5/src/coherence",
            "--github",
            "LICENSE,README.md,02_probability_axioms/stimuli/*,02_probability_axioms/analyze.py,\
             _common/*",
        ),
    ]
}

/// The pretrained encoders as sky-train checkpoints (`models/s5-base/`), each with its MLM
/// probe printed in the log.
fn import_bases() -> Vec<Step> {
    [
        ("ettin-17m", "models-src/ettin-17m"),
        ("ettin-32m", "models-src/ettin-32m"),
        ("ettin-reranker-17m", "models-src/ettin-reranker-17m"),
        ("mmbert-small", "models-src/mmbert-small"),
    ]
    .into_iter()
    .map(|(name, src)| {
        let out = format!("models/s5-base/{name}");
        Step::new(
            &format!("import-{name}"),
            "sky-train",
            &["import-hf", "--src", src, "--out", &out, "--name", name],
        )
        .gpu()
    })
    .collect()
}

/// The s5 data (`data/s5/`): the typed-decisions train split, then `sky-data s5-build`.
fn data_steps() -> Vec<Step> {
    vec![
        Step::new(
            "typed-decisions-train",
            "sky-data",
            &[
                "--cache",
                "data/hf",
                "fetch-decisions",
                "--only",
                "typed_decisions_train",
            ],
        ),
        Step::new(
            "s5-build",
            "sky-data",
            &[
                "s5-build",
                "--defs",
                "{checkout}/eval/s5/defs.json",
                "--demo",
                "{checkout}/eval/s5/demo-cases.jsonl",
                "--langs",
                "en,fr",
            ],
        ),
        // The original 2609.33209 battery, for the comparison with Jev (evaluation only).
        Step::new("s5-battery", "sky-data", &["s5-battery"]),
    ]
}

/// Teacher soft labels of the English records that `s5-mix` labels (`data/s5/scored/`): the
/// corpus and the pairs, two option orders averaged (generated records and coherence families
/// keep their exact gold targets). Only laya-typed-decisions: oscar-1-150m (fetched, a close
/// second on the bench) would have cost about 45 more minutes of GPU, which the s5-lite budget of
/// 2026-10-13 left out; with one teacher there is no teacher to choose per group, so the gold val
/// records are not scored in both renderings either (docs/s5-prereg.md).
fn teacher_steps() -> Vec<Step> {
    let (name, dir) = ("laya-td", "models-src/laya-typed-decisions");
    ["corpus", "pairs"]
        .into_iter()
        .map(|file| {
            let out = format!("data/s5/scored/{name}.native.{file}.jsonl");
            let rec = format!("data/s5/train/{file}.jsonl");
            Step::new(
                &format!("score-{name}-{file}"),
                "sky-train",
                &[
                    "score",
                    "--teacher",
                    dir,
                    "--records",
                    &rec,
                    "--out",
                    &out,
                    "--render",
                    "native",
                    "--perms",
                    "2",
                    "--lang",
                    "en",
                    "--bf16",
                ],
            )
            .gpu()
        })
        .collect()
}

/// oscar-1-17m, the public Laya-format reference, on the reassignment set (the yes/no swap of
/// 2609.26758, next to the iso guarantee) and on the fitted bench (our rendering of the s5 records
/// must give back its published 0.6775). Descriptive, outside the bets (amendment 1).
fn oscar_ref() -> Vec<Step> {
    let (gguf, dir) = (
        "models/s5-ref/oscar-1-17m.f32.gguf",
        "runs/s5-ref/oscar-1-17m/eval-f32",
    );
    let mut v = vec![
        Step::new("mkdir", "mkdir", &["-p", "models/s5-ref", dir]),
        Step::new(
            "convert",
            "sky-convert",
            &[
                "models-src/oscar",
                gguf,
                "--type",
                "f32",
                "--head-type",
                "f32",
            ],
        ),
    ];
    for set in ["typed_decisions_test", "reassign_test"] {
        v.push(Step::new(
            &format!("test-{set}"),
            "sky-eval",
            &[
                gguf,
                "--records",
                &format!("data/s5/eval/{set}.jsonl"),
                "--split",
                "test",
                "--prereg",
                "{checkout}/docs/s5-prereg.md",
                "--metrics",
                "--json",
                &format!("{dir}/{set}.json"),
                "--dump-probs",
                &format!("{dir}/{set}.dump.jsonl"),
            ],
        ));
    }
    v.push(Step::new(
        "mkdir-report",
        "mkdir",
        &["-p", "runs/s5-report"],
    ));
    v.push(Step::new(
        "robust",
        "sky-s5-report",
        &[
            "robust",
            dir,
            "--json",
            "runs/s5-report/robust-oscar-1-17m.json",
        ],
    ));
    // Per-set accuracy (the bench check); its gate verdicts mean nothing for a reference.
    v.push(Step::new(
        "sets",
        "sky-s5-report",
        &[
            "gates",
            dir,
            "--records",
            "data/s5/eval",
            "--json",
            "runs/s5-report/sets-oscar-1-17m.json",
        ],
    ));
    v
}

pub fn stages() -> Vec<Stage> {
    vec![
        Stage {
            name: "fetch-models",
            about: "encoders, reranker and teachers into models-src/ (pinned, hash-checked)",
            steps: fetch_models(),
        },
        Stage {
            name: "fetch-evals",
            about: "typed-decisions, zero-shot sets and the coherence battery into data/s5/src/",
            steps: fetch_evals(),
        },
        Stage {
            name: "data",
            about:
                "typed-decisions train split, then the s5 corpus, generators, eval sets (data/s5/)",
            steps: data_steps(),
        },
        Stage {
            name: "teachers",
            about: "soft labels of laya-typed-decisions on the English records",
            steps: teacher_steps(),
        },
        Stage {
            name: "mix",
            about: "training targets: gold + calibrated teacher soft labels (data/s5/mixed/)",
            steps: vec![Step::new("s5-mix", "sky-data", &["s5-mix"])],
        },
        Stage {
            name: "import",
            about: "the encoders as sky-train checkpoints in models/s5-base/ (MLM probe logged)",
            steps: import_bases(),
        },
        Stage {
            name: "oscar-ref",
            about: "oscar-1-17m on the reassignment set and the bench (descriptive reference)",
            steps: oscar_ref(),
        },
    ]
}
