//! s6-nano v3 (`docs/s6-nano3-prereg.md` §2, §10 item 2): the fused router, natively.
//!
//! ```text
//! sky-route embed-gguf --dump <export dir> --out <embedding.gguf>
//!     the embedding file from the export of scripts/s6-route3-embed-export.py
//! sky-route tokens --embed <embedding.gguf> --in <tokens.jsonl>
//!     token ids against the export's tokens.jsonl ({from, text, ids} per line); prints a JSON
//!     summary, exit 1 on any difference
//! sky-route fuse --model <model.gguf> --embed <embedding.gguf> --slates <slates.jsonl>
//!     [--fast] [--max N] [--out dump.jsonl] [--render bm25-line|scores-in-options]
//!     [--ref-fused F] [--ref-model M] [--ref-embedding E]
//!     one JSON line per slate {uid, gold, probs, model, embedding, answer, ask}: the fused
//!     probabilities over the tools and none, the two parts; the model at full depth (--fast:
//!     stopped at its first exit). Slates are routing slates ({opts, state = "User: <request>"},
//!     none last); the BM25 scores are computed here (sky_infer::route) and rendered as the model
//!     file's `skycmd-s2.route.render` says (no key: v2's line; `scores-in-options`: s6-route
//!     v4's D4); --render overrides the file (the latency check of docs/s6-route4-prereg.md 3.4
//!     renders v2's file by D4). Options that already carry `score <s> rank <r>; ` (a model copy
//!     rendered by `sky-data route render`) are read back to `name: description` first, so a
//!     slate file and its model copy give the same routing. Prints a JSON summary; each --ref-*
//!     dump (one JSON line with `probs` per slate, in slate order) adds max_abs_diff and
//!     argmax_agree for that part.
//! ```

#[cfg(feature = "json")]
mod imp {
    use std::io::Write;
    use std::time::Instant;

    use serde_json::{json, Value};
    use sky_gguf::OwnedValue;
    use sky_infer::embed::{self, EmbedSpec, NormTable, StaticEmbedding, ToolCache};
    use sky_infer::{route, Engine, EngineOptions, QType};

    fn die(msg: impl std::fmt::Display) -> ! {
        eprintln!("sky-route: {msg}");
        std::process::exit(2)
    }

    fn read(path: &str) -> Vec<u8> {
        std::fs::read(path).unwrap_or_else(|e| die(format!("{path}: {e}")))
    }

    fn json_file(path: &str) -> Value {
        serde_json::from_slice(&read(path)).unwrap_or_else(|e| die(format!("{path}: {e}")))
    }

    fn jsonl(path: &str) -> Vec<Value> {
        String::from_utf8(read(path))
            .unwrap_or_else(|e| die(format!("{path}: {e}")))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| die(format!("{path}: {e}"))))
            .collect()
    }

    struct Args(Vec<String>);

    impl Args {
        fn value(&self, flag: &str) -> Option<String> {
            self.0.iter().position(|a| a == flag).map(|i| {
                self.0
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| die(format!("{flag} needs a value")))
            })
        }
        fn need(&self, flag: &str) -> String {
            self.value(flag)
                .unwrap_or_else(|| die(format!("{flag} is required")))
        }
        fn has(&self, flag: &str) -> bool {
            self.0.iter().any(|a| a == flag)
        }
    }

    fn u32s(v: &Value) -> Vec<u32> {
        v.as_array()
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect())
            .unwrap_or_default()
    }

    fn ranges(v: &Value) -> Vec<(u32, u32)> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_u64().unwrap() as u32))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn embed_gguf(a: &Args) {
        let dir = a.need("--dump");
        let out = a.need("--out");
        let p = |f: &str| format!("{dir}/{f}");
        let meta = json_file(&p("meta.json"));
        let t = json_file(&p("tables.json"));
        let vocab: Vec<String> = String::from_utf8(read(&p("vocab.txt")))
            .unwrap_or_else(|e| die(e))
            .split('\n')
            .map(str::to_string)
            .collect();
        let scales_b = read(&p("scales.bin"));
        let scales: Vec<u16> = scales_b
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let s = |k: &str| {
            meta[k]
                .as_str()
                .unwrap_or_else(|| die(format!("meta.json: {k}")))
                .to_string()
        };
        let f = |k: &str| {
            meta[k]
                .as_f64()
                .unwrap_or_else(|| die(format!("meta.json: {k}")))
        };
        let src = &meta["source"];
        let ss = |k: &str| {
            src[k]
                .as_str()
                .unwrap_or_else(|| die(format!("meta.json: source.{k}")))
                .to_string()
        };
        let added: Vec<(String, u32)> = meta["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .zip(u32s(&meta["added_ids"]))
            .map(|(t, i)| (t.as_str().unwrap().to_string(), i))
            .collect();
        let m = |k: &str, v: OwnedValue| (k.to_string(), v);
        let spec = EmbedSpec {
            dim: meta["dim"].as_u64().unwrap() as usize,
            codes: read(&p("codes.bin")),
            scales,
            vocab,
            unk_id: meta["unk_id"].as_u64().unwrap() as u32,
            continuing_prefix: s("continuing_prefix"),
            max_chars_per_word: meta["max_input_chars_per_word"].as_u64().unwrap() as usize,
            max_length: meta["max_length"].as_u64().unwrap() as usize,
            normalize: meta["normalize"].as_bool().unwrap(),
            added,
            norm: NormTable {
                starts: u32s(&t["starts"]),
                ops: u32s(&t["ops"]).into_iter().map(|x| x as u8).collect(),
                args: t["args"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect(),
                strings: t["strings"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            },
            space: ranges(&t["space"]),
            punct: ranges(&t["punct"]),
            theta: f("theta"),
            temperature: f("temperature"),
            beta: f("beta"),
            tau_ask: f("tau_ask3"),
            meta: vec![
                m("general.name", "s6-route-embed".into()),
                m("general.description", "the embedding part potion-cos-4bit of the s6-route fused router (docs/s6-nano3-prereg.md section 2): potion-base-2M packed to 4 bits, its tokenizer, theta, temperature, beta and tau_ask3".into()),
                m("general.license", ss("license").into()),
                m("general.source.url", ss("url").into()),
                m("general.source.huggingface.repository", ss("repo").into()),
                m("skycmd-embed.source.commit", ss("commit").into()),
                m("skycmd-embed.source.authors", ss("authors").into()),
                m("skycmd-embed.source.license_from", ss("license_from").into()),
                m("skycmd-embed.source.model_sha256", ss("model_safetensors_sha256").into()),
                m("skycmd-embed.source.tokenizer_sha256", ss("tokenizer_json_sha256").into()),
                m("skycmd-embed.source.libraries", meta["library"].to_string().into()),
                m(embed::keys::SYSTEM, s("system").into()),
                m("skycmd-embed.route.params_from", s("params_from").into()),
                m("skycmd-embed.route.instruction", route::INSTRUCTION.into()),
                m("skycmd-embed.route.none_option", route::NONE_OPTION.into()),
                m("skycmd-embed.export", "scripts/s6-route3-embed-export.py, then sky-route embed-gguf".into()),
            ],
        };
        let bytes = spec.to_gguf().unwrap_or_else(|e| die(e));
        // a loading check before writing
        let e = StaticEmbedding::from_gguf(&bytes).unwrap_or_else(|e| die(e));
        if let Some(parent) = std::path::Path::new(&out).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&out, &bytes).unwrap_or_else(|e| die(format!("{out}: {e}")));
        println!(
            "{}",
            json!({"out": out, "bytes": bytes.len(), "n_vocab": e.n_vocab(), "dim": e.dim(),
                   "packed_bytes": spec.codes.len() + 2 * spec.scales.len()})
        );
    }

    fn tokens(a: &Args) {
        let e = StaticEmbedding::from_gguf(&read(&a.need("--embed"))).unwrap_or_else(|e| die(e));
        let rows = jsonl(&a.need("--in"));
        let mut by: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        let mut examples = Vec::new();
        let mut differ = 0;
        for r in &rows {
            let text = r["text"].as_str().unwrap_or("");
            let want = u32s(&r["ids"]);
            let got = e.token_ids(text);
            let ent = by
                .entry(r["from"].as_str().unwrap_or("?").to_string())
                .or_default();
            ent.0 += 1;
            if got != want {
                ent.1 += 1;
                differ += 1;
                if examples.len() < 10 {
                    examples.push(json!({"text": text.chars().take(200).collect::<String>(), "want": want, "got": got}));
                }
            }
        }
        let by: serde_json::Map<String, Value> = by
            .into_iter()
            .map(|(k, (n, d))| (k, json!({"n": n, "differ": d})))
            .collect();
        println!(
            "{}",
            json!({"n": rows.len(), "differ": differ, "by_source": by, "examples": examples})
        );
        std::process::exit(if differ == 0 { 0 } else { 1 });
    }

    fn request_of(state: &str) -> &str {
        let s = state.rsplit_once("\nBM25: ").map_or(state, |(a, _)| a);
        s.strip_prefix("User: ").unwrap_or(s)
    }

    fn probs(v: &Value) -> Vec<f64> {
        v["probs"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default()
    }

    #[derive(Default)]
    struct Cmp {
        max: f64,
        agree: usize,
        n: usize,
    }

    impl Cmp {
        fn add(&mut self, a: &[f64], b: &[f64]) {
            if a.len() != b.len() {
                die("a reference dump has another number of options");
            }
            for (x, y) in a.iter().zip(b) {
                self.max = self.max.max((x - y).abs());
            }
            self.agree += (embed::argmax(a) == embed::argmax(b)) as usize;
            self.n += 1;
        }
        fn json(&self) -> Value {
            json!({"n": self.n, "max_abs_diff": self.max, "argmax_agree": self.agree})
        }
    }

    fn fuse(a: &Args) {
        let model = read(&a.need("--model"));
        let render = match a.value("--render") {
            Some(r) => route::Render::parse(&r).unwrap_or_else(|| {
                die(format!(
                    "--render {r}: {} or {}",
                    route::RENDER_LINE,
                    route::RENDER_SCORES_IN_OPTIONS
                ))
            }),
            None => route::Render::of_gguf_bytes(&model).unwrap_or_else(|e| die(e)),
        };
        let e = StaticEmbedding::from_gguf(&read(&a.need("--embed"))).unwrap_or_else(|e| die(e));
        let mut eng =
            Engine::from_gguf_with(&model, EngineOptions::default()).unwrap_or_else(|e| die(e));
        let fast = a.has("--fast");
        eng.set_exit_threshold(Some(if fast { 0.0 } else { 1.0 }));
        let max: usize = a
            .value("--max")
            .map_or(usize::MAX, |v| v.parse().unwrap_or_else(|e| die(e)));
        let slates: Vec<Value> = jsonl(&a.need("--slates")).into_iter().take(max).collect();
        let refs: Vec<(&str, Option<Vec<Value>>)> = ["fused", "model", "embedding"]
            .iter()
            .map(|k| (*k, a.value(&format!("--ref-{k}")).map(|p| jsonl(&p))))
            .collect();
        let mut cmp: Vec<Cmp> = refs.iter().map(|_| Cmp::default()).collect();
        let mut out = a.value("--out").map(|p| {
            std::io::BufWriter::new(
                std::fs::File::create(&p).unwrap_or_else(|e| die(format!("{p}: {e}"))),
            )
        });
        let mut cache = ToolCache::default();
        let (mut hit, mut hit_m, mut hit_e, mut asked, mut ok_answered) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut us = Vec::new();
        for (k, r) in slates.iter().enumerate() {
            let opts: Vec<String> = r["opts"]
                .as_array()
                .unwrap_or_else(|| die("a slate without opts"))
                .iter()
                .map(|o| route::plain_option(o.as_str().unwrap_or("")).to_string())
                .collect();
            let tools = &opts[..opts.len().saturating_sub(1)];
            let request = request_of(r["state"].as_str().unwrap_or(""));
            let t0 = Instant::now();
            let (model_opts, state) = render.model_input(request, &opts);
            let pm: Vec<f64> = eng
                .decide_raw(QType::Choice, route::INSTRUCTION, &model_opts, &state)
                .unwrap_or_else(|e| die(e))
                .into_iter()
                .map(f64::from)
                .collect();
            let (pe, _) = cache.probs(&e, request, tools);
            let pf = embed::fuse(&pm, &pe, e.beta);
            us.push(t0.elapsed().as_secs_f64() * 1e6);
            let ans = embed::answer(&pf, e.tau_ask);
            let gold = r["target"]
                .as_array()
                .map(|t| embed::argmax(&t.iter().filter_map(Value::as_f64).collect::<Vec<_>>()))
                .unwrap_or(usize::MAX);
            hit += (ans.index == gold) as usize;
            hit_m += (embed::argmax(&pm) == gold) as usize;
            hit_e += (embed::argmax(&pe) == gold) as usize;
            if ans.ask {
                asked += 1;
            } else {
                ok_answered += (ans.index == gold) as usize;
            }
            for (i, (_, rows)) in refs.iter().enumerate() {
                if let Some(rows) = rows {
                    let want = probs(
                        rows.get(k)
                            .unwrap_or_else(|| die("a reference dump is shorter than the slates")),
                    );
                    let got = [&pf, &pm, &pe][i];
                    cmp[i].add(got, &want);
                }
            }
            if let Some(w) = out.as_mut() {
                let line = json!({"uid": r["uid"], "gold": gold, "probs": pf, "model": pm, "embedding": pe,
                                  "answer": ans.index, "ask": ans.ask});
                writeln!(w, "{line}").unwrap_or_else(|e| die(e));
            }
        }
        let n = slates.len().max(1) as f64;
        us.sort_by(f64::total_cmp);
        let mut summary = json!({
            "n": slates.len(), "fast": fast, "render": render.name(), "beta": e.beta, "tau_ask3": e.tau_ask,
            "accuracy": hit as f64 / n, "accuracy_model": hit_m as f64 / n, "accuracy_embedding": hit_e as f64 / n,
            "coverage": (slates.len() - asked) as f64 / n,
            "accuracy_answered": ok_answered as f64 / (slates.len() - asked).max(1) as f64,
            "median_native_us": us.get(us.len() / 2).copied().unwrap_or(0.0),
        });
        for ((name, rows), c) in refs.iter().zip(&cmp) {
            if rows.is_some() {
                summary[format!("vs_ref_{name}")] = c.json();
            }
        }
        println!("{summary}");
    }

    pub fn main() {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let Some(cmd) = argv.first().cloned() else {
            die("usage: sky-route <embed-gguf|tokens|fuse> ... (see the source header)")
        };
        let a = Args(argv[1..].to_vec());
        match cmd.as_str() {
            "embed-gguf" => embed_gguf(&a),
            "tokens" => tokens(&a),
            "fuse" => fuse(&a),
            other => die(format!(
                "unknown command {other} (embed-gguf, tokens, fuse)"
            )),
        }
    }
}

fn main() {
    #[cfg(feature = "json")]
    imp::main();
    #[cfg(not(feature = "json"))]
    eprintln!("sky-route needs the json feature");
}
