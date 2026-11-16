//! The fused tool router of s6-nano v3 (`docs/s6-nano3-prereg.md` §2): the v2 router file (the
//! model part) and the 4-bit static embedding file (the embedding part, `sky_infer::embed`),
//! fused option by option, `p(i) ∝ (p_model(i) · p_embedding(i))^β`, with "ask the user" when the
//! top probability is below τ_ask3 (β and τ_ask3 come from the embedding file).
//!
//! A request is routed against a tool list of rendered options `name: description`; *none* is
//! appended. The tool list is encoded once and cached on both parts: the model's static stream
//! (the s2 engine's LRU, keyed by the question and the options) and the tools' embeddings (the
//! last tool list). The model runs at full depth; the fast mode stops it at its first exit.

use serde_json::{json, Value};
use sky_infer::embed::{self, StaticEmbedding, ToolCache};
use sky_infer::{route, QType};

use crate::clock;
use crate::model::Model;

/// The fused router.
pub struct Router {
    model: Model,
    embed: StaticEmbedding,
    tools: ToolCache,
}

/// One routed request.
#[derive(Clone, Debug)]
pub struct Routed {
    /// Fused probabilities over the tools then *none*.
    pub probs: Vec<f64>,
    /// The model part's probabilities.
    pub model: Vec<f64>,
    /// The embedding part's probabilities.
    pub embedding: Vec<f64>,
    pub answer: embed::RouteAnswer,
    pub exit: Option<usize>,
    pub layers_used: Option<usize>,
    /// The model's static stream was cached.
    pub static_hit: bool,
    /// The tools' embeddings were cached.
    pub tools_hit: bool,
    pub micros: f64,
    pub micros_model: f64,
    pub micros_embedding: f64,
}

impl Router {
    /// From the bytes of the model file and of the embedding file.
    pub fn load(model_gguf: &[u8], embed_gguf: &[u8]) -> Result<Router, String> {
        let model = Model::load(model_gguf)?;
        if model.engine().s2_model().is_none() {
            return Err("the model part must be an s2 router file".into());
        }
        let embed = StaticEmbedding::from_gguf(embed_gguf).map_err(|e| e.to_string())?;
        Ok(Router {
            model,
            embed,
            tools: ToolCache::default(),
        })
    }

    pub fn embedding(&self) -> &StaticEmbedding {
        &self.embed
    }

    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Route `request` against `tools` (rendered `name: description`, without *none*).
    pub fn route(&mut self, request: &str, tools: &[String], fast: bool) -> Result<Routed, String> {
        if tools.is_empty() {
            return Err("at least one tool is needed".into());
        }
        let t0 = clock::now_us();
        let options = embed::model_options(tools);
        let state = route::state_for_options(request, &options);
        self.model
            .set_exit_threshold(Some(if fast { 0.0 } else { 1.0 }));
        let pm: Vec<f64> = self
            .model
            .probs(QType::Choice, route::INSTRUCTION, &options, &state)?
            .into_iter()
            .map(f64::from)
            .collect();
        let st = self.model.last_stats().cloned().unwrap_or_default();
        let t1 = clock::now_us();
        let (pe, tools_hit) = self.tools.probs(&self.embed, request, tools);
        let probs = embed::fuse(&pm, &pe, self.embed.beta);
        let t2 = clock::now_us();
        let answer = embed::answer(&probs, self.embed.tau_ask);
        Ok(Routed {
            probs,
            model: pm,
            embedding: pe,
            answer,
            exit: st.exit,
            layers_used: st.layers_used,
            static_hit: st.static_hit,
            tools_hit,
            micros: t2 - t0,
            micros_model: t1 - t0,
            micros_embedding: t2 - t1,
        })
    }

    /// [`Router::route`] as the JSON answer of the wasm API.
    pub fn route_value(
        &mut self,
        request: &str,
        tools_json: &str,
        fast: bool,
    ) -> Result<Value, String> {
        let tools: Vec<String> =
            serde_json::from_str(tools_json).map_err(|e| format!("tools: {e}"))?;
        let r = self.route(request, &tools, fast)?;
        let a = &r.answer;
        let name = if a.none {
            "none".to_string()
        } else {
            route::option_name(&tools[a.index]).to_string()
        };
        Ok(json!({
            "probs": r.probs,
            "index": a.index,
            "answer": name,
            "none": a.none,
            "ask": a.ask,
            "decision": if a.ask { "ask" } else if a.none { "none" } else { "tool" },
            "top": a.top,
            "tau_ask3": self.embed.tau_ask,
            "model": r.model,
            "embedding": r.embedding,
            "fast": fast,
            "exit": r.exit,
            "layers_used": r.layers_used,
            "static_hit": r.static_hit,
            "tools_hit": r.tools_hit,
            "micros": r.micros,
            "micros_model": r.micros_model,
            "micros_embedding": r.micros_embedding,
        }))
    }

    /// `{model, embedding: {name, file_bytes, dim, n_vocab, theta, temperature, beta, tau_ask3}}`.
    pub fn info(&mut self) -> Value {
        let e = &self.embed;
        json!({
            "model": self.model.info(),
            "embedding": {"name": e.name(), "file_bytes": e.file_bytes(), "dim": e.dim(), "n_vocab": e.n_vocab(),
                          "theta": e.theta, "temperature": e.temperature, "beta": e.beta, "tau_ask3": e.tau_ask},
        })
    }
}
