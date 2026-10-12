//! s2 ladder configurations (SPEC-S2 §3) and GGUF files with random weights, for tests and
//! benchmarks before trained models exist. The files follow SPEC-S2 §5 exactly (f32, f16 or
//! ternary storage).

use sky_gguf::{GgmlType, GgufWriter, OwnedValue};

use super::{S2Config, ARCH, DEFAULT_EXIT_THRESHOLD};
use crate::convert::{write_tokenizer, ConvertOptions};
use crate::model::N_QTYPES;
use crate::ternary;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// Storage of the weight matrices in a generated file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    F16,
    /// Ternary matrices (`X.tq` + `X.scale`) and an int8 token embedding.
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
}

/// Ladder configurations `s2-pico`, `s2-nano`, `s2-micro` (SPEC-S2 §3) and `s3-pico`, `s3-nano`
/// (SPEC-S3 §4: value-fused numbers, looped state blocks).
pub fn ladder_config(name: &str) -> Option<S2Config> {
    // (d, heads, state blocks, loops, static, fusion, ff, vocab, exits, fuse)
    type Row = (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        &'static [usize],
        bool,
    );
    let (d, heads, blocks, loops, stat, fusion, ff, vocab, exits, fuse): Row = match name {
        "s2-pico" => (64, 1, 3, 1, 2, 1, 128, 1024, &[1, 3], false),
        "s2-nano" => (128, 2, 4, 1, 2, 2, 256, 2048, &[1, 2, 4], false),
        "s2-micro" => (256, 4, 6, 1, 3, 2, 512, 4096, &[2, 4, 6], false),
        "s3-pico" => (96, 1, 1, 4, 1, 2, 192, 1024, &[1, 2, 3, 4], true),
        "s3-nano" => (128, 2, 2, 3, 2, 2, 256, 2048, &[2, 4, 6], true),
        _ => return None,
    };
    Some(S2Config {
        name: name.to_string(),
        vocab_size: vocab,
        hidden_size: d,
        intermediate_size: ff,
        num_heads: heads,
        state_layers: blocks * loops,
        state_blocks: blocks,
        state_loops: loops,
        fuse_numbers: fuse,
        temporal_carry: false,
        ffn_es: None,
        static_layers: stat,
        fusion_layers: fusion,
        exits: exits.to_vec(),
        max_state_tokens: 64,
        max_head_tokens: 96,
        num_features: crate::num::NUM_FEATURES,
        num_token_id: 5,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        exit_thresholds: [DEFAULT_EXIT_THRESHOLD; 3],
        temperatures: Vec::new(),
        weight_format: "f32".into(),
    })
}

/// The small byte-level BPE test tokenizer with `[NUM]` at id 5 (for tests and benchmarks).
pub fn test_tokenizer() -> Tokenizer {
    Tokenizer::from_tokenizer_json(include_str!("../../tests/fixtures/small_tokenizer_s2.json"))
        .expect("test tokenizer")
}

/// Write the `skycmd-s2.*` metadata keys of `cfg`.
pub fn write_metadata(w: &mut GgufWriter, cfg: &S2Config, format: &str) {
    let k = |s: &str| format!("{ARCH}.{s}");
    w.add_kv("general.architecture", ARCH);
    w.add_kv("general.name", cfg.name.as_str());
    w.add_kv(k("embedding_length"), cfg.hidden_size as u32);
    w.add_kv(k("feed_forward_length"), cfg.intermediate_size as u32);
    w.add_kv(k("head_count"), cfg.num_heads as u32);
    w.add_kv(k("state_block_count"), cfg.state_blocks as u32);
    // the s3 keys are only written when used, so s2 files stay as they were
    if cfg.state_loops != 1 {
        w.add_kv(k("state_loops"), cfg.state_loops as u32);
    }
    if cfg.fuse_numbers {
        w.add_kv(k("fuse_numbers"), true);
    }
    if cfg.temporal_carry {
        w.add_kv(k("temporal_carry"), true);
    }
    w.add_kv(k("static_block_count"), cfg.static_layers as u32);
    w.add_kv(k("fusion_block_count"), cfg.fusion_layers as u32);
    w.add_kv(
        k("exits"),
        OwnedValue::Array(
            cfg.exits
                .iter()
                .map(|&e| OwnedValue::U32(e as u32))
                .collect(),
        ),
    );
    w.add_kv(k("max_state_tokens"), cfg.max_state_tokens as u32);
    w.add_kv(k("max_head_tokens"), cfg.max_head_tokens as u32);
    w.add_kv(k("num_features"), cfg.num_features as u32);
    w.add_kv(k("num_token_id"), cfg.num_token_id);
    w.add_kv(k("rope_freq_base"), cfg.rope_theta);
    w.add_kv(k("layer_norm_epsilon"), cfg.norm_eps);
    for (t, tau) in ["choice", "score", "noul"].iter().zip(cfg.exit_thresholds) {
        w.add_kv(k(&format!("exit_threshold.{t}")), tau);
    }
    for (key, t) in &cfg.temperatures {
        w.add_kv(k(&format!("temperature.{key}")), *t);
    }
    w.add_kv(k("weight_format"), format);
}

/// Seeded xorshift values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        let s = &mut self.0;
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|_| offset + scale * self.next()).collect()
    }
}

struct Out<'w> {
    w: &'w mut GgufWriter,
    rng: Rng,
    format: WeightFormat,
}

impl Out<'_> {
    fn float(&mut self, name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
        let dims: Vec<u64> = shape.iter().rev().map(|&v| v as u64).collect();
        let ty = if self.format == WeightFormat::F16 && shape.len() == 2 {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        self.w.add_tensor_f32(name, &dims, ty, values)?;
        Ok(())
    }

    fn norm(&mut self, base: &str, d: usize, bias: bool) -> Result<()> {
        let w = self.rng.vec(d, 0.1, 1.0);
        self.float(&format!("{base}.weight"), &[d], &w)?;
        if bias {
            let b = self.rng.vec(d, 0.02, 0.0);
            self.float(&format!("{base}.bias"), &[d], &b)?;
        }
        Ok(())
    }

    /// A matrix `[n_out, n_in]` (ternary when the format says so and `ternary_ok`).
    fn lin(
        &mut self,
        base: &str,
        n_out: usize,
        n_in: usize,
        bias: bool,
        ternary_ok: bool,
    ) -> Result<()> {
        self.lin_gain(base, n_out, n_in, bias, ternary_ok, 1.7)
    }

    /// [`Out::lin`] with weights uniform in `±gain / sqrt(n_in)`.
    fn lin_gain(
        &mut self,
        base: &str,
        n_out: usize,
        n_in: usize,
        bias: bool,
        ternary_ok: bool,
        gain: f32,
    ) -> Result<()> {
        let w = self.rng.vec(n_out * n_in, gain / (n_in as f32).sqrt(), 0.0);
        if self.format == WeightFormat::Ternary && ternary_ok {
            let (t, scale) = ternary::quantize_absmean(&w, n_out, n_in);
            let packed = ternary::pack(&t, n_out, n_in);
            let rb = ternary::row_bytes(n_in) as u64;
            self.w.add_tensor(
                format!("{base}.tq"),
                &[rb, n_out as u64],
                GgmlType::I8,
                packed,
            )?;
            self.w.add_tensor_f32(
                format!("{base}.scale"),
                &[n_out as u64],
                GgmlType::F16,
                &scale,
            )?;
        } else {
            self.float(&format!("{base}.weight"), &[n_out, n_in], &w)?;
        }
        if bias {
            let b = self.rng.vec(n_out, 0.02, 0.0);
            self.float(&format!("{base}.bias"), &[n_out], &b)?;
        }
        Ok(())
    }

    fn block(&mut self, prefix: &str, n: usize, cfg: &S2Config) -> Result<()> {
        let (d, ff) = (cfg.hidden_size, cfg.intermediate_size);
        let p = |s: &str| format!("{prefix}{n}.{s}");
        if n > 0 {
            self.norm(&p("attn_norm"), d, false)?;
        }
        self.lin(&p("attn_qkv"), 3 * d, d, false, true)?;
        self.lin(&p("attn_output"), d, d, false, true)?;
        self.norm(&p("ffn_norm"), d, false)?;
        self.lin(&p("ffn_up"), 2 * ff, d, false, true)?;
        self.lin(&p("ffn_down"), d, ff, false, true)
    }
}

/// A GGUF file with random weights (seeded) for `cfg` and the given tokenizer. With
/// [`WeightFormat::Ternary`] every encoder and fusion matrix plus `cls` is ternary and the token
/// embedding is int8 with per-row scales (SPEC-S2 §5); `num_proj`, `type_embd`, the norms and
/// `cls.output` stay f32.
pub fn random_gguf(
    cfg: &S2Config,
    tok: &Tokenizer,
    seed: u64,
    format: WeightFormat,
) -> Result<Vec<u8>> {
    cfg.validate()?;
    if tok.id_to_token(cfg.num_token_id) != Some(crate::num::NUM_TOKEN) {
        return Err(Error::Format(
            "the tokenizer has no [NUM] token at num_token_id".into(),
        ));
    }
    let mut w = GgufWriter::new();
    write_metadata(&mut w, cfg, format.name());
    write_tokenizer(&mut w, tok, cfg.vocab_size, &ConvertOptions::default())?;
    let mut o = Out {
        w: &mut w,
        rng: Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1),
        format,
    };
    let (v, d) = (cfg.vocab_size, cfg.hidden_size);
    let emb = o.rng.vec(v * d, 1.0, 0.0);
    if format == WeightFormat::Ternary {
        let mut q = Vec::with_capacity(v * d);
        let mut scale = Vec::with_capacity(v);
        for row in emb.chunks_exact(d) {
            let amax = row.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // the scale is stored in f16: quantize with the rounded value
            let s = sky_gguf::f16_to_f32(sky_gguf::f32_to_f16(amax / 127.0));
            scale.push(s);
            q.extend(row.iter().map(|&x| {
                if s > 0.0 {
                    (x / s).round().clamp(-127.0, 127.0) as i8 as u8
                } else {
                    0
                }
            }));
        }
        o.w.add_tensor("token_embd.q8", &[d as u64, v as u64], GgmlType::I8, q)?;
        o.w.add_tensor_f32("token_embd.scale", &[v as u64], GgmlType::F16, &scale)?;
    } else {
        o.float("token_embd.weight", &[v, d], &emb)?;
    }
    o.lin("num_proj", d, cfg.num_features, true, false)?;
    let te = o.rng.vec(N_QTYPES * d, 0.5, 0.0);
    o.float("type_embd.weight", &[N_QTYPES, d], &te)?;
    for n in 0..cfg.static_layers {
        o.block("static.blk.", n, cfg)?;
    }
    o.norm("static_norm", d, false)?;
    for n in 0..cfg.state_blocks {
        o.block("blk.", n, cfg)?;
    }
    for e in 0..cfg.exits.len() {
        o.norm(&format!("exit.{e}.norm"), d, false)?;
    }
    for n in 0..cfg.fusion_layers {
        let p = |s: &str| format!("fusion.{n}.{s}");
        o.norm(&p("q_norm"), d, true)?;
        o.norm(&p("kv_norm"), d, true)?;
        o.lin(&p("attn_q"), d, d, true, true)?;
        o.lin(&p("attn_kv"), 2 * d, d, true, true)?;
        o.lin(&p("attn_output"), d, d, true, true)?;
        o.norm(&p("ffn_norm"), d, true)?;
        o.lin(&p("ffn_up"), 2 * d, d, true, true)?;
        o.lin(&p("ffn_down"), d, 2 * d, true, true)?;
    }
    o.norm("cls.norm", d, true)?;
    o.lin("cls", d, d, true, true)?;
    // a larger scorer gain gives far-from-uniform answers, so that comparisons are meaningful
    o.lin_gain("cls.output", 1, d, true, false, 8.0)?;
    // s3 tensors last: the weights drawn for s2 files do not change
    if cfg.state_loops > 1 {
        let le = o.rng.vec(cfg.state_loops * d, 0.5, 0.0);
        o.float("loop_embd.weight", &[cfg.state_loops, d], &le)?;
        o.norm("blk.0.attn_norm_loop", d, false)?;
    }
    if cfg.temporal_carry {
        // a trained gate starts at 0; random values make the carry visible in tests
        let gate = o.rng.vec(d, 0.5, 0.0);
        o.float("carry_gate.weight", &[d], &gate)?;
        o.norm("carry_norm", d, false)?;
    }
    Ok(w.to_bytes()?)
}
