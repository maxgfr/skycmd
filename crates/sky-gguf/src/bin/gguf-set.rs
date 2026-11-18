//! Rewrite a GGUF file with some metadata keys set or replaced; tensors are copied as is. A value
//! is a float, unless it is typed `str:<text>` (a string) or `u32:<n>` (an unsigned integer).
//!
//! ```text
//! gguf-set in.gguf out.gguf key=value [key=value ...]
//! gguf-set model.gguf model.gguf skycmd-s2.exit_threshold.choice=0.95
//! gguf-set model.gguf model.gguf skycmd-s2.route.render=str:scores-in-options
//! gguf-set v2.gguf v2-320.gguf skycmd-s2.max_head_tokens=u32:320
//! ```
use sky_gguf::{Gguf, GgufWriter, OwnedValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: gguf-set <in.gguf> <out.gguf> key=<float|str:text|u32:n> [...]");
        std::process::exit(2);
    }
    let mut updates: Vec<(String, OwnedValue)> = Vec::new();
    for kv in &args[2..] {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got {kv}"))?;
        let value = if let Some(t) = v.strip_prefix("str:") {
            OwnedValue::String(t.to_string())
        } else if let Some(n) = v.strip_prefix("u32:") {
            OwnedValue::U32(n.parse()?)
        } else {
            OwnedValue::F32(v.parse()?)
        };
        updates.push((k.to_string(), value));
    }
    let bytes = std::fs::read(&args[0])?;
    let g = Gguf::parse(&bytes)?;
    let mut w = GgufWriter::new().with_alignment(g.alignment);
    for (k, v) in &g.kvs {
        if k == &"general.alignment" {
            continue;
        }
        match updates.iter().find(|(u, _)| u == k) {
            Some((_, f)) => w.add_kv(*k, f.clone()),
            None => w.add_kv(*k, OwnedValue::from_value(v)),
        };
    }
    for (k, f) in &updates {
        if !g.kvs.iter().any(|(e, _)| e == k) {
            w.add_kv(k.clone(), f.clone());
        }
    }
    for t in &g.tensors {
        w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())?;
    }
    std::fs::write(&args[1], w.to_bytes()?)?;
    eprintln!("{} -> {}: {} keys updated", args[0], args[1], updates.len());
    Ok(())
}
