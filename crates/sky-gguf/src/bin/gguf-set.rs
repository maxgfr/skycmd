//! Rewrite a GGUF file with some float metadata keys set or replaced; tensors are copied as is.
//!
//! ```text
//! gguf-set in.gguf out.gguf key=value [key=value ...]
//! gguf-set model.gguf model.gguf skycmd-s2.exit_threshold.choice=0.95
//! ```
use sky_gguf::{Gguf, GgufWriter, OwnedValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: gguf-set <in.gguf> <out.gguf> key=float [key=float ...]");
        std::process::exit(2);
    }
    let mut updates: Vec<(String, f32)> = Vec::new();
    for kv in &args[2..] {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("expected key=value, got {kv}"))?;
        updates.push((k.to_string(), v.parse()?));
    }
    let bytes = std::fs::read(&args[0])?;
    let g = Gguf::parse(&bytes)?;
    let mut w = GgufWriter::new().with_alignment(g.alignment);
    for (k, v) in &g.kvs {
        if k == &"general.alignment" {
            continue;
        }
        match updates.iter().find(|(u, _)| u == k) {
            Some((_, f)) => w.add_kv(*k, *f),
            None => w.add_kv(*k, OwnedValue::from_value(v)),
        };
    }
    for (k, f) in &updates {
        if !g.kvs.iter().any(|(e, _)| e == k) {
            w.add_kv(k.clone(), *f);
        }
    }
    for t in &g.tensors {
        w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())?;
    }
    std::fs::write(&args[1], w.to_bytes()?)?;
    eprintln!("{} -> {}: {} keys updated", args[0], args[1], updates.len());
    Ok(())
}
