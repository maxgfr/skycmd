//! Print the metadata and tensors of a GGUF file.
//!
//! Usage: `gguf-inspect <file.gguf> [--max-array N]`

use sky_gguf::{Gguf, Value};

fn fmt_value(v: &Value<'_>, max_array: usize) -> String {
    match v {
        Value::String(s) => fmt_str(s),
        Value::Array(a) => {
            let mut parts: Vec<String> = a
                .iter()
                .take(max_array)
                .map(|e| fmt_value(&e, max_array))
                .collect();
            if a.len() > max_array {
                parts.push(format!("... ({} more)", a.len() - max_array));
            }
            format!(
                "[{}] ({} x {})",
                parts.join(", "),
                a.len(),
                a.elem_type().name()
            )
        }
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
    }
}

fn fmt_str(s: &str) -> String {
    const MAX: usize = 120;
    let mut out: String = s.chars().take(MAX).collect();
    if s.chars().count() > MAX {
        out.push_str("...");
    }
    format!("{out:?}")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut max_array = 8usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-array" => {
                max_array = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(max_array)
            }
            "-h" | "--help" => {
                println!("usage: gguf-inspect <file.gguf> [--max-array N]");
                return;
            }
            _ => path = Some(a),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: gguf-inspect <file.gguf> [--max-array N]");
        std::process::exit(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let g = match Gguf::parse(&bytes) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "{path}: GGUF v{}, {} bytes, alignment {}",
        g.version,
        bytes.len(),
        g.alignment
    );
    println!("\n{} metadata keys:", g.kvs.len());
    for (k, v) in &g.kvs {
        println!(
            "  {k} ({}) = {}",
            v.value_type().name(),
            fmt_value(v, max_array)
        );
    }
    println!(
        "\n{} tensors (ggml dims, ne[0] = innermost):",
        g.tensors.len()
    );
    let mut total = 0u64;
    let mut n_params = 0u64;
    for t in &g.tensors {
        total += t.size;
        n_params += t.n_elements();
        println!(
            "  {:<40} {:<22} {:<6} {:>10} bytes",
            t.name,
            format!("{:?}", t.dims),
            t.dtype.name(),
            t.size
        );
    }
    println!("\ntotal: {n_params} parameters, {total} bytes of tensor data");
}
