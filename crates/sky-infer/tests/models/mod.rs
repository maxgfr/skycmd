//! Location of the golden reference model (mgoeckel/oscar-1-17m, Apache-2.0), shared by the
//! golden tests. A missing model is a failure, not a skip: fetch it with
//!
//! ```text
//! cargo run --release -p sky-data -- fetch-model mgoeckel/oscar-1-17m \
//!     --rev a443c18107534cb3148941de601e15a89a2c6da3 --out models-src/oscar
//! ```
//!
//! (or `cargo xtask s5 fetch-models` into the store). It is looked up in `$SKYCMD_OSCAR_DIR`,
//! then `models-src/oscar` of this checkout, then `$SKYCMD_STORE/models-src/oscar` (worktrees
//! share the main checkout's copy).

#![allow(dead_code)]

use std::path::PathBuf;

pub const OSCAR_REV: &str = "a443c18107534cb3148941de601e15a89a2c6da3";

pub fn oscar_dir() -> PathBuf {
    let mut tried = Vec::new();
    let candidates = [
        std::env::var_os("SKYCMD_OSCAR_DIR").map(PathBuf::from),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models-src/oscar")),
        std::env::var_os("SKYCMD_STORE").map(|s| PathBuf::from(s).join("models-src/oscar")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.join("model.safetensors").exists() {
            return c;
        }
        tried.push(c.display().to_string());
    }
    panic!(
        "golden model mgoeckel/oscar-1-17m@{OSCAR_REV} not found (tried {tried:?}); fetch it with \
         `cargo run --release -p sky-data -- fetch-model mgoeckel/oscar-1-17m --rev {OSCAR_REV} \
         --out models-src/oscar`"
    )
}
