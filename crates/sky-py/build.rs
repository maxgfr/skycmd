//! Link the cdylib as a Python extension module (`-undefined dynamic_lookup` on macOS), so plain
//! `cargo build` produces a loadable module, as maturin does.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
