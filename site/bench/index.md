---
title: Bench
aside: false
pageClass: wide-page
outline: false
---

# Bench

Size, accuracy and latency of the `skycmd-s1` ladder. Accuracy, NLL and ECE come from the held-out test split (`runs/<model>/report.md`, written by `sky-train eval`). The native latency comes from `sky-bench` on the build machine. Numbers that are not measured yet show as **TBD**.

<ClientOnly>
  <BenchView />
</ClientOnly>

## Reproduce

```bash
# Convert the trained models and regenerate public/models/manifest.json and public/bench.json
scripts/export-models.sh

# Native latency of one model
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --bin sky-bench -- site/public/models/s1-pico.gguf --iters 300
```

The in-browser button loads each published model and times 300 decisions with the same wasm build as the Play page.
