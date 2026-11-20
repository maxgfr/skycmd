# Instructions for agents working on skycmd

The experiments ended on 20 November 2026: every line is closed or published, and the repository
stays as the record. The rules below are the ones the project ran under; they still apply to any
correction of a page, a card or a result file.

## Where things are published

- **Everything about skycmd is published on skycmd's own site**: the VitePress site in `site/`,
  deployed to <https://maxgfr.github.io/skycmd/> on every push to `main`. That covers results,
  model pages, demos, tutorials and **articles** (`site/articles/`).
- **The technical report lives at `site/paper/`** (<https://maxgfr.github.io/skycmd/paper/>), with its
  figure written by `scripts/paper-figures.py` and its PDF in `site/public/paper/`. It is updated
  when a result changes, like What we bring and Results; `CITATION.cff` cites it.
- **Never publish skycmd content on the owner's personal site** (the repository
  `maxgfr/maxgfr.github.io`, served at <https://maxgfr.github.io/>). That site is the owner's:
  touch it only when the owner names it in the request.
- Models and their demos also go to Hugging Face (`scripts/hf-publish.py`, `hf/cards/`,
  `hf/spaces.json`); a result is published whether its gate passed or failed.
- **A failed model is published as a result** (owner's decision of 15/11/2026): its file, its
  card and its manifest entry say first that it failed, and its verdict does not change.
- **Every published model has a Space**, failed gates included (owner's decision of 17/11/2026):
  an entry in `hf/spaces.json`, uploaded by `scripts/hf-publish.py`. When another project pins
  files of a model repository by sha256 (the `engine/` folder of `s6-secrets-verify-pico`), publish
  its Space with `--space-only`, which never touches the model repository: any other run that
  would assemble that model repository, a run without `--only` included, stops before uploading
  anything unless `--allow-pinned-engine` is given. The global Space
  `maxgfr/skycmd` (the entry of kind `hub`) gathers every demo and is republished with
  `--only skycmd` after each new model.
- The Hugging Face collection [skycmd](https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603) lists the hub Space, the models by verdict, their
  Spaces and the data sets; it is kept up to date with `scripts/hf-collection.py --apply` after a
  model or a data set is added.

## Publishing a result: checklist

Applied to every model whose gate has been read, passed or failed.

- **Before:** the gate is read once and the result block is committed at the end of the
  pre-registration, with the small result files under `runs/<name>/`.
- **The write-up, in the repository:**
  - its section in `BENCH.md` and its row in the README model table;
  - `site/results/index.md` (its row in the status table, and every pre-registered number,
    failures included), `site/novelty/index.md` and `site/bench/index.md`;
  - its card in the Models catalogue (`site/.vitepress/theme/lib/catalogue.ts`), checked by
    `npm --prefix site test`;
  - its tutorial chapter, where it has one;
  - the GGUF in `site/public/models/` and its entry in `site/public/models/manifest.json`, with
    its sha256 and `gate: passed` or `gate: failed`: **never left at a default when the model
    failed**. A forecaster also gets its line in `site/public/models/forecast.json` and the
    Forecast page;
  - `hf/cards/<name>.md`, opening with the verdict, and its Space entry in `hf/spaces.json`;
  - `DATA_LICENSES.md` for any new data set or teacher;
  - one article in `site/articles/`, listed on `site/articles/index.md`, newest first.
- **Build and dry run:** `scripts/build-wasm.sh`, `(cd site && npm run space:build)`, then
  `python3 scripts/hf-publish.py --dry-run --only <name>`.
- **Upload, once the write-up is merged into `main`:** `python3 scripts/hf-publish.py --only <name>`,
  then the global Space again with `python3 scripts/hf-publish.py --only skycmd`.
- **Done means:** the files committed, the site deployed from `main`, and every URL of the model
  (model repository, its Space, the global Space, the site pages) answering 200.

## Articles

- `site/novelty/index.md` ("What we bring") is the reference: every claim of the project in one
  table, with its measurement, its evidence and where to try it. **Update it with every result.**
  `site/results/index.md` holds the numbers. An article does not replace either and does not
  restate them: it tells the story of one line of experiments and links to both.
- The Models catalogue (`site/models/`, the home page's "What works") is driven by
  `site/.vitepress/theme/lib/catalogue.ts`. **Update it with every result, like What we bring**:
  a new model gets an entry, a verdict moves a card to its group, and every number on a card is
  copied from Results or a model card. `npm --prefix site test` checks it against the manifest.
- One Markdown file per article in `site/articles/`, in English like the rest of the site,
  listed on `site/articles/index.md`, newest first.
- An article is written for a wide technical audience: every term is explained the first time,
  every number says what it was measured on, and the limits of a result are stated as plainly as
  the result. Link to the evidence in the repository (pre-registrations in `docs/`, result files
  in `runs/`).

## Synthetic data and labels

- **Every example, label, paraphrase or teacher output that the project generates for a
  training, validation or test set comes from an open-weight model whose licence allows that
  use**: GLM-5.3-Flash (MIT), the only teacher the project used for that, through `sky-data teacher` (rules in `docs/s6-gen-prereg.md`,
  amendments 1 and 3). Never from a proprietary model, nor from a model whose licence is not a
  permissive open-weight one (check its licence on the Hub before the first call).
- Public datasets are another matter: each is used under its own licence, and `DATA_LICENSES.md`
  says so when an upstream set was itself written by a model.
- Each generated source is listed in `DATA_LICENSES.md` with its model, its licence and its role.
- Text written for a demo or a test fixture (presets, edge cases) is not training data and never
  enters a data set.

## Pre-registered experiments

- Every experiment has a pre-registration in `docs/`, frozen and committed before any
  computation it governs.
- Any change is a **dated amendment, written and committed before the computation it concerns**
  (before the run, the step or the reading it changes). It says what changes and what does not.
- **A test set is read once.** Test seeds and fresh files are opened once, by the step the
  pre-registration names.
- **No rerun "to see".** Only a technical failure justifies a new run, and it is recorded in the
  report and in the result block.
- Never choose on an intermediate checkpoint: the checkpoint rule is fixed in the
  pre-registration.
- A gate is read as written. A failure is reported as such, in the result block, `BENCH.md` and
  on the site, and its file is published as a result (above).
- A system designed after its test was read, and judged on the same records, is disclosed as a
  second look everywhere its result is reported.
- One budgeted GPU run at a time. Budgets count working minutes, so the brake that keeps the
  machine usable does not change them (`docs/idle-training.md`).

## Full test pass

CPU only, one build at a time, since the machine is shared (the pass of 19/11/2026 ran with
`CARGO_BUILD_JOBS=4`). The release build of sky-train compiles MLX; the whole pass takes a few
hours of CPU. Every command passes, or its failure is fixed and committed.

- `cargo fmt --all --check`; clippy as CI runs it:
  `cargo clippy --workspace --exclude sky-train --all-targets -- -D warnings`, then
  `cargo clippy -p sky-train --all-targets -- -D warnings`.
- `cargo test --workspace --exclude sky-train --release`, as CI's `test` job runs it (sky-infer,
  sky-data, sky-web, sky-gguf, sky-games, sky-serve, sky-arena, sky-schema, sky-metrics, sky-py
  and xtask); `cargo test --release -p sky-train` with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- `cargo test -p sky-web --test parity`; `wasm-pack test --node --release crates/sky-web --test parity`
  (with SIMD128, and also `--test iso --test forecast --test verify`, as CI does);
  `wasm-pack test --node crates/sky-web --test memory`; `node crates/sky-web/bench/verify-node.mjs --glue`.
- `python3 -I scripts/test_s6_adopt.py` and the other Python test files of `scripts/` (some need
  numpy or pinned packages through `uv`); `scripts/test-auto-brake.sh` and `scripts/test_brake.py`.
  CI's `scripts` job runs only those that need the standard library: the four that import numpy
  (`scripts/route/test_baselines.py`, `scripts/test_cast2_data.py`, `scripts/test_route_embed.py`
  and `scripts/test_turn_teacher.py`) run only in this pass, for example
  `uv run --python 3.12 --with numpy python -I scripts/test_turn_teacher.py`.
- `bash -n` and shellcheck on `scripts/*.sh`.
- `DRY_RUN=1` on every step of `scripts/train-all.sh`.
- `scripts/build-wasm.sh`, then `npm --prefix site test`, `npm --prefix site run docs:build` with no
  dead link, and `npm --prefix site run space:build`.
- `load_dataset("maxgfr/skycmd-decisions", "boolq")` loads (for example through
  `uv run --no-project --with datasets`).
- The numbers: each model's numbers are the same in `BENCH.md`, the README, Results, What we bring,
  Bench, the Models catalogue, the articles, the tutorial chapters, `hf/cards/` and the manifest,
  and agree with the result files.
