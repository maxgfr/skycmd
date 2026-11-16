# TODO: skycmd, the s6 models, LFM3-1.7B and the flexible weights

Reconciled on 16/11/2026 with the repository: the result files, the pre-registrations and their amendments, `BENCH.md`, the site, `site/public/models/manifest.json`, `hf/cards/`, the Hub and `git log`. How to read the boxes: `- [x] … **Done:** <evidence>` is finished work; `**Superseded:**` was replaced by a later decision, named with its reference; `**Not a task:**` is a rule or a reminder kept as written. Every box still unchecked is a current action and points to an item of "What remains, in order". The sections after that list are history: each opens with one status line, and their outcomes stay as they were written.

## Where we are (16/11/2026)

| model or work item | status | the number that matters (its bar) | next action |
|---|---|---|---|
| s6 recipe (tranche 1: pico pilot and ablation) | done, 12/11 | AdamW at ×2, no addition adopted; the closest, XSA, gained 0.008 (0.01 asked) | none |
| s6-nano | passed, published 14/11 | `typed_decisions_bench` 0.480 (> 0.398); a constant answer per task scores 0.4815 (`docs/s6-nano2-study.md`) | none |
| s6-micro (v2) | passed on a second look, published 16/11 | bench 0.553 (≥ 0.45), ECE 0.0588 (≤ 0.08) | none |
| s6-micro-v1 | failed and published | ECE 0.0821 (≤ 0.08) | none |
| s6-secrets-pico | passed on its synthetic bench, published 14/11; not a scanner | secret 0.9990 (> 0.977); flags 59.8% of 719 clean real files | none |
| s6-secrets-verify-pico (v2) | failed and published; an experimental option of secretgate 1.9.0, off by default | line F1 0.8561 (≥ 0.7332 + 0.15) | none: v3 (rules and verifier together) failed too, 0.8477, and the line ends |
| s6-ear-nano (v1) | failed and published | end of turn 0.5681 (≥ 0.9263, Smart Turn v3.2) | none |
| s6-cast-pico (v1) | failed and published | nGMASE 0.9471 and nWQL 0.6646 (≤ 0.838 and ≤ 0.587) | none |
| s6-route-nano (s6-nano v2) | failed and published, 16/11 | T-tools 0.8395 (bar: above BM25, 0.8500); 38.9 ms in WASM (≤ 2 ms) | none |
| s6-route (s6-nano v3) | passed on fresh slates, published 16/11: Hugging Face model and Space, the Router page, an article | T2-tools 0.8868 against BM25 0.8169, +0.0698 [0.0455, 0.0952] (lower bound > 0) | none scheduled (its limits: "Open, not scheduled") |
| s6-sleep v1 | stopped by its pilot, 15/11 | `recall` 0.2460 against 0.2650 for the control (+0.10 asked) | none |
| s6-sleep v2 | ended at its CPU check C4, 15/11; no GPU run | C4 failed its three allowed attempts | none |
| s6-sleep v3 | in progress: C5 passed, pilot verdict "works" (amendment 2); the first nano run was stopped by the memory guard at 16 GB | pilot `recall` 1.0000 against 0.2542 for the control, +0.7458 [0.7323, 0.7593] (≥ +0.10) | the memory fix, an amendment, the six nano runs (item 2) |
| s6-ear v2 (s6-ear2-nano) | in progress: probe (rung R4), data, harness check and amendment 1 done; smoke, pilot and training launched | G1: accuracy ≥ 0.85 on the 31,527 test clips; the pilot stops v2 below 0.82 on validation | finish the run, read the gate (item 1) |
| s6-cast v2 (s6-cast2-pico) | code and data ready, amendments 1 and 2; not launched | G1: nGMASE ≤ 0.838 and nWQL ≤ 0.587; G2: MASE ratio < 1.00 at each term | a launch amendment, then the run (item 3) |
| LFM3-1.7B S1/S2 router (§6) | failed, 15/11 | router 1,032 tokens against 958 for System 2 alone (H3 asked ≤ 0.5 × 958) | none |
| LFM3-1.7B fine-tuned (§6.1) | not started | G1: System 1 accuracy ≥ base + 0.03 | the local smoke (item 5) |
| §7 flexible weights | not started; `docs/weights-prereg.md` frozen 09/11, no amendment, no code | (its verdicts are per experiment) | an amendment before any computation (item 6) |
| §10.4 sleep and dreaming; §10.5 in-place TTT | not started | | item 7; §10.5 not scheduled |
| Hugging Face datasets (14, public) | the viewer is valid on all 14 (checked 16/11) | | the `load_dataset` check (item 8) |
| full test pass, consistency of the numbers | not started | | item 8 |
| clean-up | not started | 19 worktrees, 42 local branches, `data/ear` 31 GB | item 9 |

## What remains, in order

Times: "working minutes" are GPU minutes as the pre-registrations count them. Wall clock depends on the brake (`scripts/auto-brake.sh`): about ×1.15 in the night mode (0.95, 55:5), ×1.4 when the Mac is idle (0.85, 50:10) and ×2.7 when it is in use or on battery (0.5, 45:15). One budgeted GPU run at a time.

1. **s6-ear v2 (s6-ear2-nano): finish the run in progress** (`docs/s6-ear2-prereg.md`, amendment 1).
   - **Precondition:** the probe, the data, the harness check and the smoke are on file (`runs/s6-ear2-probe/rung.json`, `runs/s6-ear2-harness/check.json`); nothing else trains on the GPU; v1's `data/ear` stays reachable through its link.
   - **Commands**, from the run's worktree: `caffeinate -i scripts/train-turn.sh pilot train export calibrate` (running; after a stop, the same command resumes), then with no training running: `scripts/train-turn.sh latency wasm eval eotbench gate`.
   - **GPU:** about 293 working minutes in all (5 smoke, 48 pilot, 240 training; §6), part of it already done. Wall clock for the 293: about 5.5 h in the night mode to 13 h throttled (the pre-registration says 7 to 9 h).
   - **Done means:** `runs/s6-ear2-nano/GATE.md` and `gate.json` written, or the pilot's stop rule recorded (best arm below 0.82: failed by rule), and the result block committed in `docs/s6-ear2-prereg.md`.
2. **s6-sleep v3: the six nano runs, after the memory fix** (`docs/s6-sleep3-prereg.md`).
   - **Precondition:** the memory fix committed, with a short run showing that it keeps the computation (the same losses as before the fix); a dated amendment 3 that records the stop by the memory guard as a technical failure, the fix and the relaunch, before any nano run starts again; item 1 off the GPU.
   - **Commands:** `caffeinate -i scripts/train-sleep3.sh nano` (finished runs are skipped), then `scripts/train-sleep3.sh export eval gate` (`eval` reads the test seeds once).
   - **GPU:** six runs of 30 working minutes, 180 in all (§6). Wall clock: about 3.5 h in the night mode to 8 h throttled, plus the CPU export, test reading and arena gate.
   - **Done means:** `runs/s6-sleep3/n.json`, the report and the gate written, and the result block committed in `docs/s6-sleep3-prereg.md`.
3. **s6-cast v2 (s6-cast2-pico)** (`docs/s6-cast2-prereg.md`, amendments 1 and 2).
   - **Precondition:** a dated launch amendment (amendment 3) before any training: the counts of the leakage check run again with amendment 2's rule (`skycmd-data/s6-cast2/leak.json`, written 15/11 at 11:55), the worktree, the commit of the trainer and the brake of the run; items 1 and 2 off the GPU.
   - **Commands:** `SMOKE=1 scripts/train-cast2.sh` (a throwaway prefix; its numbers are not looked at), then `caffeinate -i scripts/train-cast2.sh pilot train export`, then with no training running `scripts/train-cast2.sh latency eval chronos gate` (G3 is dropped by amendment 2; Chronos-ZS is reported).
   - **GPU:** about 760 working minutes (12 for the pilot, 720 of training, then export, calibration and evaluation; §5). Wall clock: about 14.5 h in the night mode to 34 h throttled (the pre-registration says about 20 h).
   - **Done means:** `GATE.md` and `gate.json` written and the result block committed in `docs/s6-cast2-prereg.md`.
4. **Write-up and publication of items 1 to 3, passed or failed** (a failed model is published as a result: owner's decision of 15/11/2026; `AGENTS.md`).
   - **Precondition:** the item's gate is read.
   - **For each model:** its section in `BENCH.md`; the README model table; `site/results/index.md`, `site/novelty/index.md` ("What we bring", updated with every result) and `site/bench/index.md`; the GGUF in `site/public/models/` and its `manifest.json` entry (`gate: passed` or `failed`, never a default when failed); `hf/cards/<name>.md` and its `hf/spaces.json` entry; `scripts/build-wasm.sh`, `(cd site && npm run space:build)` and `python3 scripts/hf-publish.py --dry-run --only <name>`; then the upload, `python3 scripts/hf-publish.py --only <name>`, by the owner; an article in `site/articles/` if the line of experiments has a story to tell.
   - **s6-ear v2:** if it is published with a browser demo, the microphone check by the owner in Chrome, then Safari: the permission asked once, the decision following the speech, the latency, a clean console (moved here from §8.2).
   - **s6-sleep v3:** if it passes, the browser `GameSession` calls the memory's sleep at level changes (moved here from §10.1; the engine part exists).
   - **GPU:** none. **Done means:** the files committed, the site deployed from `main`, and each Hugging Face URL answering 200.
5. **LFM3-1.7B fine-tuned (§6.1, `docs/lfm3-ft-prereg.md`).**
   - **Precondition:** items 1 to 3 off the GPU. Amendment 1 (09/11/2026) already moves everything to the M5 and fixes the LoRA rule; what is missing before `gen` is the local smoke and the dated amendment that copies its revision and LoRA method into the pre-registration and into `BASE_REV_PINNED` and `ENGINE_PINNED` of `scripts/lfm3-ft.sh`. First, `DRY_RUN=1 scripts/lfm3-ft.sh smoke gen sft eval route gate` (the `eval` dry run was never made) and `python3 experiments/lfm3-ft/test_common.py`.
   - **Commands:** `caffeinate -i scripts/lfm3-ft.sh smoke`, then read `runs/lfm3-ft/smoke/report.json`; the amendment; `caffeinate -i scripts/lfm3-ft.sh gen sft eval`; `scripts/lfm3-ft.sh route gate`.
   - **GPU:** about 50 h or more (the generation alone is 10 to 15 h at full speed). Wall clock: about 57 h in the night mode to 135 h throttled, over several weeks of evenings and nights.
   - **Owner:** the derived weights stay private until the owner explicitly agrees to publish them (§9 of the pre-registration).
   - **Done means:** `runs/lfm3-ft/report.md` and `gate.json` (G1 to G3, and the router H1 to H4 on both models), the result block committed, `BENCH.md` and the Results page updated.
6. **§7 flexible weights (`docs/weights-prereg.md`): an amendment first.**
   - **Precondition:** before any computation, a dated amendment (or a new pre-registration, the owner's choice) that brings the frozen text of 09/11 up to date: the base is the checkpoint behind the published s6-nano, `runs/s6-nano-g3/decide/last`; s6-nano plays at the oracle's score in every game on its gate's 30 seeds, so the target-game rule of §3 may find no gap on the dev seeds, and the amendment says what happens then; the z.ai quota of the GLM-5.3-flash calls. Then the code (none is written). Order 7.1, 7.2, 7.4, 7.3 (§2 of the pre-registration).
   - **GPU:** 7.1 and 7.3 run in the arena on the CPU; 7.2 trains nine decision runs of 6,400 steps and 7.4 four copies of s6-secrets-pico at its 23-minute budget: roughly 5 to 6 h of GPU in all (estimated from the step rates of `BENCH.md`, to be fixed in the amendment), 7 to 16 h of wall clock.
   - **Done means:** `runs/weights/REPORT.md` with every arm, failures included; `BENCH.md` and the site updated.
7. **§10.4, the flexible weights with sleep and dreaming: only if justified.**
   - **What would justify it:** s6-sleep v3 passes its gate with H1 (sleep passes help at equal wake latency), and §7.3's adapter with its guardrail passes its verdict (`docs/weights-prereg.md` §7). If either fails, §10.4 is not run and this item closes as "not justified".
   - **Then:** a dated pre-registration before any code or computation; the GPU cost is unknown until it is written. §10.5 (in-place TTT on LFM3) stays not scheduled: only after §6.1, and only if §6 shows long-context failures.
8. **The full test pass and the final consistency check** (CPU only).
   - **Commands:** `cargo test` on sky-infer, sky-data, sky-web and sky-gguf; `cargo test --release -p sky-train` (with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`); `cargo test -p xtask`; clippy and fmt; `cargo test -p sky-web --test parity`, `wasm-pack test --node --release crates/sky-web --test parity` and `wasm-pack test --node crates/sky-web --test memory`; `python3 -I scripts/test_s6_adopt.py` and `python3 experiments/lfm3-ft/test_common.py`; `bash -n` and shellcheck on `scripts/`; `DRY_RUN=1` on every step of `scripts/train-all.sh` and `scripts/lfm3-ft.sh`; `cd site && npm run docs:build` with no dead link; each Space opened in a real browser (the model loads, the demo answers, the console is clean); `load_dataset("maxgfr/skycmd-decisions", "boolq")`.
   - **Numbers:** each model's numbers the same in `BENCH.md`, the README, `site/results`, `site/novelty`, `site/bench`, the tutorial chapters, `hf/cards/` and the manifest, starting with the contradictions found by the reconciliation of 16/11/2026.
   - **GPU:** none; a few hours of CPU (the release build of sky-train compiles MLX). **Done means:** every command passes, or its failure is fixed and committed, and no number disagrees.
9. **Clean-up, with the owner's agreement for anything deleted.**
   - **Worktrees and branches:** 19 worktrees and 42 local branches on 16/11 (`git worktree list`, `git branch`). Remove those whose branch is merged into `origin/main` and that hold no uncommitted work (`git worktree remove`, `git branch -d`); the main checkout is on a local `main` behind `origin/main` (`git pull --ff-only` there).
   - **Data:** `data/ear` (31 GB, v1's audio) is still read by s6-ear v2: only after item 1; `runs/s6-ear-nano-swap/` and `runs/s6-nano-lr-swap/` (runs set aside, 88 MB); the tokenized caches (`find runs -type d -name cache -prune`, §0); `target/s3-train/debug`; the throwaway smoke prefixes (`runs/turn-smoke/`, `runs/smoke-sleep3/`).
   - **GPU:** none. **Done means:** `git worktree list` shows only live work, and nothing was deleted without the owner's yes.

**Licences, decided by the owner on 16/11/2026** (after a read-only audit by GLM-5.3-Flash, checked on the Hub):

- [x] Every data set stays published, `data/dolma3_science_pdfs.parquet` of `maxgfr/skycmd-text` included. **Done:** the owner's decision; nothing is removed.
- [x] The files `glm_orders` and `paraphrased` (s1 to s3) were written by GLM-5.3-Flash (open weights, MIT). **Done:** the owner's statement, recorded in `DATA_LICENSES.md` with the fact that the name was not logged with the files; no retraining.
- [x] Generated data and labels are free to use and to publish, marked as AI-generated. **Done:** `DATA_LICENSES.md`, `AGENTS.md`.
- [ ] Publish the training records of s6-route-nano as a data set (`skycmd-route`: the slates built from the public tool-calling sets under their own licences, and the teacher's descriptions), which `scripts/hf-publish-data.py` does not cover yet. The real-credential sets of s6-secrets are never published: only their obfuscated or replaced forms could be, and they are not ours to redistribute.

**Open, not scheduled** (no pre-registration, no date): the three limits of s6-route in the notes of 15/11 below (other tool families, names masked, the weight of the three largest clusters), each a new pre-registration if wanted; §10.5; French intent data and the truncation of states (notes of 14/11).

## History: the launch of tranche 1 (written 06/11 to 11/11/2026)

> **Status (16/11/2026): done.** Tranche 1 ended on 12/11/2026 at 04:43 and wrote `runs/recipe/recipe.json` (committed in `f8a052b`); the instructions below are kept as written.

> **Le mode bridé (décidé le 09/11/2026 : le Mac chauffe trop, et il ne reste pas ouvert en permanence).** `SKY_DUTY=0.5` fait travailler le GPU la moitié du temps : deux fois moins de puissance moyenne, pour deux fois plus de temps. Le nombre de pas, les graines et les résultats ne changent pas, et les pauses ne comptent ni dans `train_minutes` ni dans les budgets en minutes (`docs/s6-prereg.md`, amendement 5).
>
> **Des repos complets, en plus (`SKY_COOL=45:15`).** Chaque heure, 45 min de travail bridé puis 15 min sans aucun calcul GPU, pour que la machine refroidisse vraiment. Ces repos non plus ne comptent ni dans `train_minutes` ni dans les budgets.
>
> **Cette tranche n'entraîne pas les s6.** Elle choisit leur recette, c'est-à-dire quelles optimisations garder, et ne produit aucun modèle final. Les 5 modèles s6 demandent environ 48 h de GPU, soit environ 128 h avec la bride et les repos. Les durées ci-dessous comptent le temps où le Mac est allumé et ouvert, pas le temps calendaire :
>
> | tranche | ce qui tourne | à pleine vitesse | bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`) | prête ? |
> |---|---|---|---|---|
> | 1 | pilote + ablation pico → recette (`scripts/s6-recipe.sh`) | ~17 h 30 | **~47 h** | **oui** |
> | 2 | s6-ear, s6-secrets puis s6-nano | ~11 h | ~29 h | oui, après la fumée `SMOKE=1` (§9.2) |
> | 3 | s6-micro | ~11 h | ~29 h | oui, après la fumée |
> | 4 | s6-cast | ~9 h | ~24 h | oui, après la fumée |
>
> Then `s6-sleep-nano` (§10, added on 11/11/2026: code, probes and a dated pre-registration first, then about 20 to 25 h throttled with the fast path of §10.1), before LFM3.
>
> Puis LFM3 S1/S2 et LFM3 fine-tuné (§9.4, tranches 5 et suivantes). Le 09/11/2026, le §9.2 a été écrit pendant la tranche 1 : les amendements, `scripts/train-all.sh` et le mode local de `scripts/lfm3-ft.sh`.

Tout est prêt pour cette tranche : le plantage au pas 74 du pré-entraînement est corrigé (`non-finite loss`, un nombre du texte trop grand pour un `f32` ; `docs/s6-prereg.md`, amendement 5). Les labels GLM sont finis à 100 %, les amendements 4 et 5 de `docs/s6-prereg.md` sont écrits, et le dry-run passe. Cette tranche ne lit pas encore `data/distill/glm.jsonl`, qui sert à la tranche 2, mais le mélange est déjà prêt. Dans un terminal, **de préférence sur secteur** :

```
cd ~/Documents/Code/skycmd && git pull
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # Metal (absent du profil du shell)
caffeinate -i scripts/s6-recipe.sh 2>&1 | tee -a runs/s6-recipe.log
```

- **Ce que ça fait, sans aucune décision à la main :**
  - le pilote du taux d'apprentissage (6 runs, environ 7 h bridé) ;
  - l'extension, seulement si un choix tombe au bord de la grille ;
  - l'ablation pico aux échelles du pilote (11 bras × 3 graines, environ 40 h bridé) ;
  - puis `scripts/s6-adopt.py adopt`, qui écrit la recette dans `runs/recipe/recipe.json`.

  Au total, environ **47 h de Mac ouvert**, à étaler sur plusieurs jours.
- **Régler la chauffe**, avec deux variables à exporter avant la commande. **Garder un seul mode pour toute la tranche 1** : changer de mode en cours de route ne bouge pas la précision, mais peut fausser `train_minutes`, sur lequel reposent deux règles d'adoption (Muon +15 %, sans polish 2 % plus rapide).
  - `SKY_DUTY` est la part du temps où le GPU travaille : `0.3` chauffe moins et prend environ 78 h, `0.7` chauffe plus et prend environ 33 h, `1` est la pleine vitesse ;
  - `SKY_COOL` est le cycle en minutes, travail puis repos : `40:20` refroidit plus longtemps (environ 53 h), et `SKY_COOL=` (vide) supprime les repos. Le journal affiche `cooling down for … s` à chaque repos ; le run n'est pas bloqué.
- **Fermer le Mac quand on veut :**
  - **rabattre l'écran** met le run en pause, et il repart seul à l'ouverture : aucune commande à retaper ;
  - **éteindre ou redémarrer** arrête le run : relancer la même commande. Les étapes finies sont sautées, seule l'étape en cours repart de son début (40 min au plus à perdre en mode bridé) ;
  - une erreur, par exemple du GPU au réveil, relance toute seule le script après 60 s. Il abandonne après 5 échecs de suite sans étape finie, et le journal le dit ;
  - **ne pas utiliser `pmset disablesleep`** dans ce mode : le Mac tournerait écran rabattu, le pire cas pour la chaleur.
- **Pour suivre :**
  - `tail -f runs/s6-recipe.log` ;
  - les journaux de chaque étape dans `runs/s5-logs/` ;
  - `ls runs/s6lr runs/s6ab`.
- **Avant, si on veut voir les commandes sans rien lancer :** `DRY_RUN=1 scripts/s6-recipe.sh`.
- **Après :** ouvrir une session Claude Code pour :
  - lire `runs/recipe/recipe.json` et commiter la recette ;
  - écrire ce qui reste au §9.2 avant la nuit 2 : les amendements s6-ear, s6-cast, s6-gen, s6-secrets et LFM3, `train-all.sh` et le mode local de `lfm3-ft.sh`.
- **À ne pas faire en même temps :** LFM3 S1/S2 (§6) et s6-ear (§8). Ils utilisent le même GPU.

## History: status notes, 05/11 to 16/11/2026 (newest first)

> **Status (16/11/2026): history.** "Where we are" at the top replaces these notes; the points they leave open are in "What remains, in order".

- **15/11/2026: two more failures, read by their rules, and what remains.**
  - **s6-cast-pico failed** both GIFT-Eval bars (nGMASE 0.9471, nWQL 0.6646 against 0.838 and 0.587) (`docs/big-run-report.md`).
  - **s6-secrets v2 (s6-secrets-verify-pico) failed on G1** (`docs/s6-secrets2-prereg.md`, result block): line F1 0.8561 on the 221 CredData test repositories against 0.7332 for the best rule scanner, +0.123 where +0.15 was asked. The six other gates hold (0.8114 against 0.5367 without the largest repository, above cbx-17M − 0.10 and the size-matched logistic regression, issue reports 0.8162, ECE 0.0073, 1,606 µs per candidate, 0.35% of clean real files). Not published, not rerun, not continued in this form, not in secretgate. The hand reading of the 110 accepted clean-text candidates is done: none looks like a real credential (`BENCH.md`). (Later the same day the file was published as a result, by the owner's decision of 15/11; secretgate 1.9.0 offers it as an experimental option, off by default, since 16/11: `BENCH.md`.)
  - **s6-secrets v3 failed on G1 again** (`docs/s6-secrets3-prereg.md`, result block): not a new model but secretgate's default rules and the v2 verifier together (a line is flagged when either flags it, tau3 0.34933 committed in `370f60f` before the test read), judged on the seven gates of v2, a second look at test files already read once, with the system defined after the first reading. Union 0.8477 against 0.7332 + 0.15 on every repository (the model alone was 0.8561; union minus model −0.0084 [−0.0181, 0.0072]); the six other gates hold. On validation all seven held; the 221 test repositories did not confirm it. No third reading: the work on this model ends (`runs/s6-secrets3/eval-test/results.md`, `BENCH.md`).
  - **s6-sleep stopped by its pilot** (`docs/s6-sleep-prereg.md` §5): `recall` 0.2460 for the memory at N = 1 against 0.2650 for the control without memory, +0.10 asked; no nano run (`runs/s6-sleep/pilot.json`, `BENCH.md`).
  - **What the study of s6-nano found after its gate** (`docs/s6-nano2-study.md`; from the evaluation dumps and from validation splits). On 17 of the 20 tasks of `typed_decisions_bench` the model gives one answer to at least 95% of the items; its 0.4805 equals answering each question with its most common training answer (0.4815). The bench's tasks were 0.4% of what it drew in decision training. A linear model of at most 715,136 bytes reaches 0.703 on the validation split of the bench's tasks, and 0.947 / 0.944 on MASSIVE English / French validation once trained on MASSIVE (a validation rehearsal, not a test). French training data: two yes/no tasks. When2Call: `tool_call` is 13.9% of training and 35% of the test, and the model predicts it 39 times in 2,000. NOTAM: no training data. So the gate of s6-nano (above the global majority class, 0.398) is passed by a constant answer per task; the verdict stands, as the gate was passed as written. Done: the notice on `hf/cards/s6-nano.md` (not uploaded), the Live text, the "At a glance" row, Results, Bench, BENCH, README, chapter 27. A narrower second version (a router, option B of the study) was studied and is **not scheduled**: the study's own estimate of clearing all its gates is about 10 to 20%.
  - **secretgate 1.8.1 released on 14/11** with the fixes found while measuring v1: https://github.com/maxgfr/secretgate/releases/tag/v1.8.1
  - **LFM3 S1/S2 ended on 15/11 at 12:44 with the verdict "failure"** (§6; `runs/lfm3-s1s2/report.md`, `BENCH.md`). Test, 1,000 questions, budget 1,024 chosen on calibration: System 1 alone 0.667 at 559 tokens, System 2 alone 0.796 at 958, the router 0.786 at 1,032, the oracle 0.814. H2 (accuracy kept) yes, H3 (compute halved) no, H4 (better than both modes) no. `docs/lfm3-ft-prereg.md` sets no condition on this result (§6.1).
  - **The four failed models are published as results** (owner's decision of 15/11/2026; the last amendment of each pre-registration): s6-micro-v1, s6-ear-nano, s6-cast-pico and s6-secrets-verify-pico, on the site (manifest `gate: failed`, never a default) and prepared for Hugging Face (cards opening with the failure; Spaces for the first three, card only for the last; `scripts/hf-publish.py`, upload by the owner). Verdicts unchanged; nothing enters secretgate. (16/11/2026: all four are on Hugging Face, and secretgate 1.9.0 offers s6-secrets-verify-pico as an experimental option, off by default: `BENCH.md`.)
  - **s6-micro v2: passed its gate on 16/11, on a second look** (amendments 12, 16 and 17 of `docs/s6-gen-prereg.md`; `runs/s6-micro2/runs/s6-micro/eval/results.md`). Bench 0.553 (gate 0.45, goal 0.50 met), ECE 0.0588 (bootstrap [0.0526, 0.0687]), the games at s3-pico's scores; CRC thresholds rejected (one extra Dodge veto), so full depth. v1 failed the ECE guard and v2 was designed after that reading on the same test records: every place that reports it says so. Deviations: one restart from `decide/last` (step 61,051) after a file-access error, the end of the run on a loaded machine (latencies not comparable), a binary that is not v1's. When2Call got worse (0.401, ECE 0.2165; `tool_call` predicted 45 times in 2,000); on 11 of the 20 bench tasks one answer covers at least 95% of the items.
    - [x] Result block (amendment 17), result files under `runs/s6-micro2/`, the two readings of amendment 16 (`scripts/s6-answer-priors.py`).
    - [x] Prepared for publication as `s6-micro`: `site/public/models/s6-micro.gguf` (sha256 `f66d7021…40ff`), manifest (`gate: passed`, with the second-look note), card `hf/cards/s6-micro.md`, a Space entry in `hf/spaces.json`; `s6-micro-v1` stays published as the failed first version.
    - [x] Uploaded on 16/11: https://huggingface.co/maxgfr/s6-micro and its Space, with the updated cards of s6-micro-v1, s6-nano, s6-ear-nano (the corrected Smart Turn bar) and s6-secrets-verify-pico; the site deploys from `main`.
    - [x] The English MASSIVE score of v2 on the 1,794 test utterances not seen in training (amendment 14): 0.530 (0.539 on the 206 seen ones); pooled ECE without the 206: 0.0594.
  - **s6-cast v2 (s6-cast2-pico): pre-registered, not run yet** (`docs/s6-cast2-prereg.md` with amendments 1 and 2: computed periodicity, a seasonal draft, USCRN sub-hourly series; gates G1 and G2 on GIFT-Eval, G3 dropped and Chronos-ZS reported). No result yet.
  - **s6-nano v2 (s6-route-nano): failed its gate on 16/11** (`docs/s6-nano2-prereg.md`, result block; `runs/s6-nano2/eval-test/results.md`, `BENCH.md`). The pilot rule chose micro (0.6390 against 0.5778 for nano, no abort); one run of 120 working minutes, 20,504 steps; τ_ask 0.72938. T-tools 0.8395: above potion-cos + 0.03 (0.6975), static-ft (0.6735) and pair-lr (0.4500), but not above BM25 (0.8500; −0.0105 [−0.0265, +0.0055]), the condition of amendment 1: G1 fails. G4 fails: 38.9 ms per request in WASM against 2 ms. G2 and G3 hold (ECE 0.0244). With every tool renamed it falls to chance (0.1050). Announced goals: T-tools ≥ 0.80 met; MetaTool, CLINC and the 0.95 at τ_ask not met. Not published as a router, no playground, not repeated. Deviations: `TEST-OPEN` created by the run's operator under the owner's delegation, the evaluation binaries and the WASM package rebuilt from `8b0ece3` after training.
    - [x] Result block, result files under `runs/s6-nano2/` and `runs/s6-nano2-pilot/`, `scripts/s6-route-explore.py` (the exploratory fusions, labelled as such).
    - [x] Prepared for publication as a result, not uploaded yet: `site/public/models/s6-route-nano.gguf` (sha256 `a124dfe8…6c82`), manifest (`gate: failed`, `task: route`), card `hf/cards/s6-route-nano.md`, a model-only entry in `hf/spaces.json` (`hf-publish.py --only s6-route-nano`); BENCH, README, SPEC-S6 and the site pages. (Uploaded on 16/11/2026: https://huggingface.co/maxgfr/s6-route-nano.)
    - [x] The fusion read in the exploratory block (model + potion-cos-4bit: 0.8885 on the already-read T-tools, nothing fitted; a hypothesis, not a result) is pre-registered as s6-nano v3, `docs/s6-nano3-prereg.md` (the v2 file fused with the 1 MB static embedding, to be read on fresh slates): read once on 16/11, gate passed (the item below).
  - **s6-nano v3 (s6-route): passed its gate on 16/11, on fresh slates; the system was defined after v2's test was read** (`docs/s6-nano3-prereg.md`, result block; `runs/s6-nano3/eval-fresh/results.md`, `BENCH.md`). No training: v2's file fused with `potion-cos-4bit`, β 0.81, τ_ask3 0.671047, 3,692,224 bytes in two files. T2-tools (945 fresh requests, 273 clusters) 0.8868, BM25 0.8169 (+0.0698 [0.0455, 0.0952]), MiniLM 0.9048; G2, G3 (ECE 0.0162) and G4 (30.9 ms in WASM, bar 60 ms) hold. Reported with it: without the three largest clusters +0.0202 [−0.0062, +0.0481] over BM25, cluster bootstrap +0.0698 [−0.0079, +0.1404]; names masked 0.2497; MiniLM above on every file; the 0.95 at τ_ask3 goal not met (0.9433 at coverage 0.8593).
    - [x] Result block, result files under `runs/s6-nano3/`, `scripts/route3-read.sh`.
    - [x] Prepared for publication, not uploaded yet: `site/public/models/s6-route-embed.gguf` (sha256 `c53fb76d…a67a`, MIT, from potion-base-2M) next to the unchanged `s6-route-nano.gguf`, manifest (`task: router`, two `files`, `gate: passed` with the disclosure), card `hf/cards/s6-route.md`, the Router page of the site and a Space of kind `router` (`hf-publish.py --only s6-route`); BENCH, README, SPEC-S6 and the site pages. (Uploaded on 16/11/2026: the box below.)
    - [x] Upload to Hugging Face (owner): the model repo `s6-route` (both files and the card) and its Space; the card of s6-route-nano with its pointer to s6-route. **Done:** checked on the Hub on 16/11/2026: https://huggingface.co/maxgfr/s6-route (both GGUF files and the card) and the Space `maxgfr/s6-route` answer 200, and the s6-route-nano card there points to s6-route.
    - [ ] **Not scheduled; next action if wanted: a dated pre-registration.** A confirmation on tool lists from sources the clusters never saw (not from v2's test side): T2-tools reuses v2's held-out test tools, whose results were already read; outside its training families the system stays near its embedding part (`BENCH.md`, s6-route, "Open"). See "Open, not scheduled" at the top.
    - [ ] **Not scheduled; any fix is a new pre-registration.** The names-masked weakness: 0.2497 with every tool renamed `tool_<k>`, where the embedding alone keeps 0.7280; the model part is confidently wrong there (accuracy 0.0720 at a mean confidence of 0.7968) and pulls the fusion with it. See "Open, not scheduled" at the top.
    - [ ] **Not scheduled; next action if wanted: a dated pre-registration of a test with a flatter cluster distribution.** The advantage over BM25 leans on the three largest clusters of T2-tools (without them +0.0202 [−0.0062, +0.0481]; the cluster bootstrap gives +0.0698 [−0.0079, +0.1404]). See "Open, not scheduled" at the top.
  - **What remains:** the runs of s6-ear v2, s6-sleep v3 and s6-cast v2, LFM3 fine-tuned (§6.1), §7, the full test pass. (16/11/2026: see "What remains, in order" at the top.)
- **After the publications of 14/11/2026 (same day).**
  - **s6-secrets-pico fails on real text**: measured for a possible use in `secretgate`, it flags 59.8% of 719 clean real files at p >= 0.9 (`docs/s6-secrets-prereg.md`, amendment 6; secretgate pull request 19). The gate on its generated bench stands; the card, the Space, the site and `BENCH.md` say "not a scanner". Nothing entered secretgate.
  - **Next: s6-secrets v2, pre-registered** (`docs/s6-secrets2-prereg.md`, frozen before any download): a rule finds the candidate and the model judges it in context; real repositories (CredData, issue reports) for training and test; seven gates, one of them on clean real text (at most 1.0% of files flagged). Downloads approved by the owner on 14/11. To do: data (running), the candidate export in secretgate, the generator freeze, the pipeline (`scripts/train-secrets2.sh`), the baselines, then one run when the GPU is free. **Done:** trained once on 14/11 and failed on G1 on 15/11 (the 15/11 bullet above, §3, `BENCH.md`).
  - **Research on where s6 is useful**: `docs/research-s6-usefulness.md`. In short: verifiers behind a high-recall rule, not a general decision model; s6-ear should read log-mel and prosody and drop the Mimi-code pretraining (v1 already read log-mel; its pretraining on codec tokens is what hurt, see the correction at the top of that file); s6-cast has to be measured against TinyCast on GIFT-Eval; every future gate should include a size-matched simple baseline.
  - **Browser engine**: two builds are shipped (SIMD128 and relaxed SIMD), the loader takes the relaxed one where the browser validates it: 1.16x to 1.30x faster under Node, same answers (`BENCH.md`). Both s6 Spaces republished with it.
  - **Site**: tutorial chapters 27 (s6-nano) and 28 (s6-secrets-pico), an "At a glance" table on "What we bring", the methods credited on Sources, `docs/SPEC-S6.md` updated.
  - **Still open from this list**: French intent data (the training set has French only in PAWS-X and Allociné, no intent task), state truncation, the s6-ear redesign, the comparison of s6-cast with TinyCast. (16/11/2026: the s6-ear redesign is s6-ear v2, item 1; the comparison of s6-cast with TinyCast is in `BENCH.md`, and its v2 is item 3; French intent data and state truncation are not scheduled.)
- **Gates read on 14/11/2026 (tranches 2 to 4, s6-cast still training).**
  - **s6-nano: passed, published on 14/11** at full depth (site, Hugging Face model and Space `maxgfr/s6-nano`): 0.480 on `typed_decisions_bench` (gate > 0.398), ECE 0.0479. The published file stores its exit thresholds at 1 (`docs/s6-gen-prereg.md`, amendment 13). A constant answer per task passes that gate too (the 15/11 study, above).
  - **s6-secrets-pico: passed** (secret 0.9990 against 0.977, service 0.9795 against 0.787; line 0.686 is weak). **Published on 14/11** at full depth (site, Hugging Face model and Space `maxgfr/s6-secrets-pico`), with its own demo; native median 786 to 789 µs per decision, WASM 1,420 µs, parity 1.2e-6. The demo stores no key-like value: the Stripe-format bodies of its presets are drawn in the browser (GitHub push protection refused the bench's own fake keys).
  - **s6-micro: failed on the ECE guard only** (0.0821 against 0.08; bench 0.547, goal 0.50 met). Not published (its file was published later as `s6-micro-v1`, a result; its second version passed on 16/11, above). Amendment 12 says where the ECE comes from (a zero-shot NOTAM question answered on the wrong side of 0.5) and pre-registers **s6-micro v2** (360 minutes of decision training instead of 180, same pretraining, same gate), to run after s6-cast, s6-sleep and LFM3 S1/S2.
  - **s6-ear-nano: failed** three of its four gates (end of turn 0.568 against 0.926, Pipecat's report for Smart Turn v3.2, where the v1 harness read 0.852; keywords 0.8885 against 0.90; zero-shot French 0.133 against 0.20); the control without Mimi pretraining is better on all four. Reported on the Results page; a new version needs a new design.
  - **Found on the way:** English MASSIVE is not zero-shot (amendment 14: 3,113 training questions come from it, 206 test utterances are seen; the gates do not change); the decision training runs uncompiled since 14/11 (amendment 11).
  - **Still missing from the pre-registered reports**, to compute for `runs/all/REPORT.md`: GLM call and token counts, agreement per source and refused texts (s6-gen), precision and recall (s6-ear), the ECE of each source and the bootstrap interval of the pooled ECE (amendment 12).
- **Tranche 1 is done (12/11/2026, 04:43): the recipe is written, by the rules alone.** `runs/recipe/recipe.json`: AdamW at ×2, no `parallax`, no `xsa`, and none of soda, exit-kd, EMA, StableQAT, curriculum or no-polish. No arm reached the gain its rule asks for; the closest were `xsa` (+0.008 for 0.01) and `parallax` (+0.005), and Muon (+0.005) lost on Flappy. `train_minutes` was noisy on the arms that ran while the Mac was in use (up to ±16 min), but rule 1 compares two arms that ran at night (23.1 and 23.2 min, ±0.3) and rule 5 is moot under AdamW. Incidents, all technical: two stops asked by the owner, one session end that took the run down, one calibration that failed once and passed on retry, and a loss of file access (macOS privacy) on 11/11 from 09:45 to the evening.
- **The smoke passed (12/11/2026): every step of the five models ran end to end** under `runs/all-smoke/`, deleted since. Two fixes came out of it: a gluonts warning printed millions of times in the cast scoring, and a clearer report. **Tranches 2 to 4 started on 12/11/2026 at 09:06** (`scripts/train-all.sh ear secrets nano micro cast report`, log in `runs/train-all.log`), throttled by day and at 0.85 / 50:10 when the Mac is unused.
- **A memory defect found and fixed on 12/11/2026 (`docs/s6-ear-prereg.md`, amendment 4).** The s6-ear decision loader kept every clip's frames in memory: 22 GB on a 16 GB Mac, 25 GB of swap, and a decision run at half its normal step rate. The frames are now read from disk per batch (1.7 GB after loading, 4 to 5 GB while training). The decision run and its control start over; the first ones are kept in `runs/s6-ear-nano-swap/`. **The brake is now automatic** (`scripts/auto-brake.sh`, `SKY_BRAKE_FILE`): 0.85 and 50:10 after 10 minutes with no input, throttled when the Mac is in use, on battery, playing a video or too hot, with no restart. It also stops a training that holds more than 13 GB. Tranches 2 to 4 started again on 12/11/2026 at 15:34.
- **Next new model, right after tranches 2 to 4: `s6-sleep-nano` (§10).** An s6-nano with a fast-weight memory that "sleeps" between decisions (offline recurrence, arXiv 2605.26099), then the flexible weights of §7.3 consolidated during sleep, with dreaming in the game simulators (arXiv 2606.03979). Added on 11/11/2026; probes and a dated pre-registration come before any training.
- **Le prochain run : tout entraîner, au mieux (§9), sur le M5.** Le 05/11, l'utilisateur avait d'abord parlé de louer des GPU, puis a tranché : « finalement je ferai tout sur mon pc ». Il n'y a donc **ni Mac loué ni HF Jobs**. **La nuit 1, la recette s6, se lance le 06/11** (voir plus haut). Le §9 dit ce qui est prêt et ce qui reste à faire pour les nuits suivantes.
  - **Les 5 modèles s6**, avec les optimisations que le pilote et l'ablation adoptent (règles appliquées automatiquement) : environ 48 h de GPU.
  - **LFM3-1.7B** : le routeur S1/S2 (§6) et le fine-tuning (§6.1), en local. Il faut d'abord les amendements du §9.2 : le §6.1 et l'amendement 1 du §6 visaient HF Jobs.
  - **Depuis le 09/11/2026, tout tourne en mode bridé (`SKY_DUTY=0.5 SKY_COOL=45:15`)** : le Mac chauffait trop. Le GPU travaille la moitié du temps, avec 15 min de repos complet par heure, donc tout prend environ 2,7 fois plus de temps sans changer les résultats (voir plus haut). On découpe en tranches (§9.4) : tout reprend là où ça s'était arrêté.
  - **Le Mac peut se mettre en veille.** Rabattre l'écran suspend le run, qui repart à l'ouverture : ce qui s'est passé dans la nuit du 04 au 05/11 (« Clamshell Sleep » à 22:12) n'est plus un problème, cela coûte seulement du temps. `caffeinate -i` empêche seulement la veille d'inactivité quand l'écran reste ouvert.
- **Labels GLM-5.3-flash (§1.1) : finis le 05/11/2026 à 19 h 40, à 100 %.** 747 077 labels et 121 textes refusés par la modération, soit les 747 198 records.
  - L'accord avec la bonne réponse est de 560 238 sur 747 077 (0,750).
  - `data/distill/glm.jsonl` est écrit : c'est le mélange final, que `train-s6.sh` attend.
  - `skycmd-glm-labels` est publié en version complète (§1.2).
  - La boucle s'est arrêtée seule à 19:22, avec 267 records restants (99,96 %, au-dessus du seuil de 99 %). Deux passes `label` les ont finis, puis `mix`.
- **Données s6-ear téléchargées (§8)** : les comptes sont dans `DATA_LICENSES.md`, et il y a bien 50 mots réservés par langue, sans amendement.
- **Nos 14 jeux de données sont publics sur Hugging Face (§1.2)** : tout ce qu'un modèle lit et tout ce qu'on a créé nous-mêmes (jeux, trajectoires de nos modèles face à l'oracle, séries synthétiques, secrets, évaluations écrites à la main), les labels GLM compris (marqués comme générés par IA). Seuls manquent l'audio de s6-ear (31 Go, reconstruit par `sky-data fetch-audio`), le jeu gated et GIFT-Eval.
- **Données s6 prêtes (§1)** : le corpus ouvert, les tokenizers 2048 et 4096, et les tests zéro-shot. Celles de s6-secrets et s6-cast aussi (§3, §4).
- **Recette s6** : le pilote et l'ablation pico décident (§2). s6-nano passe en fenêtres 512 / 160, et le taux de chaque taille est centré par μP.
- **Les entraînements s6** : la nuit 1 (pilote, ablation, recette) est prête à lancer. s6-ear, s6-secrets, s6-nano, s6-micro et s6-cast suivent les nuits 2 à 4, une fois le §9.2 fait.
- **Lu et non adopté** (04/11) : ZHA, l'attention hiérarchique de Zyphra (arXiv 2610.42112). Elle ne bat l'attention dense qu'au-delà d'environ 48K tokens et marche mal sous 1B paramètres ; nos états font 64 à 512 tokens. Les raisons sont dans `docs/research.md` (« Not adopted yet »).
- **LFM3 fine-tuné (§6.1)** : pré-enregistrement figé (`docs/lfm3-ft-prereg.md`), scripts et tests prêts, **rien n'a tourné**. Les scripts visent encore Hugging Face Jobs. Depuis le 05/11, tout se fait sur le M5 : il faut l'amendement 1 de `lfm3-ft-prereg`, l'amendement 2 de `lfm3-s1s2-prereg` (qui annule le passage à distance) et le mode local de `lfm3-ft.sh` (§9.2). (Written on 09/11/2026: amendment 1 of `docs/lfm3-ft-prereg.md`, amendment 2 of `docs/lfm3-s1s2-prereg.md` and the local mode of `scripts/lfm3-ft.sh`; nothing of §6.1 has run: item 5.)
- **Issue #75 (Weight-Space Programming sur LFM3) fermée** au profit du routage S1/S2 (§6). Les idées de l'article sont reprises là où elles sont testables, sur nos petits modèles et avec l'arena : édition ciblée sans oubli, initialisation en une fois, poids flexibles en temps réel avec garde-fou conformal, audit des poids (§7). Son worktree a été supprimé le 06/11/2026 : il ne contenait aucun travail non mergé.

> **La labellisation GLM est finie (05/11/2026, §1.1).** Il n'y a plus rien à relancer. `scripts/distill-glm.sh count` doit afficher 0 record à labelliser.

Cinq modèles ont leur code prêt et ne sont pas encore entraînés : **s6-nano** et **s6-micro** (le s6 généraliste : les jeux et des questions en texte libre), **s6-secrets-pico**, **s6-cast-pico** et **s6-ear-nano** (l'audio, §8). Chacun a un pré-enregistrement figé. **s6-pico n'est plus un modèle final** (`docs/s6-prereg.md`, amendement 2) : son pilote et son ablation restent, parce qu'ils choisissent la recette des deux tailles. On suit chaque pré-enregistrement à la lettre :

- un amendement daté (§7, §8 pour `docs/s6-gen-prereg.md`) précède le run qu'il concerne ;
- on ne relance rien pour « voir » : seul un échec technique justifie un nouveau run, et il est noté dans le rapport ;
- on ne choisit jamais sur un checkpoint intermédiaire ;
- un modèle qui échoue à sa porte n'est pas publié, et le rapport le dit.

Les autres configs `s6-*` de `configs/` ne sont pas des modèles à entraîner :

- `s6-femto`, `s6-femto-xsa` et `s6-cast-femto` sont de minuscules configs aux poids aléatoires, pour les tests golden de `sky-infer` (`tests/golden_s6.rs`, `golden_s6x.rs`, `golden_cast.rs`) et les tests unitaires ;
- `s6-pico` et `s6-pico-xsa` sont les configs du pilote et de l'ablation (§2.1, §2.2), pas des modèles à publier.

Ordre (§9, tout sur le M5, une tranche par nuit) : labels GLM (§1.1, finis le 05/11) → pilote et ablation pico → recette (`scripts/s6-adopt.py`) → s6-ear → s6-secrets → s6-nano → s6-micro → s6-cast → LFM3 S1/S2 (§6) → LFM3 fine-tuné (§6.1). Les durées sont mesurées ou estimées sur le M5.

## 0. Prérequis

> **Status (16/11/2026): done.** The disk clean-ups are in item 9.

- [x] Toolchain Metal : `xcodebuild -downloadComponent MetalToolchain`, et `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Présente le 29/10/2026 (`xcrun metal --version` : 32023.921).
- [x] `export SKYCMD_STORE=$PWD` (le checkout principal) pour `cargo xtask` : `scripts/s6-recipe.sh` le fait tout seul.
- [x] `uv` installé (s6-cast : `scripts/gift-eval.py` tourne sous Python 3.12).
- [x] `node` et `wasm-pack` (présents le 29/10/2026) : l'étape `eval` de `scripts/train-s6.sh` construit `site/public/pkg` s'il manque (`scripts/build-wasm.sh`) pour mesurer la latence WASM.
- [x] Place disque : **95 Go libres** le 06/11/2026 (15 Go le 29/10/2026), après suppression du cache de build debug (`target/s3-train/debug`, 9,2 Go, produit par `cargo test` et clippy ; il se reconstruit tout seul). Les scripts d'entraînement n'utilisent que `target/s3-train/release`. Les runs écrivent dans `runs/` et `models/`, et les checkpoints de s6-micro (5,85M paramètres, état d'optimiseur compris) pèsent bien plus que ceux de pico. Revérifier avant 2.5, et supprimer à nouveau `target/s3-train/debug` après des `cargo test`.
  - **Mesuré le 09/11/2026 : 94 Go libres, pour environ 55 Go à écrire en tout.** L'essentiel, ce sont les caches du texte tokenisé : 866 Mo par pré-entraînement, identiques d'un run à l'autre, soit environ 43 Go pour la cinquantaine de runs. Un checkpoint pico pèse moins de 5 Mo. S'il faut de la place, `find runs -type d -name cache -prune -exec rm -rf {} +` les supprime une fois les étapes finies : ils ne servent plus aux résultats, seulement à recharger plus vite une étape relancée.

## 1. Données (une seule fois, avant le premier pilote)

> **Status (16/11/2026): done,** but for the `load_dataset` check of §1.2 (item 8).

`docs/s6-prereg.md` §1 : toutes les données sortent de `scripts/prepare-data.sh`, lancé une fois avant le pilote, sans régénération ensuite.

- [x] `scripts/prepare-data.sh`, fait le 29/10/2026 (journal : `runs/prepare-data-s6b.log`) :
  - `data/text-open` : 15 011 documents et 99,5 Mo de texte (97 Mo sur disque), et non 150 : le shard Wikipedia ne contient que 4,3 Mo (`docs/s6-prereg.md`, amendement 3) ;
  - les tokenizers `data/tok/tok-s6-2048.json` et `tok-s6-4096.json` (s6-nano, s6-micro). Le premier lancement plantait après avoir sauvegardé le tokenizer 2048 : le tableau de statistiques lisait `data/text-open` comme des décisions. C'est corrigé dans `sky-data train-tokenizer` ;
  - `data/commands/massive.jsonl` (32 341 énoncés), puis `data/s4/heldout` (`s4-corpus --only-heldout`) : les tests zéro-shot MASSIVE en/fr de s6-nano et s6-micro.
- [x] Noter dans le rapport final si les données GLM (`glm_orders`, `paraphrased`) sont présentes, c'est-à-dire si `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` étaient définis lors de `prepare-data.sh`. Le 29/10/2026, elles sont **absentes** de `data/decisions`. **Done:** recorded in `docs/big-run-report.md` §7, item 9: absent (`runs/prepare-data.log` of 25/10 skips `glm_orders`; no paraphrase file in `data/decisions/`).
- [x] Recopier les nombres de docs du tableau de `fetch-text --set open` dans `DATA_LICENSES.md`, section « MLM text, open set », à la place de « not fetched yet ».
- [x] Vérifier que `data/text-open/` contient les 6 fichiers : `dolma3_*`, `txt360_*`.
- [x] Vérifier que `data/s4/heldout/` contient `zs_massive_en.jsonl` (2 000 questions), `zs_massive_fr.jsonl` (1 000) et `zs_banking77.jsonl` (2 000). Ce dernier est rapporté comme « dans la distribution », parce que Banking77 est dans `data/decisions`.

### 1.1 Professeur GLM-5.3-flash pour s6-nano et s6-micro (`docs/s6-gen-prereg.md`, amendement 1)

> **Status: done (05/11/2026).**

GLM-5.3-flash est open source : poids sous licence MIT, [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). On l'appelle par l'API de z.ai pour ne pas faire chauffer la machine.

À faire une fois, avant le pilote de s6-nano. Il ne faut que `data/decisions`, déjà présent : on peut lancer dès maintenant, en même temps que le reste du §1 et que le pilote et l'ablation pico. Tout tourne chez z.ai : la machine ne chauffe pas.

- [x] `ccs list` doit montrer le fournisseur `zai` configuré (`ccs use zai` sinon). Le script lit l'endpoint et la clé dans la config de `ccs`, sans jamais afficher la clé ; `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` priment s'ils sont définis.
- [x] `scripts/distill-glm.sh count` (aucun appel) : 747 198 records (tous les records `train` des jeux Hugging Face de `data/decisions`) et 37 360 appels de 20 attendus. Un autre compte veut dire que `data/decisions` a changé : comprendre pourquoi avant d'appeler.
- [x] **Lancer le tout, sans surveillance :** `caffeinate -i scripts/distill-glm.sh loop mix`, dans un terminal qui reste ouvert (ou `mkdir -p runs && nohup caffeinate -i scripts/distill-glm.sh loop mix > runs/distill-glm.log 2>&1 &`).
  - **En cours depuis le 29/10/2026 à 16:28**, avec `nohup` (journal : `runs/distill-glm.log`). Pour savoir s'il tourne encore : `pgrep -fl distill-glm`. Cocher quand `loop` affiche « coverage reached » et que `mix` a écrit `data/distill/glm.jsonl`.
  - **Le 04/11/2026 à 19:15** : 539 229 sur 747 198 (72 %), 63 refus. La boucle s'était arrêtée à 11:38 (relance des sessions) ; relancée à 12:32, environ 3,3 labels/s depuis.
  - **Le 03/11/2026** : 350 949 labels sur 747 198 (47 %). Le redémarrage de la machine du 02/11 à 21:17 avait arrêté la boucle ; elle a été relancée le 03/11. Il reste 396 249 records, soit 19 813 appels.
  - **Refus de la modération z.ai** (HTTP 400, code 1301, amendement 3) : un lot refusé est coupé en deux jusqu'à isoler les textes refusés. Ceux-ci vont dans `data/teacher/glm_labels.refused.jsonl` et ne sont plus envoyés. `count` et le journal donnent leur nombre.
  - **4 workers** (`JOBS=4`), c'est le maximum utile. Mesuré le 29/10/2026 : z.ai plafonne vers 2,5 labels/s quel que soit le nombre de workers, et dès 8 workers une partie des appels part en relances (8 workers : 10 relances sur 24 appels ; 16 workers : 22 sur 32).
  - **Environ 80 h de labellisation**, réparties sur les fenêtres de 5 h de l'abonnement z.ai, avec de l'ordre de 100 à 200 M jetons en entrée et 55 M en sortie.
  - Quand une fenêtre est épuisée, chaque worker s'arrête après 3 lots ratés de suite. `loop` fait une pause de 30 min (`PAUSE=1800`) et reprend là où il s'était arrêté, jusqu'à ce qu'il reste au plus 1 % de la sélection (`COVER_MISS=1`). Il abandonne après 24 pauses sans progrès (`TRIES=24`), soit 12 h.
  - Avancement à tout moment : `scripts/distill-glm.sh count` (sans appel).
  - Si `loop` s'arrête avant la fin (machine éteinte, `TRIES` atteint), relancer la même commande : rien n'est perdu.
- [x] **Couverture : 99,98 %** (747 077 sur 747 198, les 121 refus comptés comme non labellisés), donc pas d'amendement. Les labels doivent couvrir au moins **95 %** de la sélection (`loop` vise 99 %, les textes refusés comptant comme traités). Les textes refusés par la modération comptent comme **non labellisés** pour ces 95 %. Sinon, un amendement daté avant tout run. Le rapport donne le nombre de refus par source (`glm_labels.refused.jsonl`).
- [x] **Fini le 05/11/2026 :** 560 238 accords sur 747 077 (0,750), le détail par source est dans `data/distill/agreement.json`. `mix` (lancé par `loop mix`, ou seul : `scripts/distill-glm.sh mix`) écrit `data/distill/glm.jsonl` et le taux d'accord par source (`data/distill/agreement.json`). À recopier dans le rapport avec le nombre d'appels et de jetons. Les essais du 29/10/2026 donnaient 15 accords sur 20, puis 19 sur 26.
- [x] Remplacer « 747,198 selected » et « the agreeing labels » par les vrais nombres dans `DATA_LICENSES.md` (section s6-nano / s6-micro).

### 1.2 Publier nos jeux de données sur Hugging Face

> **Status (16/11/2026): done,** but for the `load_dataset` check (item 8); the viewer is valid on all 14 datasets.

Demandé le 03/11/2026 : tous nos jeux de données sur le Hugging Face de l'utilisateur, **en public**, les labels GLM compris (en partiel maintenant, puis mis à jour à la fin). Un dataset repo par jeu, une config parquet par source, et une carte qui donne la source, la licence, la commande qui a produit les fichiers et le commit.

- [x] `scripts/hf-publish-data.py` (`--dry-run`, `--only`, `--private`, `--user`). Il assemble chaque jeu sous `target/hf-data/<repo>/` : un parquet par source, `MANIFEST.json` (le sha256 des fichiers d'origine) et la carte `README.md`. Le dry-run du 03/11/2026 donne :
  - `skycmd-decisions` : 27 sources, 823 387 lignes, 183 Mo ;
  - `skycmd-games` : 280 000 lignes ;
  - `skycmd-numeric` : 100 000 ;
  - `skycmd-secrets` : 201 242 ;
  - `skycmd-commands` : 32 341 ;
  - `skycmd-zero-shot` : 5 000.
- [x] **Publiés le 03/11/2026, en public**, sur `maxgfr` :
  - `skycmd-decisions` : 27 sources, 823 387 lignes ;
  - `skycmd-games` : 280 000 ;
  - `skycmd-numeric` : 100 000 ;
  - `skycmd-secrets` : 201 242 ;
  - `skycmd-commands` : 32 341 ;
  - `skycmd-zero-shot` : 5 000.

  Les liens sont dans le README, `DATA_LICENSES.md` et la page Sources.
- [x] **Le 05/11/2026 : tout ce qu'un modèle lit est publié.** Les 6 jeux du 03/11 n'ont pas changé : sha256 identiques à leur `MANIFEST.json` sur le Hub. Quatre nouveaux dépôts, en public, ont été ajoutés :
  - `skycmd-series` : les séries de s6-cast, 8 sources, 10 789 séries ;
  - `skycmd-text` : le texte MLM (`data/text` et `data/text-open`), 11 sources, 112 671 documents, plus les 3 tokenizers ;
  - `skycmd-ear` : les questions de s6-ear, l'index des clips et les codes Mimi, 1 233 211 lignes ;
  - `skycmd-lfm3-math` : les 600 questions de calibration et les 1 000 de test du routeur LFM3.
  - puis, à la demande de l'utilisateur (« sur Hugging Face, les données qu'on a créées nous-mêmes »), deux jeux de plus :
    - `skycmd-cast-synth` : un échantillon fixe de nos 4 générateurs de séries synthétiques et du mélange d'entraînement, 8 000 séries de 2 048 points (`sky-train cast-synth --seed 1`). s6-cast s'entraîne sur un flux sans fin des mêmes générateurs, pas sur ces lignes ;
    - `skycmd-evals` : notre matériel d'évaluation écrit à la main, à savoir les 56 cas des démos, `defs.json` et `paraphrases.json` de s5.
  - et un 14e, le plus original, `skycmd-trajectories`, avec `scripts/gen-trajectories.sh` et le nouveau `sky-arena --trajectories`, CPU seulement, en quelques minutes. Ce sont des parties complètes des 4 jeux, jouées par l'oracle et par nos modèles publiés s1-pico, s2-nano, s2-nano-doom et s3-pico : 1 033 412 décisions sur 2 100 parties, 17 Mo en parquet.
    - À chaque décision, il y a l'état, les probabilités du modèle, la cible de l'oracle, l'action, la récompense et les événements ; il y a aussi une ligne par partie. Le jeu sert à l'imitation et au DAgger, à l'apprentissage par renforcement hors ligne et à la calibration.
    - L'accord avec l'oracle va de 0,77 à 0,96 sur flappy, rescue et dodge. Sur doom, il est de 0,91 à 0,92 pour s2-nano-doom et s3-pico, et presque nul pour s1-pico et s2-nano, qui n'ont jamais vu doom : ces échecs complets sont gardés, ils sont utiles en hors ligne.
    - s4-micro et s5-lite sont écartés : ce ne sont pas des modèles de jeu.

  Non republiés : `LocalLLaMA/typed-decisions` (gated), l'audio de s6-ear (31 Go de log-mel, que `sky-data fetch-audio` reconstruit) et GIFT-Eval (évaluation seulement).
- [ ] Check that `load_dataset("maxgfr/skycmd-decisions", "boolq")` loads (item 8). The viewer part is done: on 16/11/2026, `https://datasets-server.huggingface.co/is-valid?dataset=maxgfr/<repo>` gives `"viewer": true` for all 14 datasets.
- **Règles appliquées par le script :**
  - chaque source garde sa licence, et les sources share-alike restent share-alike ;
  - `LocalLLaMA/typed-decisions` (le train et le bench) n'est **pas** republié : ce jeu est gated sur le Hub, et le republier en accès libre contournerait la barrière ;
  - le texte de pré-entraînement et les clips audio ne sont pas republiés. On publie leur recette : la commande `sky-data`, la graine et les sha256.
- [x] **Labels GLM-5.3-flash publiés en partiel**, sur décision de l'utilisateur (« c'est open source, tu peux publier »), malgré la clause §1(f)(xii) des [conditions de l'API z.ai](https://docs.z.ai/legal-agreement/terms-of-use). Cette clause interdit d'utiliser les sorties pour « training, labeling, fine-tuning … related to external models ». La licence MIT couvre les poids, pas l'API.
  - Le repo est `maxgfr/skycmd-glm-labels`. Il est marqué comme contenu généré par IA, comme le demande le §III.5(d).
  - Il a deux configs : `agree` (les records dont l'argmax du professeur est la réponse juste, avec pour cible les probabilités du professeur) et `labels` (tous les labels bruts).
  - Il ne contient ni les textes refusés par la modération ni `typed_decisions_train`.
- [x] **Republié le 04/11/2026 à 72 %** : `labels` 535 278 lignes, `agree` 403 040 (93,7 Mo). L'accord avec la bonne réponse est de 405 990 sur 539 368 (0,753) sur tout ce qui est labellisé, `typed_decisions` compris.
- [x] **`skycmd-glm-labels` publié en version complète le 05/11/2026** (`--glm-final`) : `labels` 741 437 lignes et `agree` 556 175, sans `typed_decisions_train`, qui est gated. Les commandes, depuis le checkout principal :
  - `target/s3-train/release/sky-data teacher mix-labels --labels data/teacher/glm_labels.jsonl --from data/decisions --out <tmp>/glm-mix --alpha 0` ;
  - puis `uv run scripts/hf-publish-data.py --glm <tmp>/glm-mix --glm-date "<date>" --glm-final`.

  Ne **jamais** écrire ce mélange partiel dans `data/distill` : `train-s6.sh` le prendrait pour le mélange final.
- **Vitesse de la labellisation sur z.ai** (03/11/2026). Le quota n'est pas la limite, la concurrence l'est. Les mesures :
  - 4 workers × 20 records par appel : environ 240 labels/min, sans erreur ;
  - 6 workers × 50 : erreurs HTTP 429 « Rate limit reached for requests » (code 1302) ;
  - 4 workers × 50 : environ 150 labels/min, plus lent (le temps de génération d'un appel croît plus vite que le nombre de records) ;
  - 5 workers × 20 : environ 240 labels/min, comme à 4 workers, avec des erreurs 429.

  - **Le 04/11/2026 : le prompt numérote les items (1 à 20) et passe en JSON compact.** Avant, GLM recopiait pour chaque item un identifiant de record de 16 chiffres hexadécimaux. Mesuré sur le même lot de 20 (deux essais) : 913 et 1 035 jetons générés au lieu de 1 337 et 1 449, et 11,3 et 13,7 s par appel au lieu de 16,2 et 18,0 s. `parse_labels` remet les identifiants de records. La consigne et les probabilités demandées ne changent pas. En continu, 4 × 20 : **4,7 labels/s** au lieu de 3,3 à 3,6 (+31 %), et 3,4 % des appels perdent un label au lieu d'environ 6 % (le journal n'écrit une ligne « 19 / 20 valid » que pour les appels incomplets).

  - **Le 05/11/2026, 6 workers × 20 avec le nouveau prompt :** 3,99 labels/s en 10 min, contre 4,7 à 4 workers, avec des erreurs HTTP 429 (code 1302), et un worker qui s'arrête après 3 échecs de suite. Retour à 4 workers.

  **La limite de z.ai porte sur le nombre de requêtes en parallèle (4 utiles) : on gagne en raccourcissant les appels, pas en ajoutant des workers.** Au 03/11, avec l'ancien prompt, le plafond était d'environ 4 labels/s. On garde le défaut, 4 × 20, relancé le 03/11 à 20:30. Il reste environ 333 000 records, soit environ 23 h de labellisation, plus les pauses entre fenêtres. Pour relancer : `nohup caffeinate -i scripts/distill-glm.sh loop mix >> runs/distill-glm.log 2>&1 &`.
- [x] Plus tard : le jev audio de s6-ear (§8). Les données sont prêtes depuis le 04/11 (31 Go de log-mel : voir s'il vaut mieux ne publier que les records et la recette). On y met les records et les log-mel, pas les clips (les sources restent sous leur licence CC-BY-4.0, avec attribution). **Superseded:** `skycmd-ear` (05/11/2026, above) already publishes the records, the clip index and the Mimi codes; the 31 GB of log-mel are not republished, since `sky-data fetch-audio` rebuilds them ("Non republiés", above); s6-ear v2 trains on Pipecat's public `smart-turn-data-v3.2-train` (`docs/s6-ear2-prereg.md` §3).

## 2. Le s6 généraliste : pilote et ablation pico, puis s6-nano et s6-micro (`docs/s6-prereg.md`, `docs/s6-gen-prereg.md`, `docs/SPEC-S6.md` §15)

> **Status (16/11/2026): done.** Tranche 1 (12/11), s6-nano (13/11), s6-micro v1 (14/11) and v2 (16/11): `BENCH.md`, section "skycmd-s6 (s6-nano, s6-micro)".

Pré-entraînement sur `data/text-open,data/text` (amendement 1), décisions sur `data/decisions,data/games,data/numeric`.

### 2.1 Pilote de learning rate pico (environ 2 h 30)

- [x] `cargo xtask s6lr`, soit 6 runs : `softmax-adamw` et `parallax-muon` à ×0,5, ×1 et ×2. La sortie est `runs/s6lr/summary.json`. **Done:** `runs/s6lr/summary.json` (the six runs and the extension), tranche 1.
- [x] Appliquer la règle §2 : garder ×1 sauf si une autre échelle bat la précision ternaire finale (`val`) d'au moins **0,01**, pour chaque optimiseur. **Done:** AdamW at ×2 (0.5775 against 0.553 at ×1), Muon at ×1 (`runs/s6lr/summary.json`, `lr_scale`).
- [x] Si le choix tombe au bord (×0,5 ou ×2) : une seule extension (×0,25 ou ×4), **par amendement daté avant de la lancer**. **Done:** the extension ran by itself (`cargo xtask s6lr --extend`; amendment 4 of `docs/s6-prereg.md` replaced the separate amendment): AdamW at ×4 scored 0.5245, so ×2 stays.

### 2.2 Ablation pico (environ 15 h)

- [x] `cargo xtask s6ab --adamw-lr-scale <a> --muon-lr-scale <m>`, avec les échelles du pilote : 11 bras × 3 graines (7, 11, 13). La sortie est `runs/s6ab/summary.json`. **Done:** `runs/s6ab/summary.json`, 11 arms × 3 seeds (tranche 1).
- [x] Ne rien changer aux données ni aux budgets entre le pilote et l'ablation. **Not a task:** a rule, and it held: amendment 6 of `docs/s6-prereg.md`, the only one written during the ablation, changes the brake mode only.

### 2.3 Décider ce qui est adopté (`docs/s6-prereg.md` §4)

- [x] Muon v2 contre AdamW : gain significatif (≥ 0,01 et plus grand que l'écart-type des deux bras), sans perte arena (> 5 % sur un jeu). Le gain exigé passe à 0,02 si le surcoût en `train_minutes` dépasse 15 %. **Done:** not adopted (gain 0.0051 for 0.01, and Flappy 91.43 against 141.7; rule `1_muon` of `runs/recipe/recipe.json`).
- [x] Parallax : `parallax-muon` contre `softmax-muon`, même règle. **Done:** not adopted, read with the kept optimizer, `parallax-adamw` against `softmax-adamw` (gain 0.0054 for 0.01; rule `2_parallax`).
- [x] XSA : `xsa-muon` contre `softmax-muon`, même règle. **Done:** not adopted (gain 0.0081 for 0.01; rule `3_xsa`).
- [x] SODA, KD de sortie, EMA, StableQAT et curriculum, lus contre `parallax-muon` : gain ≥ 0,005, sans perte arena, ECE pas plus de 0,01 pire. **Done:** none adopted (every gain below zero, three with an arena loss; rules `4_*`).
- [x] Sans polish : non inférieur (≥ −0,005) **et** au moins 2 % plus rapide. **Done:** not adopted (gain −0.0028; moot under AdamW; rule `5_nopolish`).
- [x] **Un commit, avant le pilote de s6-nano** (`docs/s6-gen-prereg.md` §2) : **Superseded:** by amendment 4 of `docs/s6-prereg.md`: `scripts/s6-adopt.py` writes the recipe to `runs/recipe/recipe.json` and the five configs to `runs/recipe/configs/`, which the scripts read; the recipe was committed with the gate table in `f8a052b` (14/11/2026).
  - dans `configs/s6-nano.json` et `configs/s6-micro.json`, `parallax` reste à `true` si la règle 2 l'adopte, sinon il passe à `false` (et corriger alors les nombres de `s6_parameter_counts`) ; `"xsa": true` si la règle 3 adopte XSA ;
  - noter `OPT` **sans taux** (`--no-muon` si AdamW est gardé, `--wd-mode soda`, `--ns-polish 0`), `LR_SCALE` (l'échelle du pilote pour l'optimiseur retenu), `PRETRAIN` et `DECIDE`. s6-secrets reprend `OPT` et `LR_SCALE`.
- [x] Décider **avant le run** si `DECIDE` contient `--ema`. Le checkpoint exporté en dépend (`last-ema` ou `last`). **Done:** no EMA in the recipe (`"ema": false`), so `last` is exported.

### 2.4 s6-nano (1,57M paramètres, environ 7 h)

Fenêtres 512 / 160 depuis l'amendement 2 de `docs/s6-gen-prereg.md`. Mesuré le 29/10/2026 avec `tok-s6-2048` : 1,8 % des états du bench tronqués, aucune tête. Avec 256 / 128, c'était 71,6 % des états. Labels GLM nécessaires (§1.1) : `train-s6.sh` refuse de démarrer sans `data/distill/glm.jsonl`.


- [x] Contrôle à blanc : `SIZE=nano DRY_RUN=1 OPT="…" LR_SCALE=… PRETRAIN="…" DECIDE="…" scripts/train-s6.sh lr train dagger exit eval`. Les chemins et les taux doivent être les bons, avec des points décimaux. **Superseded:** by `scripts/train-all.sh` and its `SMOKE=1` run of 12/11/2026, which ran every step of the five models end to end (the notes of 12/11 above).
- [x] `SIZE=nano … scripts/train-s6.sh lr` : le pilote ×0,5, ×1 et ×2 autour du taux pico déplacé par μP, √(96/144) = ×0,816 (20 + 20 min chacun). On garde le centre sauf si un autre taux gagne ≥ **0,01** en `val`. Pas d'extension. La sortie est `runs/s6-nano-lr/summary.json`. **Done:** `runs/s6-nano-lr/summary.json`; ×1 kept (`BENCH.md`, "Recipe").
- [x] `SIZE=nano … scripts/train-s6.sh train dagger exit eval`. Les étapes : **Done:** 13/11/2026 (`runs/all/nano-*.log`, `runs/s6-nano/eval/results.md`).
  - pré-entraînement 90 min ;
  - décision 90 min (`decisions=0.25,distill=0.30,games=0.35,numeric=0.10`) ;
  - 3 rounds DAgger de 15 min (oracle 0,2 → 0,1 → 0,05, 30 % de décisions gardées, dont 20 % du mélange GLM) ;
  - seuils CRC (α = 0,02) ;
  - évaluation et porte (`scripts/s6-gate.py`, résultat dans `runs/s6-nano/eval/results.md`).
- [x] Porte : `typed_decisions_bench` **> 0,398**, aucun jeu à −5 % de s3-pico, ECE **≤ 0,08**. Lue à τ = 1. Publié aux seuils CRC si elle y tient aussi. **Done:** passed at τ = 1 (0.480, ECE 0.0479, every game at s3-pico's score); the CRC thresholds failed in closed loop, so it is published at full depth (14/11).
- **Outcome (14/11, read again on 15/11).** Passed and published at full depth: 0.480, ECE 0.0479, every game at s3-pico's score. The study of 15/11 (`docs/s6-nano2-study.md`) shows that a constant answer per task passes this gate too: the most common training answer of each question scores 0.4815 (s6-nano 0.4805), and on 17 of the 20 bench tasks the model gives one answer to at least 95% of the items. The verdict stands. A narrower second version was studied and is not scheduled (the study's estimate of clearing all its gates: about 10 to 20%).

### 2.5 s6-micro (5,85M paramètres, environ 11 h)

- [x] Même contrôle à blanc avec `SIZE=micro`. Fenêtres 384 / 128 : 4,2 % des états du bench tronqués, aucune tête (mesuré le 29/10/2026 avec `tok-s6-4096`). **Superseded:** by the `SMOKE=1` run of 12/11/2026, as for s6-nano.
- [x] `SIZE=micro … scripts/train-s6.sh lr` (centre √(96/256) = ×0,612), puis `train dagger exit eval`. Les budgets : pré-entraînement 180 min, décision 180 min (batch 64), 3 rounds DAgger de 20 min. **Done:** `runs/s6-micro-lr/summary.json` (×0.5 kept), then 13 to 14/11/2026 (`runs/s6-micro/eval/results.md`).
- [x] Porte : `typed_decisions_bench` **≥ 0,45**, aucun jeu à −5 % de s3-pico, ECE **≤ 0,08**. Dire sans ambiguïté si l'objectif de **0,50** est atteint. **Done:** v1 failed on ECE only (0.0821; bench 0.547, the 0.50 goal met); v2 passed on 16/11 on a second look (0.553, ECE 0.0588; amendment 17 of `docs/s6-gen-prereg.md`).
- [x] Le rapport, quel que soit le résultat : **Done:** `BENCH.md`, section "skycmd-s6 (s6-nano, s6-micro)": the reference rows, "From nano to micro", MASSIVE en and fr, Banking77, the truncated records, the WASM lines.
  - les deux tailles face à s3-pico (0,382), s4-micro (0,343), oscar-1-17m (0,6775) et s5-lite (0,696) ;
  - la courbe nano → micro ;
  - MASSIVE en/fr zéro-shot, Banking77 « dans la distribution » ;
  - la part tronquée et la latence WASM.

## 3. s6-secrets-pico (`docs/s6-secrets-prereg.md`)

> **Status (16/11/2026): done.** v1 passed on its bench and was published (14/11); v2 and v3 failed G1 (15/11); the work on this model ends.

Après 2.3, parce qu'il reprend l'`OPT` et le `LR_SCALE` adoptés (amendement 2 de son pré-enregistrement). Pré-entraînement sur `data/text-open,data/text` (amendement 1).

- [x] `scripts/train-secrets.sh data` : jev `data/secrets` (50 000 textes, plus `secret_bench` de 2 000) et score du scanner à règles. Celui-ci doit redonner **0,977** (secret) et **0,787** (service), sinon on s'arrête et on comprend pourquoi. Fait le 29/10/2026 : 183 712 questions train, 9 546 val, 7 984 au bench, et le scanner redonne exactement 0,977 (précision 0,966, rappel 0,989) et 0,787. Résultat dans `runs/s6-secrets-pico/baseline.txt`.
- [x] Décider **avant le run** si `DECIDE` contient `--ema`, ce qui fixe le checkpoint exporté. **Done:** no EMA (`runs/recipe/recipe.json`), so `last` is exported (`BENCH.md`, s6-secrets-pico).
- [x] `OPT="…" LR_SCALE=… DECIDE="…" scripts/train-secrets.sh train eval` : pré-entraînement 8 min en séquences de 256 jetons, décision 15 min sur `data/secrets` seul, calibration, export ternaire, `s2-eval` sur `secret_bench`. Graine 0, une seule fois. **Done:** 12/11/2026, seed 0, once (`runs/all/secrets-train.log`: 6,336 pretraining and 7,555 decision steps).
- [x] Porte : exactitude secret **> 0,977** et service **> 0,787**, en ternaire, à pleine profondeur. **Done:** passed (secret 0.9990, service 0.9795; `runs/s6-secrets-pico/report.md`).
- [x] Le rapport donne aussi kind, line, la validation, la sortie anticipée, la taille et la latence, quel que soit le résultat. **Done:** `BENCH.md`, section "s6-secrets-pico" (per exit, early exit, validation, size, native and WASM latency).
- **Outcome of v1 and v2 (15/11/2026).** s6-secrets-pico passed its gate on its synthetic bench and was published on 14/11, then failed on real text (59.8% of 719 clean real files flagged at p >= 0.9; amendment 6). **v2, s6-secrets-verify-pico** (`docs/s6-secrets2-prereg.md`): a rule proposes the candidate, the model judges it in context; trained once on 14/11 (19:39 to 20:42, 11,296 pretraining and 14,117 decision steps) and tested once on 221 CredData repositories and the issue reports. **Failed on G1**: line F1 0.8561 against 0.7332 for Betterleaks, +0.123 where +0.15 was asked; G1 holds without the largest repository, and G2 to G7 hold. Not published, not rerun, not continued in this form, not in secretgate. Hand reading of the 110 clean-text candidates it accepts: none looks like a real credential. The whole-scan latency of §9 was not built. Numbers: `BENCH.md`, section "s6-secrets-verify-pico".
- **v3 (15/11/2026), the rules and the verifier together: failed on G1 again; the work on this model ends.** `docs/s6-secrets3-prereg.md`: no training; a line is flagged when secretgate's default engine flags it or when the v2 verifier accepts a candidate on it with P(real) >= tau3 = 0.34933 (chosen on the 16 validation repositories, committed in `370f60f` at 09:05, test read once at 09:07:55). Judged on the seven gates of v2, a second look at test files read once for v2, the system defined after that reading. G1: 0.8477 against 0.7332 + 0.15 on every repository (0.8062 against 0.6867 without the largest); the model alone was 0.8561. G2 to G7 hold. All seven held on validation, not on the 221 test repositories. No third reading. Numbers: `BENCH.md`, section "s6-secrets v3".

## 4. s6-cast-pico (`docs/s6-cast-prereg.md`, `docs/SPEC-S6-CAST.md`)

> **Status (16/11/2026): done** (v1 failed, published as a result); v2 is item 3.

Indépendant des autres s6, il ne lit pas de texte.

- [x] `scripts/train-cast.sh data`. Relancé le 29/10/2026 : tout a sauté. `data/series/`, les 194 fichiers de fenêtres GIFT-Eval (`data/gift-eval/`) et les références seasonal naive et TinyCast (`data/gift-eval-out/`) sont là.
- [x] Banc déjà validé le 26/10/2026 : seasonal naive nGMASE 0,99986, TinyCast porté 0,786 / 0,557. Si le banc est régénéré, revalider avant toute lecture de s6-cast. **Done:** the bench was not regenerated, and the run's harness gives the references back: seasonal naive 0.9999, TinyCast port 0.7856 / 0.5570 (`docs/big-run-report.md` §5).
- [x] `scripts/train-cast.sh pilot` : 4 runs de 3 000 pas, soit lr 1e-3, 3e-3 et 1e-2, puis la distillation TinyCast au taux retenu. **Done:** `runs/s6-cast-pilot/choice.json`: 3e-3 kept, distillation kept (0.05076 against 0.050891).
  - Learning rate : garder 3e-3 sauf si un autre taux baisse la pinball de validation d'au moins 2 %.
  - Distillation : on distille, sauf si la pinball est pire de plus de 1 % (sinon `TEACHER=none`).
- [x] `LR=… [TEACHER=none] scripts/train-cast.sh train export eval` : **480 min** d'entraînement (8 h), QAT sur les 20 % finaux, EMA 0,999, graine 1. Ensuite viennent l'export (`last-ema` si sa pinball ternaire est meilleure, sinon `last`), la calibration de sortie (α = 0,05, δ = 0,02) et GIFT-Eval avec et sans retournement. **Done:** 14/11/2026, 480 working minutes, 250,092 steps, `last-ema` exported, GIFT-Eval with and without flip (`runs/all/cast-*.log`).
- [x] Porte, sans retournement : **nGMASE ≤ 0,838 et nWQL ≤ 0,587**. L'objectif annoncé est de battre TinyCast (< 0,774 / < 0,545). Le rapport dit sans ambiguïté s'il est atteint. **Done:** failed (nGMASE 0.9471, nWQL 0.6646); the TinyCast goal not met either (`BENCH.md`, s6-cast-pico).

## 5. Publier un modèle qui passe sa porte

> **Status (16/11/2026): a standing checklist, not a task.** Applied to every published model; for the models of items 1 to 3 it is item 4.

- [x] GGUF dans `site/public/models/` et une entrée dans `site/public/models/manifest.json`. Pour s6-cast, c'est `site/public/models/forecast.json` et la page [Forecast](site/forecast/index.md). **Not a task:** a standing checklist, applied to every published model (11 s6 entries in the manifest); for the models of items 1 to 3 it is item 4.
- [x] Carte `hf/cards/<modèle>.md` et entrée dans `hf/spaces.json`, puis `scripts/build-wasm.sh && (cd site && npm run space:build)` et `python3 scripts/hf-publish.py --only <modèle>`. **Not a task:** part of the standing checklist (item 4 for the next models).
- [x] Tableau des modèles du README, section « Coming next » de `site/bench/index.md` (le modèle passe dans les résultats), section du modèle dans `BENCH.md`. **Not a task:** part of the standing checklist (item 4 for the next models).
- [x] Rapport : tous les chiffres pré-enregistrés, y compris les échecs, sur la page Results et dans le chapitre du tutoriel. **Not a task:** part of the standing checklist (item 4 for the next models).

Un modèle qui échoue n'est pas publié, mais son résultat va quand même dans le rapport et sur le site. (Superseded on 15/11/2026: by the owner's decision, a failed model is published too, as a result, its card and manifest entry saying first that it failed.)

## 6. LFM3-1.7B : Système 1 d'abord, Système 2 seulement si besoin (`docs/lfm3-s1s2-prereg.md`)

> **Status (16/11/2026): done** (failure, 15/11/2026).

**L'idée.** LFM3-1.7B réfléchit par défaut, et trop réfléchir peut le faire renoncer à une bonne première réponse. Sans réflexion, il peut aussi partir en boucle : l'essai de fumée l'a montré sur une question de GSM8K. On ne touche pas aux poids, on ne l'entraîne pas.
1. Il répond d'abord sans réfléchir (**Système 1**), et une petite sonde mesure sa confiance.
2. Il ne réfléchit, avec un budget borné, que si la confiance est basse (**Système 2**).
3. Le seuil vient du **contrôle conformal du risque** (α = 0,02), comme les sorties anticipées de skycmd : en espérance, au plus 2 points perdus par rapport à « toujours réfléchir ».

Le but est autant de précision, voire plus, pour bien moins de calcul.

**Outcome (15/11/2026): failure.** One run, from 14/11 at 23:58:15 to 15/11 at 12:44:33, exit code 0 (12 h 46 min, `runs/chain.log`), under the brake modes 0.85 with 50:10 until 08:31, 0.5 with 45:15 (on battery) from 08:31 to 09:08, then 0.95 with 55:5 to the end (`runs/brake.log`). Model `LiquidAI/LFM3-1.7B-MLX-4bit@deb2986b`, α = 0.02. Tables: `runs/lfm3-s1s2/report-calib.md`, `runs/lfm3-s1s2/report.md`; all of them in `BENCH.md`, section "LFM3-1.7B".

- **Budget B\* = 1,024**, chosen before the test as the best always-s2 accuracy on calibration (0.797 against 0.742 for s2-256). τ = 0.332, calibration risk 0.018.
- **Calibration** (600, s2-1024): s1 0.687, s2 0.797, router 0.802, oracle 0.820; tokens 535 / 934 / 1003; s1 share 0.85, s2 forced 0.33.
- **Test** (s2-1024): GSM8K (500) s1 0.824, s2 0.878, router 0.862, oracle 0.910, tokens 396 / 728 / 547, s1 share 0.90, s2 forced 0.23; MATH-500 (500) s1 0.510, s2 0.714, router 0.710, oracle 0.718, tokens 722 / 1188 / 1518, s1 share 0.78, s2 forced 0.46; all (1,000) s1 0.667, s2 0.796, router 0.786, oracle 0.814, tokens 559 / 958 / 1032, s1 share 0.84, s2 forced 0.35.
- **Claims** (the block of `runs/lfm3-s1s2.log`): H2 router 0.786 ≥ 0.796 − 0.02: **yes**; H3 router 1032 ≤ 0.5 × 958 tokens: **no**; H4 router ≥ max(s1, s2) + 0.01: **no**. Verdict: **failure**. H1: thinking helps on both tasks.
- s2-256 (reported): test all s1 0.667, s2 0.724, router 0.729, oracle 0.749, tokens 559 / 663 / 870.
- The checklist below is kept as written before the run.

**Prêt le 03/11/2026 :**
- `scripts/lfm3-s1s2.sh` (étapes `data smoke calib test route`) et `experiments/lfm3-s1s2/s1s2.py` ;
- le modèle `LiquidAI/LFM3-1.7B-MLX-4bit` (rév. `deb2986b`, 0,97 Go) déjà téléchargé ;
- les données écrites dans `data/lfm3/` : 600 questions de calibration, 1 000 de test ;
- un essai de fumée de 3 questions par mode, réussi (`runs/lfm3-s1s2-smoke`). Débit : 98 jetons/s sans réflexion, 80 avec, environ 4,5 s par question.

- **Le 05/11/2026, deux décisions de l'utilisateur se sont succédé.**
  - D'abord « pas en local » : le M5 chauffe trop. L'amendement 1 du pré-enregistrement a fait passer le run à distance (bf16, Hugging Face Jobs, vLLM ou transformers).
  - Puis « finalement je ferai tout sur mon pc ». Il faut donc un **amendement 2** qui annule le 1 et revient au protocole d'origine : MLX-4bit, mlx-vlm et le M5 (§9.2).
- [x] **La commande (§9.4, tranche 5), après l'amendement 2 :** `SKY_DUTY=0.5 SKY_COOL=45:15 caffeinate -i scripts/lfm3-s1s2.sh`. Elle enchaîne `data` (déjà fait, sauté), `calib`, `test` puis `route`. **Done:** after amendment 2 (09/11), one run from 14/11 23:58 to 15/11 12:44 (`runs/chain.log`, `runs/lfm3-s1s2.log`).
  - Durée estimée à pleine vitesse : environ 1 h 30 pour s1, 2 h pour s2-256 et 5 à 6 h pour s2-1024, sur 1 600 questions, soit **environ 9 à 10 h, donc 24 à 27 h bridé**. La calibration passe en premier (600 questions, environ 3 h 30 à pleine vitesse).
  - Inférence locale sur le GPU du M5. `SKY_DUTY` ajoute une pause après chaque question. Le critère H3 compte des jetons, et `seconds` reste le temps de génération : le mode bridé ne change aucun des deux.
  - Tout reprend après une coupure : relancer la même commande, les questions déjà faites sont sautées.
  - La labellisation GLM (§1.1) peut continuer en même temps : elle ne fait que du réseau.
- [x] **Au réveil**, lire `runs/lfm3-s1s2/report-calib.md` (calibration) puis `runs/lfm3-s1s2/report.md` (test, avec le verdict). Si la nuit n'a pas suffi, relancer `scripts/lfm3-s1s2.sh calib test route` : ça reprend. **Done:** both read on 15/11/2026; their tables are in `BENCH.md`, section "LFM3-1.7B".
- [x] **Critères** (pré-enregistrés, §6 du pré-enregistrement) : **Done:** H1 thinking helps on both tasks; H2 yes, H3 no, H4 no: failure.
  - **H1** : réfléchir aide-t-il ? C'est un diagnostic, sans seuil.
  - **H2** : le routeur garde la précision de s2, à 0,02 près.
  - **H3** : le routeur divise le calcul au moins par deux.
  - **H4** : le routeur fait mieux que les deux modes, d'au moins 0,01.

  **Succès** = H2 et H3. **« LFM3 amélioré »** = H4 en plus.
- [x] Rapport, quel que soit le résultat : les tableaux, l'oracle (le plafond d'un routeur parfait), la part routée vers s1 et la part de réflexions forcées. À mettre sur le site et dans un chapitre du tutoriel si c'est un succès. **Done:** `BENCH.md` and the Results page (`site/results/index.md`, LFM3-1.7B); no tutorial chapter, since it is a failure.
- [x] **Ensuite, seulement si le rapport le justifie :** **Superseded (not scheduled):** S1/S2 failed on H3 and H4 (15/11/2026) and the plan moved on to §6.1 (item 5); none of these follow-ups is planned, and each would still need a dated amendment.
  - un budget de 64 jetons ;
  - IFEval ;
  - une sonde sur les états cachés au lieu des log-probas ;
  - le Weight-Space Programming de l'issue #75 (fermée), mais seulement si le diagnostic montre un défaut que l'édition des poids pourrait lever (des boucles, par exemple).

  Chacun passe par un amendement daté du pré-enregistrement.

### 6.1 LFM3 fine-tuné (`docs/lfm3-ft-prereg.md`) — **rien n'a tourné**

> **Status (16/11/2026): not started, still open (item 5).** The Hugging Face Jobs plan below is superseded by amendment 1 of `docs/lfm3-ft-prereg.md` (09/11/2026: everything local).

> **After the failure of S1/S2 (15/11/2026).** `docs/lfm3-ft-prereg.md` says nothing about running the fine-tuning when S1/S2 fails: it sets no precondition on the router's result and does not change its goal. Its only sentences on the router are in §1, « Il ne remplace pas le routeur : le routeur est mesuré sur le modèle de base **et** sur le modèle fine-tuné, avec le même protocole (§5). », and in §6, « Ce verdict est celui du fine-tuning. Il est distinct du « LFM3 amélioré » du routeur (H4 de `docs/lfm3-s1s2-prereg.md`), que le rapport donne sous son propre nom (« H4 »). » Its amendment 1 reuses the base generations of `runs/lfm3-s1s2` as the reference if the conversion check of the smoke allows it. Whether to run it is the owner's decision; nothing has run.

> **Changement du 05/11/2026 : tout se fait sur le M5, plus sur HF Jobs** (décision de l'utilisateur). Le texte ci-dessous décrit encore la version HF Jobs. Avant le moindre calcul, il faut l'amendement 1 de `docs/lfm3-ft-prereg.md` et l'amendement 2 de `docs/lfm3-s1s2-prereg.md`, qui remplace le 1 (§9.2), ainsi qu'un mode local dans `scripts/lfm3-ft.sh`.
> - **Génération et évaluation :** avec mlx-vlm, comme `experiments/lfm3-s1s2/s1s2.py` (mlx-lm ne charge pas LFM3, et vLLM ne tourne pas sur Mac).
> - **La LoRA :** avec transformers + peft sur MPS, ou avec l'entraîneur LoRA de mlx-vlm. Un smoke local de quelques minutes tranche, avec une règle écrite avant.
> - **La génération locale est lente** : environ 80 jetons/s, soit environ 4,5 s par question. Le plafond pré-enregistré de 1 000 questions par tâche s'applique donc d'office : environ 2 000 questions × 4 tirages, soit de l'ordre de 10 à 15 h, à étaler sur plusieurs nuits.

**L'idée.** Le routeur choisit quand réfléchir, mais ne répare pas le Système 1 : sans réflexion, LFM3-1.7B peut boucler jusqu'à la limite (l'essai de fumée). Ici, on apprend au modèle à répondre juste et court sans réfléchir. C'est un LoRA entraîné sur :
- les solutions rédigées de GSM8K et MATH (train, MIT) ;
- ses propres bonnes réponses en s2-1024 (auto-distillation).

Aucune question de calibration, de test ou de MMLU-Pro n'y entre. Les labels GLM ne servent **jamais** ici : les conditions de z.ai l'interdisent.

**Où ça tourne.** Tout passe par Hugging Face Jobs, rien en local à part `route` et `gate` (CPU). Les artefacts vont dans le dataset **privé** `maxgfr/skycmd-lfm3-ft-work`, le modèle dans le dépôt **privé** `maxgfr/skycmd-lfm3-1.7b-s1`. Rien n'est public, et les poids dérivés restent privés tant que l'utilisateur n'a pas dit oui.

**Prêt le 05/11/2026, pas lancé :**
- `scripts/lfm3-ft.sh` ;
- `experiments/lfm3-ft/` : `smoke.py`, `gen.py`, `sft.py`, `eval.py`, `common.py`, `route.py`, `gate.py` ;
- les tests CPU : `python3 experiments/lfm3-ft/test_common.py`.

`DRY_RUN=1 scripts/lfm3-ft.sh smoke gen sft eval` affiche les commandes `hf jobs` sans rien lancer. Sans `DRY_RUN`, chaque job payant affiche son coût estimé et maximal, puis attend « y ». **L'utilisateur confirme le coût avant chaque job payant.**

Depuis le checkout principal à jour de main (`data/lfm3` y est déjà) :

- [x] **Smoke** (L4, 30 à 40 min, environ 0,50 $, au plus 0,80 $) : `scripts/lfm3-ft.sh smoke`. Le suivre avec `hf jobs ps`, puis `hf jobs logs <id>`. Il vérifie : **Superseded:** by the local mode (amendment 1 of `docs/lfm3-ft-prereg.md`, 09/11/2026): the smoke runs on the M5, `scripts/lfm3-ft.sh smoke` (item 5).
  - la révision du modèle de base, qu'il rend ;
  - si vLLM 0.33.0 charge LFM3 et s'accorde avec transformers ;
  - le gabarit avec et sans réflexion ;
  - les noms des modules LoRA ;
  - deux pas de TRL, la fusion et le rechargement.
- [x] `scripts/lfm3-ft.sh pull`, puis lire `runs/lfm3-ft/smoke/report.json` : `revision`, `engine`, `lora_targets_match`, `trl.ok`. **Superseded:** `pull` has nothing to do in the local mode; the smoke report is read from `runs/lfm3-ft/smoke/report.json` (item 5).
- [x] **Amendement 1 de `docs/lfm3-ft-prereg.md`** (daté, commité) : recopier la révision et le moteur. Les reporter dans `BASE_REV_PINNED` et `ENGINE_PINNED` en tête de `scripts/lfm3-ft.sh`. Si `lora_targets_match` ou `trl.ok` est faux, on s'arrête : l'écart passe d'abord par l'amendement. **Superseded:** amendment 1 became the local-mode amendment (09/11/2026); the revision and the LoRA method are pinned by a later dated amendment, after the local smoke and before `gen` (item 5).
- [x] **Génération** (A100, environ 2 h, environ 5 $, au plus 10 $) : `scripts/lfm3-ft.sh gen`. Elle reprend si le job est coupé : relancer la même commande. **Superseded:** local generation with mlx-vlm, capped at 1,000 questions per task (amendment 1; item 5).
- [x] **Entraînement** (A100, 1 à 2 h, environ 5 $, au plus 10 $) : `scripts/lfm3-ft.sh sft`. Puis `scripts/lfm3-ft.sh pull`, pour avoir `runs/lfm3-ft/sft/result.json` et le commit du modèle. **Superseded:** local LoRA by the rule of amendment 1 (transformers and peft on MPS, else mlx-vlm; item 5).
- [x] **Évaluation** (A100, environ 1 h 30 par modèle, environ 7,50 $ pour les deux, au plus 15 $) : `scripts/lfm3-ft.sh eval`. C'est un job pour la base, un pour le modèle fine-tuné. Le job de la base est aussi le run à distance du routeur du §6 (amendement 1). **Superseded:** local evaluation; the base may reuse the generations of `runs/lfm3-s1s2` if the conversion check of the smoke allows it (amendment 1; item 5).
- [x] **Lecture** (local, gratuit) : `scripts/lfm3-ft.sh pull route gate`. Elle donne `runs/lfm3-ft/report.md` et `gate.json`, et `runs/lfm3-ft/eval/<base|ft>/report.md` pour le routeur. **Superseded:** the same reading in the local mode, `scripts/lfm3-ft.sh route gate`, without `pull` (item 5).
- [x] **Portes** (pré-enregistrées, §6) : **Not a task:** the gates, unchanged by amendment 1, are read in item 5.
  - **G1** : précision s1 sur le test ≥ base + 0,03, et arrêts par limite ≤ la moitié de la base ;
  - **G2** : précision s2-1024 ≥ base − 0,01, et réflexion ≤ 0,6 × la base. Elle est rapportée à part ;
  - **G3** : MMLU-Pro (500, s1) ≥ base − 0,02.

  **« LFM3 amélioré »** = G1 et G3. Le routeur (H1 à H4) est rapporté pour les deux modèles.
- [x] Total estimé : **environ 18 $** (au plus 36 $ si chaque job va jusqu'à son délai). Pas de relance pour « voir », pas de choix de checkpoint, les échecs vont au rapport. **Superseded:** everything runs on the M5 and nothing is paid (amendment 1).

## 7. Poids flexibles : le Weight-Space Programming sur les petits modèles de skycmd (à faire après le §1.1 et le §6)

> **Status (16/11/2026): not started, still open (item 6).** `docs/weights-prereg.md` is frozen since 09/11/2026, with no amendment and no code.

**Ce que l'article du MIT a montré, et ce qui reste une vision.** Les cinq promesses n'en sont pas au même point :

| promesse | montré dans l'article ? | ce qui le montre |
|---|---|---|
| auditer un modèle sans le faire tourner | en partie | un modèle reconnaît le jeu (4 sur 4) ou la langue d'un réseau en lisant ses poids. Mais le modèle flagorneur n'est pas détecté par les poids seuls (0 sur 4) : il faut voir ses activations (3 sur 3) |
| initialiser un modèle en une fois | oui, sur de petits réseaux | des réseaux MinAtar écrits d'un seul coup : Breakout 517 contre 25 pour PPO après 500M pas |
| ajouter ou retirer une capacité sans données ni oubli | retirer un défaut, oui ; ajouter une capacité, non | Qwen3.5-0.8B débarrassé de ses boucles. Sur Gemma 3 270M, aucun gain : « ça lève un obstacle, ça n'ajoute pas de capacité » |
| greffer un circuit d'une architecture dans une autre | non, c'est une perspective | — |
| auto-amélioration récursive | non, c'est une perspective | — |

**Pourquoi skycmd est le bon terrain.** Les résultats solides de l'article portent sur de petits réseaux de jeux (17 000 à 1,5M poids), la taille de nos modèles : s3-pico fait 445 000 poids, s6-nano 1,57M. L'arena donne en plus deux choses en microsecondes, en local, sans chauffer la machine :
- un **retour** pour chaque édition ;
- une **mesure de l'oubli catastrophique** sur les 4 jeux à la fois.

Personne n'a publié ça sur des modèles ternaires.

- [ ] **7.1 Édition ciblée sans oubli** (la promesse 3). GLM-5.3-flash, via `ccs` et z.ai, lit les poids et les activations de s3-pico publié. Il propose des éditions locales pour corriger un défaut mesuré, par exemple les vetos de Dodge ou les crashs de Flappy, et l'arena dit si on garde l'édition. **Open (item 6):** protocol in `docs/weights-prereg.md` §4, which chose s6-nano as the base; an amendment comes before any computation.
  - Règle : une édition est gardée seulement si le jeu ciblé gagne et qu'aucun autre jeu ne perd plus de 2 %.
  - Contrainte propre au ternaire : on édite le checkpoint flottant d'avant la QAT, puis on réexporte. Une édition directe dans {−1, 0, +1} est une variante à mesurer à part.
  - **Constat du 09/11/2026 : le checkpoint flottant de s3-pico n'existe plus.** `runs/` n'en contient aucun, et Hugging Face n'a que les GGUF (`scripts/hf-publish.py`). Le pré-enregistrement du §7 doit donc choisir, avant tout run, entre réentraîner s3-pico (`scripts/train-s3.sh`) et partir du checkpoint flottant d'un modèle s6 qui joue, s6-nano une fois la tranche 2 finie (`runs/s6-nano-g3/decide/last`).
  - Bras : base, édition GLM à 10, 25 et 50 passes, une édition au hasard de même taille, et une baseline sans LLM (différence de moyennes des activations).
  - À lancer après la labellisation du §1.1, qui utilise le même quota z.ai.
- [ ] **7.2 Initialisation en une fois** (la promesse 2). GLM écrit les poids de la tête de décision d'une politique de jeu à partir de la description du jeu et des traits d'état. On mesure combien de pas d'entraînement cela fait gagner pour atteindre la qualité de s3-pico. Le but est un coût d'optimisation réduit, pas un modèle sans entraînement. **Open (item 6):** protocol in `docs/weights-prereg.md` §5.
- [ ] **7.3 Des poids flexibles en temps réel** (ton idée : « un modèle qui a des poids flexibles »). Un petit adaptateur de bas rang par-dessus le modèle ternaire figé, mis à jour pendant le jeu dans le navigateur. Chaque fois que le réflexe de sécurité (le veto) corrige le modèle, l'adaptateur apprend de cette correction. **Open (item 6):** protocol in `docs/weights-prereg.md` §7.
  - **Le garde-fou de skycmd.** Le contrôle conformal du risque décide si l'adaptateur reste actif : s'il dégrade la qualité au-delà de α par rapport au modèle figé, on revient au modèle figé. Ce sont des poids qui bougent, mais avec une garantie.
  - C'est la version mesurable de « l'auto-amélioration ». Les poids de base ne bougent jamais, donc on peut toujours revenir en arrière.
  - Les idées voisines dans la littérature : les « fast weights », l'entraînement au moment du test, les mémoires neuronales. Ce sont des souvenirs de modèle, à vérifier par une recherche avant le pré-enregistrement.
- [ ] **7.4 Audit des poids** (la promesse 1). On implante une porte dérobée dans une copie de s6-secrets, par exemple un déclencheur qui fait dire « pas de secret ». On regarde si GLM la trouve en lisant les poids, puis les activations. Il faut un petit entraînement local pour implanter la porte, donc c'est pour plus tard. **Open (item 6):** protocol in `docs/weights-prereg.md` §6.
- Hors de portée pour l'instant : la greffe de circuits entre architectures (promesse 4) et l'auto-amélioration récursive libre (promesse 5). L'article ne les montre pas, et 7.3 en est la version testable.

Chaque point passe par un pré-enregistrement figé avant le run, comme pour s6 et le §6. **Fait le 09/11/2026 : `docs/weights-prereg.md`** (base s6-nano, bras, règles et verdicts). Le code reste à écrire, après les tranches GPU.


## 8. s6-ear : un s6 qui écoute (`docs/s6-ear-prereg.md`, `docs/SPEC-S6.md` §16)

> **Status (16/11/2026): v1 done** (gates failed, published as a result on 15/11/2026). v2 is a new pre-registration, `docs/s6-ear2-prereg.md` (item 1); the tests left are in item 8.

Code complet : le front-end log-mel (Rust natif et WASM), le pré-entraînement « Mimi comme professeur », le jev audio, les questions `turn_end` et `keyword` (dont le zéro-shot), le contrôle conformal, les portes. Rien n'est entraîné. **Ne pas lancer en même temps que le run LFM3 du §6 : les deux utilisent le GPU.** La labellisation GLM du §1.1 peut continuer, elle ne fait que du réseau.

- [x] **Prérequis.** Le compilateur Metal de Xcode : sans lui, mlx ne se recompile pas, et `sky-train` ne se construit dans aucun nouveau dossier `target`. Il a été installé le 03/11/2026 avec `xcodebuild -downloadComponent MetalToolchain` (Metal Toolchain 27B47). Les scripts passent `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, parce que `xcode-select` pointe sur les Command Line Tools.
- [x] **Données** (`scripts/train-ear.sh data`, fini le 04/11/2026) : 31 Go dans `data/ear`. Les comptes sont recopiés dans `DATA_LICENSES.md` :
  - Smart Turn : 215 655 / 11 370 / 31 527 (train / val / test, le test étant `smart-turn-data-v3.2-test`) ;
  - Speech Commands : 40 538 / 4 861 / 4 890 ;
  - MSWC en + fr : 120 000 par split ;
  - LibriSpeech + Mimi : 28 539 / 2 703 ;
  - jev : `turn_end` 215 655 / 11 370 / 31 527, `keyword` 40 538 / 4 861 / 4 890, `keyword_zs` 113 238 / 109 049 / 2 000 (50 mots × 20 questions × 2 langues).
  - 50 mots réservés par langue (`held_out` de `data/ear/jev/REPORT.json`) : pas d'amendement.
  - **À dire dans le rapport :** la première archive MSWC suit l'ordre alphabétique. En anglais, 48 des 50 mots réservés commencent par « a » ; en français, ils vont de « a » à « d ». Les options d'entraînement aussi. Le zéro-shot en est plutôt plus dur : la première syllabe ne départage pas les options.
  - Fumée d'abord : `MAX_CLIPS=200 scripts/train-ear.sh data`. Le 03/11/2026, sur 200 clips par set :
    - MSWC fr : 200 / 200 / 200 (train / val / test) ;
    - LibriSpeech + Mimi : 200 / 200 (les codes Mimi ont une trame de plus que nos log-mel, la paire est coupée à la plus courte) ;
    - Speech Commands : 876 / 113 / 4 890 (le test officiel n'est pas plafonné).
  - Smart Turn v3.2 sur 25 clips de test de la fumée : exactitude 0,84, 31 ms par clip. Ces clips venaient encore du jeu d'entraînement de Smart Turn ; le test est depuis `smart-turn-data-v3.2-test`.
  - Fumée de bout en bout, le 03/11/2026 :
    - `s2-pretrain --audio`, 30 pas : la CE passe de 7,6 (le hasard sur 2 048 classes) à 7,0, à environ 100 000 trames/s une fois lancé ;
    - `s2-decide`, 40 pas, compilé, puis `s2-calibrate` et l'export ternaire : 1 615 681 paramètres, 901 Kio ;
    - `sky-eval` natif à pleine profondeur, par le chemin PCM : 0,6 ms par décision pour 1 s d'audio et 5 ms pour 10 s, log-mel compris ;
    - parité WASM sur 20 clips (`parity-node.mjs`) : écart maximal 2,4e-7, 20 argmax sur 20, 0,67 ms par décision en médiane.
- [x] **Pré-entraînement Mimi** (60 min) : `scripts/train-ear.sh pretrain`. Rapporter la précision du codebook 0 sur les trames masquées de val (`runs/s6-ear-nano/pretrain/log.jsonl`, `val_acc_cb0`). **Done:** 48,626 steps; `val_acc_cb0` 0.21963 against a technical bar of 0.00488 (`runs/all/ear-check.log`).
- [x] **Décision et témoin** (2 × 60 min) : `scripts/train-ear.sh decide`. Le témoin sans pré-entraînement est pré-enregistré : il faut le lancer aussi. **Done:** decision 48,205 steps (a first run discarded for a memory defect, amendment 4), control 49,131 (`runs/all/ear-train.log`).
- [x] **Évaluation et portes** : `scripts/train-ear.sh eval`. Smart Turn v3.2 tourne sur les mêmes clips de test (`uv run scripts/ear-baseline.py`, CPU). Ensuite viennent la parité WASM sur 20 clips et `scripts/ear-gate.py gate`. Les trois portes v1 sont : **Done:** `runs/s6-ear-nano/eval/gate.json`: three gates of four failed (`BENCH.md`, s6-ear-nano; the end-of-turn bar corrected to 0.9263 on 16/11).
  - fin de tour ≥ Smart Turn ;
  - mots-clés ≥ 0,90 ;
  - zéro-shot en et fr ≥ 0,20.
- [x] Publier si les portes tiennent (§5). On ajoute une démo micro sur le site : on parle, et le modèle montre en direct sa décision et sa confiance (`Engine.decideAudio`, PCM rééchantillonné à 16 kHz). **Superseded:** the gates failed; the file is published as a result (amendment 5 of `docs/s6-ear-prereg.md`, 15/11/2026) with its Space; a demo for v2 belongs to item 4.
- [x] **v2, après le §1.1** (même quota z.ai) : `sky-data teacher label-transcripts`, puis `sky-data audio-jev --about data/teacher/ear_about_labels.jsonl`, et un nouveau `decide` avec `about`. **v2 duplex** : écrire le téléchargeur AMI (la règle `audio::duplex_label` existe et elle est testée), puis la question `duplex_action` contre la règle « parler après 500 ms de silence ». **Superseded:** v2 is another design, end of turn only (`docs/s6-ear2-prereg.md`); its §3.5 leaves keywords, `duplex_action` and `about` out.

### 8.1 Le prochain run, pas à pas (s6-ear-nano)

> **Status: done (12/11/2026).**

À lancer quand le GPU est libre (pas pendant LFM3), depuis le checkout principal à jour de main, dans un terminal qui reste ouvert :

- [x] `SKY_DUTY=0.5 SKY_COOL=45:15 RESUME=1 caffeinate -i scripts/train-ear.sh pretrain decide eval` (`data` est fait), ou une étape à la fois. Il faut compter environ 3 h de GPU à pleine vitesse, donc 8 h bridé : 60 min de pré-entraînement, 60 min de décision, 60 min pour le témoin. Le téléchargement, d'environ 30 Go, se fait au fil de l'eau. `RESUME=1` reprend aux checkpoints (toutes les 10 min) après un arrêt : relancer la même commande. **Done:** run by `scripts/train-all.sh ear` from 12/11/2026 09:06 (`docs/big-run-report.md` §1).
- [x] Après `data` (04/11/2026, les nombres au §8), mettre dans le rapport :
  - les comptes par set et par split ;
  - le nombre de mots réservés par langue (`data/ear/jev/REPORT.json`, `held_out`) : il en faut 50 par langue. Moins de 50 veut dire que la première archive MSWC ne suffit pas. Il faut alors un amendement daté avec `--mswc-archives 2`, **avant** `pretrain`.
- [x] Après `pretrain`, rapporter `val_acc_cb0`. Il doit dépasser nettement 1/2 048 : sinon, c'est un échec technique, à noter, avant de lancer `decide`. **Done:** 0.21963 against 0.00488 (`runs/all/ear-check.log`).
- [x] Après `eval`, recopier dans le rapport : **Done:** `BENCH.md`, s6-ear-nano, and `docs/big-run-report.md` §4 and §7 (the gates with the control, the CRC thresholds and coverage, the WASM parity of 1.79e-5 on 20 clips, Smart Turn's precision 0.7832 and recall 0.9713; the model's own precision and recall were not computed by the run).
  - `runs/s6-ear-nano/eval/gate.json` (les trois portes, le témoin à côté) ;
  - `crc.json` (les seuils, la couverture du Système 1) ;
  - `wasm.json` ;
  - `baseline-smart-turn.json`.

### 8.2 Tester dans le navigateur

> **Status (16/11/2026): the demo is done;** the microphone check and the publication steps move to item 4.

- [x] **La démo est écrite** (04/11/2026) : la page `/ear/` (« Listen ») et le composant `EarDemo.vue`.
  - **Entrées :** le micro (`getUserMedia` et un `AudioWorklet`), ou l'un de 3 extraits CC-BY-4.0 joués par le même chemin (`site/public/ear/`).
  - **Audio :** un contexte audio à 16 kHz quand le navigateur le permet, sinon un rééchantillonnage en JS (le cas de Firefox) ; un tampon glissant de 10,24 s.
  - **Décisions :** `engine.decideAudio` toutes les 80 ms, pour `turn_end` et les mots-clés. La page affiche la décision, la confiance, la sortie utilisée, la latence, et « je passe la main » sous le seuil de renvoi (`route` dans le manifest, 0,5 sinon).
  - **Sans modèle :** si le manifest n'a aucune entrée `task: "ear"`, la page dit que s6-ear n'est pas encore entraîné. C'est le cas aujourd'hui.
- [x] **Testée dans Chrome** (04/11/2026), avec le modèle de fumée de 40 pas, en `docs:build` + `vitepress preview`. En `docs:dev`, Vite refuse d'importer le module WASM depuis `/public`, comme pour les autres pages.
  - le modèle se charge ;
  - les 3 extraits remplissent le tampon et donnent une décision toutes les 80 ms : 2,9 ms en médiane pour les deux questions, log-mel compris ;
  - changer d'extrait en pleine lecture repart bien de zéro (bug corrigé : les rappels de l'extrait précédent fuyaient dans le nouveau) ;
  - aucune erreur dans la console.

  Pour tester un modèle local, le copier dans `site/public/ear/dev/` (ignoré par git) et ouvrir `/skycmd/ear/?model=/ear/dev/<fichier>.gguf`.
- [x] **Le micro, par l'utilisateur** : la permission micro du navigateur ne s'automatise pas. À faire dans Chrome puis Safari, avec le modèle entraîné. Vérifier : **Superseded for v1:** its model failed its gates; the check moves to item 4, for s6-ear v2 if it is published with a browser demo.
  - que la permission est demandée une seule fois ;
  - que la décision suit la parole : parler, se taire, dire « yes » ou « stop » ;
  - que la latence reste sous 80 ms ;
  - qu'il n'y a aucune erreur dans la console.
- [x] **À la publication du modèle :** **Superseded:** v1 is published as evaluated, as a failed result (`site/public/models/s6-ear-nano.gguf`, manifest `task: ear`, `gate: failed`; amendment 5), with no `route`, no "Listen" entry in the navigation and no GIF; for v2 these steps belong to item 4.
  - copier `models/s6-ear-nano/s6-ear-nano-crc.gguf` dans `site/public/models/`, avec une entrée `manifest.json` (`task: "ear"`, `arch: "skycmd-s2"`, et `route: {"noul": ρ, "choice": ρ}` tiré de `runs/s6-ear-nano/eval/crc.json`) ;
  - ajouter « Listen » à la barre de navigation (`site/.vitepress/config.ts`) ;
  - enregistrer un GIF de la démo pour le README et la carte Hugging Face.

### 8.3 Que tout soit carré avant de publier

> **Status (16/11/2026): partly open (item 8).**

- [x] Les trois portes lues **telles quelles**, sans relance pour « voir ». Un échec est publié comme tel sur la page Results. **Done:** read as they were, and published as a failure on the Results page (`site/results/index.md`, s6-ear-nano).
- [ ] Run `cargo test -p sky-web --test parity` and `wasm-pack test --node --release crates/sky-web --test parity` (item 8). The parity itself is measured: max |Δp| 1.79e-5, 20 argmax of 20 (`runs/s6-ear-nano/eval/wasm.json`, `docs/big-run-report.md` §7).
- [ ] `cargo test` passe sur sky-infer, sky-data, sky-web et sky-gguf, et `cargo test --release -p sky-train` passe (avec `DEVELOPER_DIR` pointé sur Xcode). Clippy et fmt sont propres, et `bash -n` et shellcheck passent sur les scripts. **Open (item 8).**
- [x] Dans `DATA_LICENSES.md`, les vrais comptes remplacent « not fetched yet » (04/11/2026). Rien de `data/ear` n'est commité.
- [ ] Les chiffres sont cohérents partout, sans un seul « à mesurer » qui traîne : **Open (item 8):** the 0.9263 bar is corrected wherever it appears (16/11), but `site/novelty/index.md` §8 still describes s6-ear as designed ("Measured in the smoke run", transcripts labelled by GLM, which never ran) without its result.
  - SPEC-S6 §16 ;
  - `docs/s6-ear-prereg.md` ;
  - README ;
  - la page Bench (« Coming next » vers les résultats) ;
  - la page « What we bring » (§8) ;
  - le chapitre 26 du tutoriel ;
  - la carte `hf/cards/s6-ear-nano.md`.
- [ ] `cd site && npm run docs:build` passe (sans lien mort). Le Space Hugging Face est vérifié dans un vrai navigateur (le modèle se charge, la démo répond, la console est propre). **Open (item 8).**

## 9. Le grand run : tout entraîner, au mieux, sur le M5 (décidé le 05/11/2026, à lancer une prochaine fois)

> **Status (16/11/2026): done.** The big run ran from 12/11 09:06 to 14/11 19:32 (`docs/big-run-report.md`), LFM3 S1/S2 on 14 and 15/11.

**La demande.** « Entraîne tout au mieux avec tout ce qu'il faut », puis, le 05/11 : « finalement je ferai tout sur mon pc ». Concrètement :
- **Les 5 modèles s6** (s6-ear-nano, s6-nano, s6-micro, s6-secrets-pico, s6-cast-pico), avec les optimisations que le pilote et l'ablation pico adoptent. Le choix de l'utilisateur, « ablation automatique puis tout », est la version honnête de « toutes les optimisations » : une option n'est adoptée que si les règles pré-enregistrées le disent, et **aucune décision n'est prise à la main**.
- **LFM3-1.7B** : le routeur S1/S2 (§6) et le fine-tuning (§6.1).
- **Tout sur le M5.** Il n'y a ni Mac loué ni HF Jobs, donc rien à payer. En contrepartie, la machine chauffe pendant plusieurs jours, découpés en nuits (§9.4).

**Ce que ça change par rapport au plan avec location.**
- `TIME_SCALE` reste à 1, puisque les budgets sont ceux du M5. `scripts/speed-probe.sh` ne sert pas, et `cloud-mac.sh` n'est plus à écrire.
- L'ablation tourne entière sur une seule machine (environ 15 h), sans `--summary-only`.
- `s6lr --extend`, `s6-adopt.py` et la recette passée aux scripts servent tels quels.

### 9.1 Ce qui est prêt (05/11/2026, testé)

- **`scripts/s6-adopt.py`** applique les règles 1 à 6 de `docs/s6-prereg.md` §4, à la lettre et sans jugement. Ses tests sont dans `scripts/test_s6_adopt.py` (9 tests, `python3 -I scripts/test_s6_adopt.py`).
  - `adopt` lit `runs/s6lr/summary.json` et `runs/s6ab/summary.json`, et applique :
    - Muon contre AdamW, avec un gain exigé de 0,02 si Muon coûte plus de 15 % de temps ;
    - Parallax contre son pendant softmax, avec l'optimiseur retenu ;
    - XSA ;
    - SODA, KD de sortie, EMA, StableQAT et curriculum, lus contre `parallax-muon` (gain ≥ 0,005, pas de perte arena, ECE pas plus de 0,01 pire) ;
    - sans polish : ≥ −0,005 et au moins 2 % plus rapide ;
    - la combinaison des options adoptées.
  - Il écrit `runs/recipe/recipe.json` (chaque règle avec ses chiffres, le sha256 des deux résumés et l'horodatage) et les configs des 5 modèles, avec Parallax et XSA tels qu'adoptés, dans `runs/recipe/configs/`.
  - `env MODEL` donne les variables de chaque script.
- **`cargo xtask s6lr --extend`** lance l'extension unique (×0,25 ou ×4) d'un choix au bord de la grille, et rien sinon. Le résumé du pilote compte l'extension.
- **`cargo xtask s6ab --summary-only`** refait le résumé à partir des runs rapatriés des 3 machines, sans rien relancer. Les marqueurs « fait » de xtask sont propres à chaque machine et à chaque stage.
- **Le résumé de `s6ab`** enregistre aussi les drapeaux de pré-entraînement de chaque bras, ce qui est le cas du curriculum.
- **Les scripts prennent la recette.** Sans recette, ils gardent leur comportement d'avant : vérifié à blanc.
  - `train-s6.sh` et `train-secrets.sh` gagnent `CONFIG` ; `train-s6.sh` gagne aussi `DAGGER_N`.
  - `train-ear.sh` gagne `CONFIG`, `OPT`, `PRETRAIN`, `DECIDE` et `LR_SCALE` (le taux du pilote pico, déplacé à d = 144 par μP comme pour s6-nano). Avec `--ema`, il calibre et exporte `last-ema`, pour le modèle comme pour son témoin.
  - `train-cast.sh` gagne `CAST_OPT`. Son `pilot` applique désormais seul ses deux règles : il choisit le taux, distille à ce taux, puis tranche le professeur. Il écrit `runs/s6-cast-pilot/choice.json`, que `train` lit quand `LR` et `TEACHER` ne sont pas donnés.
  - Tous gagnent `TIME_SCALE`, qui multiplie leurs budgets en minutes.
- **`scripts/speed-probe.sh`** mesure la vitesse d'une machine : 300 pas de `s2-decide` sur s6-nano. `scale` donne `TIME_SCALE`, soit la vitesse du M5 sur celle de l'hôte, pour qu'un budget en minutes donne le même nombre de pas qu'aurait fait le M5.

### 9.2 Ce qui reste à faire avant le run (par Claude, sans GPU)

> **Status (16/11/2026): done,** but for the checks of its last box (item 8).

- [x] **Les amendements datés, commités avant le moindre run (écrits le 09/11/2026, chacun avec sa ligne sur le mode bridé) :**
  - [x] **`docs/s6-prereg.md` §7, amendement 4 (commité le 06/11/2026, avec `scripts/s6-recipe.sh`)** : les règles §4 sont appliquées par `scripts/s6-adopt.py`. La recette écrite dans `runs/recipe/recipe.json` remplace le commit manuel, et elle est commitée dans le rapport. L'extension au bord est lancée automatiquement par `s6lr --extend`.
  - **`docs/s6-ear-prereg.md` §8, amendement 2.** s6-ear tourne après l'ablation, avec la recette, pour le modèle et pour son témoin. Son taux est centré par μP. Le curriculum est sans effet en audio.
  - **`docs/s6-cast-prereg.md` §7, amendement 3.** s6-cast prend Muon ou AdamW, SODA, sans polish, StableQAT et XSA si l'ablation les adopte. Son pilote décide seul, par ses propres règles.
  - **`docs/s6-gen-prereg.md` et `docs/s6-secrets-prereg.md`** : une ligne chacun, pour dire que la recette passe par le fichier.
  - **`docs/lfm3-s1s2-prereg.md`, amendement 2** : il annule l'amendement 1 (à distance), et le routeur repart en local, avec MLX-4bit et mlx-vlm, comme pré-enregistré à l'origine.
  - **`docs/lfm3-ft-prereg.md`, amendement 1** : moteur local, LoRA par la méthode que le smoke local tranche, et le plafond de 1 000 questions par tâche. Les portes G1 à G3 ne changent pas.
- [x] **`scripts/train-all.sh`, l'orchestrateur, sur une seule machine (écrit le 09/11/2026 ; `SMOKE=1` écrit tout sous `runs/all-smoke/`, et `report` lit les portes avec `scripts/all-report.py`).** Chaque étape est idempotente, avec un marqueur dans `runs/all/done/`, un journal dans `runs/all/` et `DRY_RUN=1`. Les étapes, dans l'ordre :
  - `pilot` (`cargo xtask s6lr`, puis `s6lr --extend`) ;
  - `ablation` (`cargo xtask s6ab --adamw-lr-scale … --muon-lr-scale …`, avec les échelles lues dans `runs/s6lr/summary.json`) ;
  - `adopt` ;
  - les modèles, chacun avec `eval "$(scripts/s6-adopt.py env …)"` :
    - `ear` : `pretrain decide eval` ;
    - `secrets` : `train eval` ;
    - `nano` et `micro` : `lr train dagger exit eval`, après avoir vérifié `data/distill/glm.jsonl` et une couverture d'au moins 95 % ;
    - `cast` : `pilot train export eval` ;
  - `report`, qui rassemble les portes de chaque modèle dans `runs/all/REPORT.md`.
  - Un mode `SMOKE=1` avec des budgets de quelques minutes, `ROUNDS=1`, `DAGGER_N=200` et `EPISODES=2`.
  - **Le mode bridé et la reprise, par défaut.** `train-all.sh` exporte `SKY_DUTY=${SKY_DUTY:-0.5}`, `SKY_COOL=${SKY_COOL-45:15}` et `RESUME=1` (`sky-train` et les `train-*.sh` les lisent déjà). Attention : avec `RESUME=1`, un run reprend tout ce qu'il trouve dans son dossier. `SMOKE=1` doit donc écrire ailleurs que `runs/s6-*`, ou nettoyer ses dossiers, sinon les vrais budgets repartiraient d'un checkpoint de fumée.
- [x] **Le mode local de `scripts/lfm3-ft.sh`** (`gen`, `sft` et `eval` avec mlx-vlm ou MPS, sans `hf jobs`) et son smoke local, dans `experiments/lfm3-ft/local.py`. Il respecte `SKY_DUTY` et `SKY_COOL` comme `experiments/lfm3-s1s2/s1s2.py`. **Écrit le 09/11/2026, jamais exécuté hors dry-run** : le smoke local reste à lancer (§9.4).
- [x] **Dans chaque amendement ci-dessus :** une ligne pour dire que le run se fait en mode bridé, comme l'amendement 5 de `docs/s6-prereg.md`.
- [ ] **The checks still to run (item 8):** `cargo test -p xtask` and the `DRY_RUN=1` of the `eval` step of `lfm3-ft.sh` were never run (the others passed on 09/11/2026, before the code of 12 to 16/11); run them all again in the full test pass:
  - `cargo test -p xtask` ;
  - `python3 -I scripts/test_s6_adopt.py` ;
  - `python3 experiments/lfm3-ft/test_common.py` ;
  - `bash -n` et shellcheck ;
  - `DRY_RUN=1` sur chaque étape de `train-all.sh` et de `lfm3-ft.sh`.

### 9.3 Ce que fait l'utilisateur

> **Status (16/11/2026): superseded;** nothing here is a task now.

- [x] Garder le Mac ouvert quand c'est possible, de préférence sur secteur : le run n'avance que Mac ouvert. Rabattre l'écran le suspend, éteindre l'arrête, et il suffit alors de relancer la même commande (voir en tête du TODO). Pas de `pmset disablesleep`. **Not a task:** an operating reminder; `scripts/auto-brake.sh` now sets the modes by itself.
- [x] Launch each tranche (§9.4), or have the session's operator launch it on request. **Superseded:** tranches 1 to 4 and LFM3 S1/S2 ran from 09/11 to 15/11/2026; the runs left are items 1 to 3 and 5.

### 9.4 Le run sur le M5, en tranches (une fois le 9.2 fait)

> **Status (16/11/2026):** steps 1 to 6 and 8 are done (the smoke, tranches 1 to 4, s6-sleep v1's pilot, LFM3 S1/S2, `runs/all/REPORT.md` and `docs/big-run-report.md`); step 7 is item 5; step 9 is done for the published models and is item 4 for the next ones.

Chaque tranche reprend là où la précédente s'est arrêtée (marqueurs de `train-all.sh`, reprise de xtask et des scripts). Les durées sont celles du M5 à pleine vitesse : **à multiplier par environ 2,7 en mode bridé (`SKY_DUTY=0.5 SKY_COOL=45:15`)**. Une tranche peut s'étaler sur plusieurs jours, et les « nuits » ci-dessous sont des numéros de tranche.

1. **La fumée `SMOKE=1`**, environ 30 min : les 5 modèles de bout en bout, en quelques minutes chacun, sortent leur GGUF et leurs portes.
2. **Nuit 1 : le pilote et l'ablation pico, environ 17 h 30, avec `caffeinate -i scripts/s6-recipe.sh`** (le mode d'emploi est en tête du TODO). Le pilote prend environ 2 h 30, plus l'extension si besoin ; l'ablation prend environ 15 h. Puis `adopt`, qui écrit la recette. Elle peut tourner avant la fumée et le reste du §9.2, qui ne servent qu'à partir de la nuit 2.
3. **Nuit 2 : s6-ear, s6-secrets puis s6-nano, environ 11 h.** s6-ear prend environ 3 h, s6-secrets environ 30 min et s6-nano environ 7 h. Les labels GLM doivent être finis et mélangés (`data/distill/glm.jsonl`).
4. **Nuit 3 : s6-micro, environ 11 h.**
5. **Nuit 4 : s6-cast, environ 9 h** (pilote, puis 8 h d'entraînement).
   - **Then `s6-sleep-nano` (§10), before step 6.** It is not part of `scripts/train-all.sh`: its code, its probes and its pre-registration are written first, without GPU. The steps below keep their numbers.
6. **Nuit 5 : LFM3 S1/S2 (§6), environ 9 à 10 h**, avec `caffeinate -i scripts/lfm3-s1s2.sh`.
7. **Nuits 6 et suivantes : LFM3 fine-tuné (§6.1).** La génération prend de l'ordre de 10 à 15 h, puis viennent la LoRA et l'évaluation des deux modèles.
8. **Les résultats.** `report` rassemble tout. Chaque porte est lue telle quelle, sans relance pour « voir », et un échec est publié comme tel. On met à jour le TODO, `BENCH.md` et la page Results, et on commite la recette et `REPORT.md`.
9. **La publication (§5)** se fait seulement avec l'accord de l'utilisateur, modèle par modèle.


## 10. Next model: `s6-sleep-nano`, an s6 that sleeps (added 11/11/2026; pre-registered 12/11/2026, no code and no run yet)

> **Status (16/11/2026):** v1 stopped by its pilot (15/11), v2 ended at its check C4 (15/11), v3 in progress (item 2). The plan below is v1's; `docs/s6-sleep2-prereg.md` and `docs/s6-sleep3-prereg.md` are the reference where they differ.

**Status (15/11/2026): stopped by the pilot, no nano run.** The pico pilot ran on 14/11 (four runs of 3,684 steps, ended at 23:58). On the dev seeds of `recall`, the memory read at N = 1 scores 0.2460 against 0.2650 for the control without memory; the rule asked for +0.10, so "the memory works" is false (truncation kept: 0.2565 against 0.2430; drawn N not kept: 0.2460 against 0.2565). `scripts/s6-sleep-report.py` wrote `runs/s6-sleep/pilot.json` with no flags for the nano arm, and `scripts/train-sleep.sh` stopped: "the pilot says the memory does not work: no nano run". In each memory run, `recall` is the same at N = 1, 2 and 4. H0 to H3 are not tested, the test seeds are not read, nothing is published. Numbers: `BENCH.md`, section "s6-sleep".

**The code, checked after the pilot (15/11/2026): the negative result stands** (`docs/s6-sleep-diagnosis.md`; nothing built, nothing rerun). The number of passes reaches the computation; the write, carry and read paths are correct; every memory weight moved from its initial value; the memory never lowered the training loss (P1 −0.0007, P2 +0.0042, P3 +0.0015 against the control, where `recall` solved would be worth about 0.069); each model answers a near-constant option on the ask frames (P2 and P3 both right on 15 of 4,000 asks, about 250 expected at chance); the probe is sound. The gradients reaching the memory projections are 10 to 20 times smaller than those of the base blocks (a fact, not an explanation). No rerun.

- [x] **Still to add:** a unit test that a hand-written fact reads back as the right answer through the memory (hand-set weights; a show frame, idle frames, then an ask giving the right colour). No current test checks it. **Done:** `a_fact_written_by_hand_is_read_back_as_its_colour_at_the_ask` in `crates/sky-train/src/s2/memory.rs` (commit `f88f745`, 15/11/2026).

**Status (12/11/2026, evening).** On branch `s6-sleep-nano`, not merged while tranches 2 to 4 run (the run rebuilds `main` at each stage): the two probes (`sky-games`: `recall`, `rule110`), the memory layer and the sleep passes (`sky-train`: `s2::memory`), the sleep training (`s2-decide --sleep-records`), the probe reader (`s2-probe`), and the run (`scripts/train-sleep.sh`, `scripts/s6-sleep-report.py`). Unit tests pass and a 40-step code check ran on s6-pico in the three forms (drawn N, fixed N with full gradient, control). `docs/s6-sleep-prereg.md` has amendments 1 and 2, both before any run. **Left:** the pilot and the nano runs, after tranche 4. The `sky-infer` engine, the golden test and the WASM `sleep()` are written only if the hypotheses hold, because they serve the publication gate alone.

**Where it goes.** This is the next new model. It comes right after tranches 2 to 4 of the big run, and before the LFM3 tranches and §7, because it needs two things those tranches produce: the adopted recipe (`runs/recipe/recipe.json`) and s6-nano, its baseline.

**Sources.** The figures below were read in the papers themselves (arXiv HTML, 11/11/2026), not only in the video.

- bycloud, ["The Next Scaling Axis of LLM Is… Sleeping?"](https://www.youtube.com/watch?v=rAxTxqIHh9E), 10/11/2026, read with `maxgfr/ultrawatch` (auto-captions).
- Lee, McLeish, Goldstein, Fanti, *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference*, [arXiv 2605.26099](https://arxiv.org/abs/2605.26099).
- Behrouz, Hashemi, Javanmard, Mirrokni, *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, [arXiv 2606.03979](https://arxiv.org/abs/2606.03979).
- Feng et al., *In-Place Test-Time Training*, ICLR 2026 oral, [arXiv 2604.06169](https://arxiv.org/abs/2604.06169).
- Hernandez-Garcia, Figliolia, Millidge, *Can Scale Save Us From Plasticity Loss in Large Language Models?*, [arXiv 2606.24752](https://arxiv.org/abs/2606.24752).

### 10.0 What the papers show, and what it means here

- **Sleep as offline recurrence (2605.26099).** This is the core of `s6-sleep-nano`.
  - **Architecture.** Attention and gated delta network (GDN) blocks alternate. The GDN state is a fixed-size matrix of fast weights, updated as `S_t = α_t·S_{t−1} + β_t·v_t·k_tᵀ` plus a delta-rule correction.
  - **Sleep.** When the context window is full, the model stops reading and reruns the whole stack N times on the same chunk. Each pass rewrites the fast weights. Then the KV cache is cleared and only the fast weights stay. A prediction is still one forward pass.
  - **Training.** N is fixed per model, with full backpropagation through every pass (no truncation), Muon, and the loss only on answer tokens.
  - **Rule 110** (4-layer attention → GDN → attention → GDN hybrid, d = 256, context of 24 tokens, t = 32, about 55B tokens): exact accuracy is about 10% at N = 1, about 20% at N = 2, and above 30% at N = 3 or 4, with the same memory size.
  - **GSM-Infinite**, without chain of thought:
    - Jet-Nemotron 2B, 6 operations: 0.742 at N = 1, 0.812 at N = 6;
    - Ouro 1.4B, 6 operations: 0.419 at N = 1, 0.615 at N = 4.
  - **Limits stated.** Training cost grows about linearly with N and can be unstable. A reset of the state erases everything.
- **Sleep as consolidation and dreaming (2606.03979).** This is the idea behind 10.4.
  - **Knowledge seeding.** At sleep, a new low-rank expert (A, B) is added to the next, slower memory level. Only this expert is trained, so that the model without the temporary memory imitates the model that has it. Old experts are never overwritten, and the temporary experts are reset.
  - **Dreaming.** The model samples synthetic tasks, fine-tunes one copy per task, and is rewarded with 1 when the copy does better (ReST-EM, as in SEAL).
  - **Qwen3-1.7B:** AIME-24 goes from 49.8 to 53.2 and AIME-25 from 34.5 to 40.2. The best baseline, OPSD, reaches 51.6 / 40.0.
  - **SQuAD**, one passage: 31.9 for the base model, 33.4 fine-tuned, 46.7 for SEAL, 48.9 for sleep, and 35.7 for sleep without dreaming. Dreaming carries most of the gain.
  - **ARC with Llama-3.2-1B:** 0% in context, 10% with test-time training, 80% with sleep.
  - **Caveats.** No seeds or error bars. Several values the method depends on are not given (expert rank, divergence, reward weights).
- **In-place TTT (2604.06169).** The down projection of a gated MLP becomes fast weights, updated chunk by chunk with an objective aligned with next-token prediction, as a drop-in on a pretrained model.
  - **Gains are on long context.** RULER on Qwen3-4B: +4.4 points at 64k, but −0.5 at 4k.
  - **Implication.** Our tasks are short, so this is the least useful source for skycmd (10.5).
- **Plasticity loss (2606.24752).** The onset follows `T = 1.3·10⁻⁵ · P^0.8269`, in tasks of 5B tokens each. The 5M model loses plasticity after 6 tasks, that is 30B tokens.
  - **Implication.** A pretraining run of the s6 recipe sees about 40M tokens (4,800 steps in about 6 minutes at about 105k tokens/s, from `runs/s6-recipe.log`), several hundred times less. Plasticity loss is not a risk for the current runs, and measuring it is not a next step (10.6).

**Why skycmd is a good testbed for 2605.26099.**

- **Almost the same shape.** Its Rule 110 model is a 4-layer attention / GDN hybrid at d = 256. s6-nano applies its blocks 4 times (2 blocks × 2 loops) at d = 144.
- **Already recurrent.** The s6 encoder loops over shared weights (`state_loops`, SPEC-S3 §2) and carries one latent from frame to frame (`carry_gate`, SPEC-S3 §7).
- **Idle time for free.** A game has natural moments where no decision is asked, such as level changes or pauses, so sleep costs nothing at decision time.

### 10.1 The model

> **Status (16/11/2026): done,** but for the browser call of the last box (item 4, only if v3 passes).

- [x] **`configs/s6-sleep-nano.json`**: s6-nano (vocabulary 2,048, d = 144, ff = 288, 2 blocks × 2 loops, exits 1 to 4, 2 heads), with the recipe of tranche 1. **Superseded:** the memory's design is fixed by `docs/s6-sleep-prereg.md`, then v2 and v3 (`docs/s6-sleep2-prereg.md`, `docs/s6-sleep3-prereg.md`); there is no `configs/s6-sleep-nano.json`: `scripts/train-sleep3.sh` writes the nano config from the s6-nano checkpoint (its `config` step).
  - **Memory.** One GDN layer after each of the 2 state blocks, shared across loops like the blocks themselves: attention → GDN → attention → GDN, as in the paper. Each head holds a 72 × 72 fast-weight matrix per memory layer, so 20,736 floats per game stream, kept in f32.
  - **Parameters.** The new projections (q, k, v, out, gates) add about 170k parameters, roughly +10% over s6-nano's 1.57M. They are ternary like the rest, and the fast-weight state stays f32.
  - **The memory replaces nothing** (corrected 12/11/2026). s6-nano has no temporal carry (`temporal_carry` is off in its config), so the control arm is a model with no memory at all. A memory that starts at zero is a cold start, so stateless questions (`typed_decisions_bench`) work as before.
  - **Sleep.** Every K frames of a stream (K = 8, to be fixed in the pre-registration), and at every episode or level boundary, the model reruns the last K frames N times and only the memory changes. Then the frames are dropped.
  - **Wake decision unchanged.** One forward pass, the same exits and the same CRC thresholds, so the µs budget of the browser does not move.
- [x] **Training (sky-train, MLX).** **Done:** `crates/sky-train/src/s2/memory.rs` and `s2-decide --sleep-records` (v1, 12/11), the v2 memory (15/11), v3's held gates and auxiliary loss (`643e5ae`, 16/11); v1's pilot read the fast path.
  - Sequences of consecutive frames of an episode, as the carry training already does (SPEC-S3 §7), with sleep inserted every K frames.
  - The loss is only on decision frames. The optimizer is the recipe's, AdamW at ×2 (the paper uses Muon).
  - **Fast path (12/11/2026), three choices that cut the cost by about three.** Each one departs from the paper, so the pico pilot checks it once before the nano runs.
    - **Truncated backpropagation.** The first N − 1 passes run forward only; the gradient flows through the last pass and the wake frames. The paper uses full backpropagation and names truncation as the way to cut its cost. A step then costs about half of a full one at N = 4.
    - **One model for every N.** N is drawn from {1, 2, 3, 4} at each batch, in the spirit of `--loop-dropout`. A single training serves every sleep budget, and H1 is read on the same weights. The paper fixes N per model.
    - **Warm start from s6-nano.** The GDN layers are added to the trained s6-nano of tranche 2 with a zero output projection, so training starts exactly at s6-nano, as the drop-in of 2604.06169 does. No new pretraining, and a decision budget of a third of s6-nano's.
- [x] **Inference (sky-infer, Rust and WASM).** **Done for the engine:** the memory in `sky-infer`'s s2 streams (read, write, sleep passes, memory off), with golden parity against the training code natively and on wasm32 (`36ff13d`, `22a4f4e`, 16/11/2026). The browser part is the open box below.
  - A GDN kernel, with its golden test against MLX and the ternary export of its projections.
  - `decide_raw_stream` keeps the memory per `stream_id` (instead of the carry), and a new `sleep(stream_id, n)` runs between decisions.
  - [ ] `GameSession` calls the memory's sleep at level changes in the browser: only if s6-sleep v3 passes its gate, as part of its publication (item 4).

### 10.2 The probes: where memory matters

> **Status: done (12/11/2026).**

A null result on the 4 games would say nothing if they never need memory beyond one frame. So, before any training:

- [x] **A `sky-games` memory task.** A fact shown once (a door colour, a key, the side of a hazard) is hidden, and the decision that depends on it comes more than K frames later. The generator sets the delay, from 1 to 4K frames, so the result can be plotted against the delay. **Done:** `crates/sky-games/src/recall.rs` (commit `43d1a32`, 12/11/2026).
- [x] **Rule 110, at our scale.** The paper's own sanity check, in a reduced form: t ∈ {4, 8} instead of 32, because 55B tokens would take days on the M5. It checks that our GDN learns at all before the games. **Done:** `crates/sky-games/src/rule110.rs` (`43d1a32`); in v3's pilot nothing was learned on it (0.5443 for every arm, amendment 2 of `docs/s6-sleep3-prereg.md`).

### 10.3 Pre-registration and runs

> **Status (16/11/2026):** v1's pre-registration and pilot are done; the nano runs are v3's (item 2).

- [x] **`docs/s6-sleep-prereg.md`, frozen on 12/11/2026, before any code or training.** It fixes everything below, and it is the reference where it differs from this list (H0 added, budgets in steps, wake frames also write the memory, the two probes and their seeds):
  - **Hypotheses.**
    - H1: at equal wake latency, the score on the memory task rises with N.
    - H2: on the arena, `s6-sleep-nano` at its best N does at least as well as s6-nano.
    - H3: no loss beyond 0.01 on `typed_decisions_bench`.
  - **Arms and seeds.** Two arms, 3 seeds each: s6-nano fine-tuned for the same steps without memory (the control), and `s6-sleep-nano` with sampled N, read at N = 1, 2 and 4.
  - **Choice of N, by rule.** The smallest N within 1 point of the best on the dev seeds of the memory task.
  - **Test seeds**, read only once.
  - **Publication gate.** s6-nano's gate, plus H1.
- [x] **Pico pilot first** (done 14/11/2026: the memory does not beat the control, see the status above) (`s6-sleep-pico`, same changes on s6-pico), about 1 h throttled per run, 4 runs: the fast path; fixed N = 4 with truncation; fixed N = 4 with full backpropagation; the control. A shortcut is kept only if it loses less than 1 point on the memory task against the paper's way.
- [x] **Then nano.** The two arms above, 3 seeds each. Not run: the pilot's rule stopped it (15/11/2026). **Superseded:** v1's nano runs never ran (its pilot rule, 15/11); v3's six nano runs are item 2.
- **Cost, to confirm in the pre-registration.** About 5 h throttled for the pilot and 15 to 20 h for the nano runs, so 20 to 25 h in all (about 12 h in the night mode), instead of the 55 to 80 h of the first plan. If the pilot rejects a shortcut, the paper's way comes back for it, with its cost.

### 10.4 Then: the flexible weights of §7.3 with sleep and dreaming (2606.03979)

> **Status (16/11/2026): not started; only if justified (item 7).**

- [ ] **§7.3 changed into consolidation.** The vetoes of a session are collected instead of updating the adapter after each one. At sleep, a new rank-4 low-rank expert is grown and trained so that the frozen model with it imitates the frozen model with its temporary adapter. Then the temporary adapter is reset. **Open (item 7):** only if s6-sleep v3 and §7.3 justify it; a dated pre-registration first.
- [ ] **Dreaming without an LLM.** Self-play episodes in the `sky-games` simulators, started around the states of the vetoes. The reward is real: the arena score after consolidation is better, or not. This is where skycmd beats the paper: no synthetic curriculum to invent, and episodes cost µs. **Open (item 7):** same condition.
- **Arms.** Frozen; §7.3 online adapter; sleep without dreaming; sleep with dreaming. Every arm has the same CRC guardrail (α = 0.02).
- **The paper's lesson.** On SQuAD, dreaming carries most of the gain (48.9 vs 35.7). It is the arm to watch.

### 10.5 Optional: in-place TTT on LFM3-1.7B (2604.06169)

> **Status (16/11/2026): not scheduled** (item 7 says when it would be).

The gains are on long context only (−0.5 at 4k on RULER), and the §6 tasks are short, so the expected value is low.

- **First checks.** That LFM3's hybrid blocks have a gated MLP the method applies to, and that the released code runs on MPS.
- **When.** Only after §6.1, and only if §6 shows long-context failures.

### 10.6 Not a next step: plasticity loss (2606.24752)

> **Status: not a step.**

At the measured onset, a 5M model loses plasticity after about 30B tokens, several hundred times what an s6 run sees. Measure it only if a sleeping model is ever trained continuously for a long time, for example in the browser across many sessions.

**Rules.** Each item has a dated pre-registration before any computation. Nothing runs during a GPU tranche, and nothing is published unless it passes its own gate.
