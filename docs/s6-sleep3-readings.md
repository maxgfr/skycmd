# s6-sleep v3: how the code reads the pre-registrations

Written with the code of v3 (`scripts/train-sleep3.sh`, `scripts/s6-sleep3-report.py`, the memory of `sky-infer`), before any GPU run of v3 and before any dev2 or test seed was read. `docs/s6-sleep3-prereg.md` (v3) and `docs/s6-sleep2-prereg.md` (v2) are not changed; each item below is a place where their text left a choice open, and the choice the code makes.

## Flagged: nothing is impossible as written; one condition is read through v2's own amendments

- **"C1 to C3 of v2 on file as passed"** cannot be read on the raw files: `runs/s6-sleep2/checks/c2.json` says `"pass": false` (0.390 against the bar of 0.40) and `c3.json` says `"pass": false` (the cosine bar that amendment 1 withdrew). v3 §3 says "C1 to C3 of v2 stand as read", so the code reads them as v2 read them, from v2's `verdict.json`: C1 passed (contrastive, amendment 3), C2 taken by the stated deviation of amendment 4 (`c2_deviation`, form `frame`), C3 passed as the sanity check of amendment 1 (no memory weight but the read-out has a gradient at step 0). v2's own `"pass": false` comes from C4, which C5 replaces, and is not read. The pilot is refused unless these three hold and `c5.json` passes (`s6-sleep3-report.py ready`).

## C5 (v3 §3)

1. **Pooled over the asks.** Every reading of C5 is pooled over the ask frames of the recall steps of the window (each step's value weighted by its asks), as `scripts/s6-sleep3-trace.py splits` read the numbers that v3 §1 quotes; the mean of the per-step cross-entropies (v2's reading of C4) is written next to it and decides nothing.
2. **The window** is the recall sequence steps whose logged `step` is in [550, 600], both ends included (v2's C4 reader includes both); stateless steps carry no ask and are not read.
3. **"Last-exit accuracy on ask frames"** is the log's `ask_acc`, the share of ask frames the last exit answers right on the step's batch before its update; the cross-entropy is the log's `ask_ce`, the deep-supervision measure.
4. **"The same weights with the memory off"** is read on the steps that `--memory-off-eval 4` answered a second time (one sequence step in four from the first), pooled over their asks, cross-entropy from `off.ask_ce` (deep supervision) and accuracy from `off.ask_acc`. A window with no such step fails condition 2.
5. **Auxiliary accuracy** is the log's `aux_acc` (asks decoded right over asks times memory layers), pooled over the asks.
6. **The bars are inclusive** (≤ 0.30, ≥ 0.90, ≥ 1.30, ≤ 0.30, ≥ 0.95) and a reading that cannot be made (NaN) fails.
7. **"The step at which the answer left chance"**, reported and deciding nothing, is the diagnosis's definition (§6.1): the first step of the first ten consecutive recall steps whose mean ask-frame cross-entropy is under 1.36.
8. **The setting** is v2's C4 exactly as `scripts/train-sleep2.sh c4` ran it (the command of the diagnosis §7), with `--steps 600`, `--seed 0` or `1` (which draws both the batches and the fresh memory weights), and T1, T2 and `--memory-off-eval 4`. The config is built from the pico base with v2's memory and `memory_aux: true`, and the script refuses it unless it equals v2's `sleep2.json` beyond the name and `memory_aux`.
9. **One binary, recorded.** The two seeds run one after the other at nice 19 on the CPU, from a copy of the release trainer built at the commit the branch started from (no training code changed after it); the command of each run is kept in its `cmd.txt`.

## The pilot (v2 §5, v3 §4)

10. **T1 and T2 in both phases of M** (`--mem-gate-lr-mult 0 --memory-aux-weight 1`, with v2's `--mem-lr-mult 10`); phase B starts from phase A's `last` with a fresh optimiser, so held gates stay at zero. The control gets none of these flags (no memory, no auxiliary loss); its batches are drawn from the same seed.
11. **No `--memory-off-eval` in the pilot**: it is not asked for and would spend working minutes of a minutes-budgeted run.
12. **"Phase A's number of sequence steps is reported"**: `stop.json` holds the sequence steps of phase A (every game), its recall steps and the step at which its answer left chance (item 7).
13. **The stop rule** is v2's reader unchanged: the mean of the per-step ask-frame cross-entropies over the last 50 recall steps of phase A, above 1.336 stops.
14. **The splits** (by delay bucket, and a door shown with one colour against several) are computed from `s2-probe`'s per-ask records: `s2-probe` now writes, for each ask, the distinct colours its door had been shown with before it in the episode, counted as the training log counts them (`s2::seq::ask_splits`). "One colour" is exactly one; "several" is two or more. They are reported for C, for M at N = 1, 2, 4 and for M with its memory zeroed.
15. **The bootstrap** is v2's: paired over episodes, 10,000 draws, seed 0; "the bootstrap bound above 0" is the lower end of the 95% interval.
16. **The disclosure** of v2 §10 and of v3 §5 is written into `pilot.json` and `REPORT.md`, with amendment 4's deviation on C2.

## The nano runs, N, the test reading (v2 §6, v1 §6 and §7)

17. **Phases.** Every nano run has a phase A (v2's M4 mix, N = 1) and a phase B (v1's mix, N drawn) from phase A's `last`; the sleep arm at seed 0 trains 12 then 18 working minutes, and the five other runs train its steps phase by phase. The control's phases use the same mixes (same batches, no memory).
18. **The nano recipe** is v1's: the s6-nano of tranche 2 (`runs/s6-nano-g3/decide/last`), the records and weights of `scripts/train-sleep.sh`, `rate 144 LR_X` with `LR_X` from `runs/s6-nano-lr/summary.json`. v1's exception for a base that does not play is not checked by the script: tranche 2's arena already says it plays.
19. **Choice of N** on dev2 (v2 §6): the sleep arm read at N = 1, 2 and 4 on dev2 recall (with v1's reader), the smallest N within 0.01 of the best mean over the three seeds.
20. **The test seeds are read once** (`test/done`), and the step refuses unless all twelve phase runs ended with their final line and `n.json` exists. What is read is v1 §7's list: recall and rule110 at N = 1, 2, 4 (the control once) on the first 1,000 seeds ≡ 19 (mod 20) from 5,000,000,000; the four games at the chosen N through `s2-probe` from seed 5000 (200 episodes; v1 amendment 2); `typed_decisions_bench` through `s2-eval`'s ternary row (zero memory). The memory zeroed is not read on test seeds: no file asks for it there.
21. **H0 to H3** are read by v1's reader (its thresholds, its paired bootstrap over the three seeds pooled); the splits by delay and by colours are added to `REPORT.md` and decide nothing.

## Export and gate (v1 §6, §8; v2 §7; v3 §4)

22. **Export.** `s2-calibrate` on the nano records (stateless, so a zero memory) then `s2-export --format ternary`, as s6-nano; the file carries `mem.{l}.qk`, `v`, `out` ternary and the norm and the gates f32, and never `mem.{l}.aux` (`s2::gguf::to_gguf` drops it; the golden test checks that no tensor of a `memory_aux` model's files names it). `s2-export` now prints the count of the parameters it writes.
23. **Item 1 of the gate (the s6-nano gate)** on the exported seed-0 sleep model, with the protocol of docs/s6-gen-prereg.md §6: `sky-eval` at τ = 1 on the language test sets (a stateless question reads a zero memory, so it is answered as the base computes), `typed_decisions_bench` above 0.398 and ECE (`all`) at most 0.08; the arena with 4 games × 30 episodes at sky-arena's seeds for the model and for s3-pico, at τ = 1, the model played with its memory at the chosen N (`sky-arena --memory-passes N`, new: each episode one stream). No game may be more than 5% under s3-pico.
24. **Item 3 of the gate (parity)** is the golden test of the memory: a tiny random model of `configs/s6-sleep3-pico.json` exported in f32 and ternary, its stateless cases and five memory sequences (no sleep; N = 2 with a pause; N = 3; a noul question that exits at its first exit, so its frames are written at full depth after an early answer; the memory off), played by the training code and by sky-infer natively (`cargo test -p sky-infer --test golden_s6m`, every ternary compute path, with the size of the memory after every frame) and on wasm32 (`wasm-pack test --release --node crates/sky-web --test memory`). It is a test of the engine on the trained model's form, not of the trained model's weights.

## The engine (`sky-infer`)

25. **Frames.** A memory stream starts a new frame on a new state text or after `new_frame(stream, pause)`; the questions of one frame read the same memory and the frame writes once, when the next one starts or at `end_frame`, from the state encoder run to full depth whatever exit answered (training plays every frame at full depth). This is `s2-probe`'s order: write the previous frame, reset if the memory is off, sleep if `K` = 8 frames were seen since the last sleep or the game paused, then read.
26. **Only v1's write rule** (`memory_write` absent or `average`) runs in the engine; the rules of the diagnoses (delta, exact, centred, slot) were never trained and are refused at load. A model with both a memory and a temporal carry is refused (the training plays a memory without the carry).

## The script

27. **SMOKE=1** runs every step tiny under `runs/smoke-sleep3/` and reads no seed of the experiment: its probes start at 1,000 and 3,000 and its arena at 9,000, away from dev2 (2,000,000,000), the test seeds (5,000,000,000) and the arena's 5000. Its verdicts are printed and not obeyed.
28. **Devices.** C5 and every probe run on the CPU at nice 19; the pilot, the nano runs, calibration, export and `s2-eval` run on the default device (the GPU) unless `CPU=1`, and the pilot and the nano runs refuse to start while another training runs.
