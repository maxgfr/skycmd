# s6-ear v2 (s6-ear2-nano): the readings the code took

The places where `docs/s6-ear2-prereg.md` had to be read to be built, each with the reading taken.
They are the code's choices, not amendments: the owner accepts, changes or rejects them by an
amendment of that file before training. Written on 16/11/2026, before any training; no test label
or score was read (the harness was sanity-run on synthetic WAVs only).

## Flagged: places where the pre-registration cannot be applied as written

These five are the closest faithful readings of text that is impossible or self-contradictory as
it stands.

1. **"4 s2 encoder blocks (... GELU FFN ...)" and "reusing the s2 blocks" (4.2, 4.5).** The s2
   blocks have a GeGLU feed-forward (`ffn_up` of `[2·ff, d]`) and no attention norm on block 0.
   The study's counts (307,520 per block at d 160 / ff 640; 1.28M, 1.83M, 2.71M, 1.30M for the
   rungs) hold only for a plain GELU feed-forward (`ffn_up [ff, d]`, `ffn_down [d, ff]`) with a
   weight-only LayerNorm before both sublayers of every block, block 0 included, and no bias in
   the blocks. That is what is built; "reusing" is read as the s2 layout, names, ternary
   encodings, loaders and kernels. A test fixes the counts: 2,714,754 / 1,829,250 / 1,278,658 /
   1,306,178 (R1-R4).
2. **The PCM-hash rule against "the WAVs already written by v1's fetch" (3.1, 3.2).** v1 wrote
   each test clip as a 16-bit WAV of its last 128 frames of 80 ms (10.24 s at most), not the whole
   clip, and the study says no new test download. The rule therefore hashes the same tail of each
   training clip (decoded, cut on the clip-start frame grid, quantized as v1's writer does) and
   compares it with the test WAV's data. Equal whole clips have equal tails; equal tails of
   different clips are dropped too, a stricter rule.
3. **`seq_proj` "average-pooled by 4 into 25 steps" for R4 (4.2, 4.3).** R4 has 50 positions of
   80 ms; pooling by 4 gives 12.5 steps. The pooling factor is `positions / 25`: 4 for R1-R3, 2
   for R4, so the 25 steps of 160 ms stay aligned with the teacher's.
4. **"voicing = the peak normalised autocorrelation" with the test "f0 of a pulse train at 300 Hz
   within 1%" (4.1, 4.5).** At 300 Hz, lags 53.3, 106.7 and 160 samples are all in the 40-200
   search and all near 1 after the window normalisation; the global maximum often lands on a
   multiple of the period and the test fails. The peak is taken among the local maxima with
   Praat's octave cost (0.01 per octave towards the shorter lag, Boersma 1993); voicing is the
   parabola-refined value at that peak.
5. **The disk plan (3.4) against the disk at the time of the build.** §3.4 counts a peak of about
   15 GB with 25 GB free. On 16/11/2026 the disk had 14-15 GB free, and the build is told to leave
   at least 8 GB: the build stops cleanly at that floor (exit status 3) and resumes later. The test
   features also take about 1.1 GB, not 0.5 GB, because most test clips are longer than 4 s.

## The features (4.1)

6. **The hop grid.** Hop `t` reads the 40 ms span `[t·160, t·160 + 640)`; the 25 ms mel window is
   centred in it (samples 120-520 of the span). A hop exists once its whole span has arrived, so a
   stream and a batch give the same hops bit for bit; the grid starts at the first sample, and up
   to 10 ms plus the half-span (20 ms) at the end of a clip are not the centre of any hop.
7. **Stored values.** Raw log10 mel energies (floor 1e-8), not v1's `(log10 + 2) / 4` map:
   "values are stored before normalisation" and the rule is written on log10 values.
8. **Energy "on the log-mel scale".** log10 of the sum of the 40 mel filter energies of the same
   25 ms window, floor 1e-8.
9. **"The energy channel takes the same shift".** `(x − M + 4) / 4`, without the clamp at `M − 8`
   (the clamp belongs to the mel channels).
10. **The window's M.** Over the hops that enter the positions: the last `4·P` hops of the window,
    `P = hops / 4`; the grid of positions ends at the last hop, and the leading remainder hops
    are dropped.
11. **The autocorrelation.** No mean is subtracted before the window; the normalized value is
    clamped to [0, 1]; a span with no energy, or with no local maximum in 40-200, has voicing 0
    and f0 channel 0. The f0 channel is `log2(f0 / 100)` when voicing ≥ 0.45.
12. **The test signals of §4.5.** The pulse trains are 1.5 ms Hann pulses at fractional periods (an
    impulse train at 300 Hz has no integer period); white noise is a seeded xorshift. Measured:
    pitch within 1% at 100, 200, 300 Hz with voicing above 0.9; white-noise voicing mean 0.155,
    maximum 0.274.
13. **Determinism.** Every log is the IEEE-only `log10_det` of v1 (log2 through it), so native and
    WASM agree bit for bit on the hops (`crates/sky-web/tests/parity.rs`, passed on wasm32).

## Data (3)

14. **The training release's revision.** Not named by the pre-registration: pinned to the commit of
    the build, `f8c95d284595defb6b88a715c356cbc5c31b7ad1` (83 shards, hash-checked).
15. **eot-bench "revision `main`".** Downloaded at `main`, the commit it resolves to recorded in the
    report (`ae834af9...` on 16/11/2026).
16. **The 5% validation hash.** v1's `split_of` with seed 1234 and share 0.05.
17. **Clips too short for one position** (under 40 ms of hops) stay in the store and in the
    teacher's pass but are never trained on or validated.
18. **"HF datasets decoding" of the teacher.** The `Audio` feature's decoder of `datasets` < 4
    (soundfile, librosa) on the parquet bytes, without building the library's Arrow cache, which
    would double the disk of each shard.
19. **Test labels.** No builder step reads them: `turn-test` writes ids, hops and hashes only. The
    labels are read by the harness check (§8.1) and by the gate.

## Augmentation (3.3)

20. **Context jitter.** With probability 0.3 the kept length is drawn uniformly among the position
    counts of 1-4 s (25-100 positions; 13-50 for R4) and capped by the clip; the window is then
    normalised on what is kept.
21. **Masks.** Applied to every training view (no probability is given), widths uniform in 0-4
    bands and 0-8 hops (SpecAugment), masked values set to 0 after the normalisation. The time
    mask covers the 43 channels and never the last 100 hops of the kept window ("the first 3 s"
    of a 4 s window; O16's "away from the turn's end" for a shorter one).
22. **Sampling.** Shuffled epochs, each a function of the seed and the epoch, so every training
    clip has the same probability and a resumed run sees the same batches.

## Teacher and loss (5)

23. **u8 states.** Zero point 128, scale `max|x| / 127` per 384-d step vector; e_T in float16;
    z_T as float32.
24. **Short clips and jittered windows.** A step of the 25 counts when all its positions are in the
    student's window, which lies inside the clip.
25. **The hard-label arm** is the label term alone at weight 1.
26. **`pool_proj` and `seq_proj`** have no bias; the targets are layer-normalized without affine
    weights (eps 1e-5) before each cosine.

## Run (6)

27. **Head counts.** Not stated: d 192 uses 3 heads of 64, d 160 uses 2 heads of 80, near the
    repository's head widths.
28. **The recipe's default rate.** The decision phase's 1e-3 × `LR_SCALE` (2.0 in
    `runs/recipe/recipe.json`) × √(96 / d), as `scripts/train-ear.sh` moved it to d = 144; the
    optimizer flags of `scripts/s6-adopt.py env s6-ear-nano` (`--no-muon`: AdamW). Pilot rates
    0.5×, 1× and 2× that.
29. **The pilot arms.** No QAT and no EMA; each arm is judged on its final f32 validation accuracy
    (`σ(z) ≥ 0.5`); between equal rates the lower is taken; the stop rule reads the best of the
    four arms.
30. **Validation accuracy during training** is read on the uncalibrated logit (z > 0): a
    temperature does not move the decision at 0.5.
31. **The validation curve.** Every 2,000 steps on the first 2,000 validation clips (a full pass
    over about 13,500 clips would take a sizeable share of the working minutes); the validations
    at the QAT switch and at the end, which the checkpoint rule reads, use every validation clip.
    Validation time counts as working time, as in s6-cast.
32. **Batch on the GPU, CPU sanity.** `turn-train --cpu` runs the graph on the CPU (tests, sanity);
    every pre-registered run is without it.

## Checkpoint, calibration, export (7)

33. **"Ternary validation BCE"** is the one the training log records at the end, the weights
    fake-quantized in the graph (f32 scales), as `scripts/train-cast2.sh` picks; the file stores
    f16 scales.
34. **The temperature** minimizes the validation NLL of the ternary file's logits by a
    golden-section search of log T over [0.05, 20]; an uncalibrated export is used for the fit,
    then the file is exported again with the temperature and thresholds.
35. **τ₈₅** is the ⌈0.85 · n⌉-th largest calibrated probability among the finished validation
    clips: the highest threshold with END recall ≥ 0.85. v2 answers "finished" when `p ≥ τ`;
    Smart Turn keeps its own `p > 0.5`.
36. **WASM parity on 20 test clips** goes through the WAV path natively (`sky-turn --records`) and
    `TurnDetector.probEndPcm` in Node; "the same decision" is at τ₀.₅. No label is read.

## Evaluation and gate (8, 9)

37. **The harness check's timer.** "Must give 0.4979" is read as the test's share of finished
    clips rounded to four decimals; the check also asks for all 31,527 clips and the left padding.
38. **v2's test probabilities** come from the deployed path (each WAV through the front end and the
    file as exported), not from the float16 hops `turn-test` stored.
39. **ECE** uses 10 equal-width bins of the calibrated probability. Bootstrap intervals are 95%
    percentile intervals over 1,000 resamples of clips, seeded.
40. **Subsets.** English is language `eng` (v1's manifest uses ISO 639-3); the English human
    subsets are its clips of `human_5` and `human_convcollector_1`. Speko's oracle point is
    computed on all test clips and on that subset, reported only.
41. **Teacher-student agreement.** Validation: `p ≥ τ₀.₅` against the stored z_T > 0. Test: against
    the harness check's per-clip dump of Smart Turn (`p > 0.5`).
42. **eot-bench.** The cut is `round(start · 16000)` samples; v2 reads the last 400 complete hops
    before it, Smart Turn the last 8 s left-padded; a cut either model cannot decide (under 40 ms
    of audio) is left out of both and counted. v2 is read at its stored τ₈₅, Smart Turn at its 0.5;
    the timer answers `eot` everywhere. Cohorts and languages are those of the data. Smart Turn's
    pass runs file by file while the parquet is on disk; its probabilities are written and not read
    until the gate.
43. **G3.** "≤ 3 MB" is 3,000,000 bytes. The front end per hop is measured by pushing 10 ms chunks
    into a stream (median of five passes); the decision time includes the window's normalisation
    (`probEnd` on the stream's 4 s), the features being computed.
44. **The probe's random weights** are the training initialisation (seed 7) exported ternary; the
    rule reads the WASM medians of R1 to R4 in order.
45. **SMOKE** trains 5 working minutes on 2,000 clips (pilot arms of 1 minute) and evaluates on
    validation clips in place of the test clips, with no harness check and no eot-bench: no v2 test
    score exists before §8.1 and §8.3 are on file (§13).
46. **The steps of `scripts/train-turn.sh` (4.5).** "data, teacher" run as one step, shard by
    shard (the teacher needs the shard on disk, and the shard is removed once its targets are
    written); "smoke" is `SMOKE=1`, the whole chain under a throwaway prefix, as in
    `scripts/train-cast2.sh`; the harness check, the latencies and eot-bench are steps of their own
    so that §13's order is enforced by the script.
