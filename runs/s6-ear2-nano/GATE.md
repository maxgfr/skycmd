# s6-ear2-nano: gate (docs/s6-ear2-prereg.md §9)

**Verdict: PASS.** v2 passes.

Take v2 when size and speed matter: it gives up 2.7 points of accuracy against Smart Turn for a file of 0.63 MB instead of 8.7 MB and about 2.4 ms per decision in WASM instead of about 32 ms natively; take Smart Turn's file when those points matter more.

| term | value | bar | pass |
|---|---|---|---|
| G1 accuracy at tau50 | 0.8993 | >= 0.85 and >= T - 0.08 (T = 0.9260) | True |
| G2 false cut-off at tau85 | 0.0847 | <= 0.20 | True |
| G2 END recall at tau85 | 0.8627 | >= 0.80 | True |
| G3 wasm_median, frontend, size, parity | {'wasm_median': True, 'frontend': True, 'size': True, 'parity': True} | WASM <= 5.0 ms, front end <= 0.10 ms/hop, <= 3 MB, parity | True |

## Test (31527 clips, 0 without a decision; 95% bootstrap intervals, 1000 resamples by clip)

| metric | value |
|---|---|
| accuracy@0.5 | 0.8993 [0.8962, 0.9027] |
| precision@0.5 | 0.8776 [0.8727, 0.8828] |
| recall@0.5 | 0.9271 [0.9232, 0.9313] |
| f1@0.5 | 0.9017 [0.8984, 0.9052] |
| fpr_all@0.5 | 0.0644 [0.0617, 0.0671] |
| fnr_all@0.5 | 0.0363 [0.0343, 0.0383] |
| false_cutoff@0.724398 | 0.0847 [0.0800, 0.0888] |
| recall@0.724398 | 0.8627 [0.8577, 0.8680] |
| auc | 0.9600 [0.9581, 0.9621] |
| ece | 0.0160 [0.0138, 0.0191] |

Baselines on the same clips:

| model | accuracy |
|---|---|
| silence timer (always "finished") | 0.4979 |
| s6-ear-nano (v1) | 0.5681 |
| its control | 0.7647 |
| Smart Turn v3.2 CPU, as checked (§8.1) | 0.9260 |
| Smart Turn v3.2 GPU (Pipecat's published) | 0.9371 |
| v1's wrong harness number (right padding), marked as such | 0.8518 |
| s6-ear2-nano | 0.8993 [0.8962, 0.9027] |

Per language:

| language | clips | accuracy |
|---|---|---|
| ara | 947 | 0.8585 [0.8353, 0.8796] |
| ben | 1000 | 0.7950 [0.7690, 0.8200] |
| dan | 779 | 0.8960 [0.8742, 0.9191] |
| deu | 1322 | 0.9493 [0.9365, 0.9614] |
| eng | 7820 | 0.9056 [0.8987, 0.9127] |
| fin | 1010 | 0.9129 [0.8970, 0.9307] |
| fra | 1252 | 0.9297 [0.9153, 0.9441] |
| hin | 1284 | 0.8754 [0.8575, 0.8941] |
| ind | 971 | 0.9279 [0.9114, 0.9434] |
| ita | 782 | 0.9169 [0.8964, 0.9361] |
| jpn | 834 | 0.9341 [0.9161, 0.9508] |
| kor | 889 | 0.9449 [0.9303, 0.9595] |
| mar | 774 | 0.7907 [0.7623, 0.8178] |
| nld | 1398 | 0.9471 [0.9349, 0.9592] |
| nor | 1014 | 0.9004 [0.8807, 0.9181] |
| pol | 974 | 0.9148 [0.8963, 0.9322] |
| por | 1398 | 0.9421 [0.9292, 0.9535] |
| rus | 1468 | 0.9060 [0.8910, 0.9210] |
| spa | 1783 | 0.8716 [0.8559, 0.8861] |
| tur | 966 | 0.9369 [0.9203, 0.9524] |
| ukr | 929 | 0.8988 [0.8794, 0.9193] |
| vie | 1004 | 0.7749 [0.7490, 0.8018] |
| zho | 929 | 0.8654 [0.8428, 0.8891] |

Per dataset:

| dataset | clips | accuracy |
|---|---|---|
| chirp3_1 | 16254 | 0.9192 [0.9150, 0.9232] |
| chirp3_2 | 8428 | 0.8622 [0.8548, 0.8694] |
| chirp3_3_short | 104 | 0.8462 [0.7692, 0.9135] |
| human_5 | 402 | 0.9353 [0.9104, 0.9602] |
| human_convcollector_1 | 90 | 0.8444 [0.7667, 0.9111] |
| liva_1 | 3831 | 0.8925 [0.8815, 0.9019] |
| midcentury_1 | 1044 | 0.9636 [0.9511, 0.9751] |
| mundo_1 | 496 | 0.7843 [0.7500, 0.8206] |
| orpheus_endfiller_1 | 181 | 0.9503 [0.9171, 0.9779] |
| orpheus_grammar_1 | 163 | 0.8466 [0.7914, 0.8957] |
| orpheus_midfiller_1 | 140 | 0.8571 [0.7929, 0.9143] |
| rime_2 | 394 | 0.9188 [0.8909, 0.9442] |

Subset english: accuracy 0.9056 [0.8995, 0.9123], false cut-off at tau85 0.0845 [0.0761, 0.0927], END recall 0.8719 [0.8618, 0.8827]

Subset english_human: accuracy 0.9187 [0.8923, 0.9431], false cut-off at tau85 0.0688 [0.0427, 0.1011], END recall 0.8704 [0.8251, 0.9116]
Speko's oracle point (all, threshold chosen on the evaluated set, reported only): false cut-off 0.0799 at END recall 0.8500, accuracy 0.8852
Speko's oracle point (english_human, threshold chosen on the evaluated set, reported only): false cut-off 0.0580 at END recall 0.8519, accuracy 0.9024

## Teacher-student agreement (§10)

{"val": {"n": 13593, "agreement": 0.9138527183108953}, "test": {"n": 31527, "agreement": 0.918070225520982}}

## eot-bench (§8.4, reported, not gated)

| model | cuts | AUC | false cut-off | eot recall |
|---|---|---|---|---|
| v2 (tau85) | 17934 | 0.7299 | 0.2230 | 0.5253 |
| Smart Turn v3.2 CPU (0.5) | 17934 | 0.7838 | 0.3678 | 0.7998 |
| timer | - | 0.5000 | 1.0000 | 1.0000 |

These numbers are not comparable with LiveKit's FC@300 / FC@600, which come from a policy sweep; this harness cuts each labelled pause at its start and decides eot or hold once.

## Reported (§10)

Pilot: {"lr": 0.0007745, "rate_multiplier": 0.5, "base_rate": 0.001549, "distil": true, "stop": false, "best_val_acc": 0.8714779666004561, "arms": {"lr0.5": {"val_acc_f32": 0.8714779666004561, "lr": 0.0007745, "distil": true}, "lr1": {"val_acc_f32": 0.8656661516957257, "lr": 0.001549, "distil": true}, "lr2": {"val_acc_f32": 0.8003384094754653, "lr": 0.003098, "distil": true}, "hard": {"val_acc_f32": 0.8437431030677555, "lr": 0.0007745, "distil": false}}, "rules": {"choice": "validation accuracy at tau 0.5 on f32 weights (\u00a76.4)", "distil": "distil unless the hard-label arm is higher by more than 0.005", "stop": "the best arm below 0.82 stops v2 (\u00a76.5)"}}

Probe: {"rung": "R4", "rule": "the first rung of R1-R4 whose WASM median is <= 4.0 ms (\u00a74.4)", "table": [{"rung": "R1", "wasm_median_ms": 10.099207999999862, "wasm_p95_ms": 10.21758299999965, "wasm_frontend_ms_per_hop": 0.017361950782997786, "native_median_ms": 5.947708, "native_p95_ms": 20.738666000000002, "native_frontend_ms_per_hop": 0.01335234899328859, "params": 2714754, "file_bytes": 931264, "qualifies": false}, {"rung": "R2", "wasm_median_ms": 6.785542000001442, "wasm_p95_ms": 6.862999999999374, "wasm_frontend_ms_per_hop": 0.016783371364650412, "native_median_ms": 8.025459000000001, "native_p95_ms": 20.891416999999997, "native_frontend_ms_per_hop": 0.02042216554809843, "params": 1829250, "file_bytes": 699008, "qualifies": false}, {"rung": "R3", "wasm_median_ms": 4.885207999999693, "wasm_p95_ms": 4.9398330000003625, "wasm_frontend_ms_per_hop": 0.016752888143174056, "native_median_ms": 5.980459, "native_p95_ms": 19.502333999999998, "native_frontend_ms_per_hop": 0.026225017897091724, "params": 1278658, "file_bytes": 521856, "qualifies": false}, {"rung": "R4", "wasm_median_ms": 2.3902909999997064, "wasm_p95_ms": 2.4231659999968542, "wasm_frontend_ms_per_hop": 0.01672958612975717, "native_median_ms": 2.309167, "native_p95_ms": 7.597625, "native_frontend_ms_per_hop": 0.020578114093959734, "params": 1306178, "file_bytes": 631936, "qualifies": true}], "measured": "2026-11-16T09:53:44+0100", "host": "Apple M5"}

Leakage filter (§3.2): {"rows": 270946, "kept": 270946, "train": 257353, "val": 13593, "dropped_test_id": 0, "dropped_test_pcm": 0, "undecodable": 0, "no_hops": 0}
Sentence-level overlap cannot be checked: spoken_text is empty, as for Smart Turn's own numbers.

Training: {"steps": 305700, "working_minutes": 240.00070010890002, "epochs": 152.06431632815628}

Calibration: {"temperature": 1.494785470484989, "tau50": 0.5, "tau85": 0.7243981397923976}

Operating notes (§10): run the model once when the VAD reports a pause (Silero, or the engine's energy VAD in the demo) with Pipecat's `stop_secs` 0.2; answer "finished" when the probability is at least the threshold (tau50 for accuracy, tau85 for 85% END recall on validation), else keep listening; a fallback timer ends the turn after a longer silence (Pipecat's `SmartTurnParams.stop_secs`); if speech resumes, the turn's audio is kept and the model runs again at the next pause.

**Disclosure (§11).** v2 was designed after v1's test results. Its design study read measurements made on 3,000 clips of the same test set: v1's two models per source, Smart Turn per source, Smart Turn with 8, 4, 3, 2 and 1 s of context, and the trailing silence of 1,500 clips. The 4 s window was chosen from Smart Turn's accuracy on those test clips, and the data change was motivated by per-source test results. The official test set is therefore partly in-sample for v2's design choices, though never for its weights: no test clip is trained on, and no threshold is chosen on test. eot-bench, which the design never read, is the cleaner check, and it is reported (§8.4).
