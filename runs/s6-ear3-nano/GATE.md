# s6-ear3-nano: gate (docs/s6-ear3-prereg.md §9)

**Verdict: FAIL.** E1 fails on its lower bound: no improvement on real human-to-agent speech is shown. Test (c) separates the causes: if (c) improves over v2, the human-to-human signal does not transfer to agents; if not, the model did not learn the signal.

Keep v2 (or take Smart Turn's 8.7 MB file when real conversations matter most): v3 does not pass; on eot-bench at the policy's first question (0.2 s into each pause) its AUC is 0.7197 against v2's 0.7448 (change -0.0252, 95% interval [-0.0336, -0.0167]) and Smart Turn's 0.7873; on Smart Turn's test clips its accuracy is 0.8978 against v2's 0.8993.

| term | value | bar | pass |
|---|---|---|---|
| A1 accuracy at tau50 (Smart Turn's test) | 0.8978 | >= 0.8893 | True |
| A2 false cut-off at tau85 | 0.0881 | <= 0.20 | True |
| A2 END recall at tau85 | 0.8597 | >= 0.80 | True |
| E1 AUC(v3) - AUC(v2), eot-bench pooled | -0.0252 [-0.0336, -0.0167] | >= +0.02 and lower bound > 0 | False |
| E2 recall at a false cut-off of 0.10, v3 - v2 | -0.0563 [-0.0832, -0.0315] | lower bound > 0 | False |
| E3 AUC(v3) - AUC(v2), the 13 other languages | -0.0209 [-0.0308, -0.0111] | lower bound > -0.02 | False |
| G3 wasm_median, frontend, size, parity, parity_conversations, same_tensor_list_as_v2 | {"wasm_median": true, "frontend": true, "size": true, "parity": true, "parity_conversations": true, "same_tensor_list_as_v2": true} | WASM <= 5.0 ms, front end <= 0.10 ms/hop, <= 3 MB, parity, v2's tensor list | True |

## eot-bench at start + 0.2 s (16230 cuts, 7342 turns; paired bootstrap by turn, 2000 resamples, seed 20261117)

| cuts | model | AUC | recall@FC0.10 | recall@FC0.05 | recall@FC0.20 |
|---|---|---|---|---|---|
| all | v3 | 0.7197 | 0.2754 | 0.1499 | 0.4735 |
| all | v2 | 0.7448 | 0.3317 | 0.1840 | 0.5094 |
| all | smart_turn | 0.7873 | 0.3854 | 0.2284 | 0.5818 |
| english | v3 | 0.7375 | 0.3119 | 0.1709 | 0.5155 |
| english | v2 | 0.7737 | 0.3469 | 0.2443 | 0.5636 |
| english | smart_turn | 0.8172 | 0.4358 | 0.3073 | 0.6474 |
| non_english | v3 | 0.7179 | 0.2681 | 0.1369 | 0.4691 |
| non_english | v2 | 0.7388 | 0.3144 | 0.1771 | 0.5067 |
| non_english | smart_turn | 0.7852 | 0.3825 | 0.2406 | 0.5892 |

Recall at a false cut-off is a ranking measure (the ROC point), not a deployable threshold.

| language | cuts | v3 | v2 | Smart Turn |
|---|---|---|---|---|
| ar | 963 | 0.7219 | 0.7251 | 0.7806 |
| de | 977 | 0.6986 | 0.7128 | 0.8000 |
| en | 3902 | 0.7375 | 0.7737 | 0.8172 |
| es | 1082 | 0.7495 | 0.7724 | 0.8005 |
| fr | 958 | 0.7541 | 0.7497 | 0.8162 |
| hi | 952 | 0.6860 | 0.6981 | 0.7217 |
| id | 1145 | 0.7088 | 0.7707 | 0.7769 |
| it | 833 | 0.6813 | 0.7146 | 0.7741 |
| ja | 1050 | 0.7670 | 0.8119 | 0.8299 |
| ko | 759 | 0.7686 | 0.7632 | 0.8474 |
| nl | 847 | 0.7328 | 0.7214 | 0.7829 |
| pt | 1047 | 0.7203 | 0.7544 | 0.8006 |
| tr | 763 | 0.6560 | 0.6973 | 0.7074 |
| zh | 952 | 0.6946 | 0.6692 | 0.7493 |

Policy reading per ask (AUC): {"200": {"cuts": 16230, "v3_auc": 0.7196588685091984, "v2_auc": 0.744825755766614}, "400": {"cuts": 11987, "v3_auc": 0.7100359169094135, "v2_auc": 0.7368992961762525}, "600": {"cuts": 8905, "v3_auc": 0.7103403868952703, "v2_auc": 0.7383266429878987}, "800": {"cuts": 7082, "v3_auc": 0.7115479866111407, "v2_auc": 0.7398080522824899}, "1000": {"cuts": 5718, "v3_auc": 0.7107193780377382, "v2_auc": 0.7361088850264875}, "1200": {"cuts": 4555, "v3_auc": 0.7175838563509125, "v2_auc": 0.7269896293543416}, "1400": {"cuts": 3669, "v3_auc": 0.7202590073066264, "v2_auc": 0.7292073570168808}, "1600": {"cuts": 2634, "v3_auc": 0.7163773896514654, "v2_auc": 0.7422896249785769}, "1800": {"cuts": 2195, "v3_auc": 0.7329001913265306, "v2_auc": 0.7444445551658163}, "2000": {"cuts": 1904, "v3_auc": 0.7548503325942351, "v2_auc": 0.7508314855875832}, "2200": {"cuts": 1714, "v3_auc": 0.7690425181278839, "v2_auc": 0.7954185893210284}, "2400": {"cuts": 1620, "v3_auc": 0.7920307259099756, "v2_auc": 0.7773096194459398}}

Policy simulated per span: {"v3@tau85": {"tau": 0.6655501782133095, "eot_spans": 6798, "eot_answered_before_timer": 0.8231832892027067, "eot_answer_ms_median": 400, "hold_spans": 9432, "hold_with_a_false_cutoff": 0.2589058524173028}, "v3@tau_c": {"tau": 0.6948428153991699, "eot_spans": 6798, "eot_answered_before_timer": 0.8028832009414534, "eot_answer_ms_median": 400, "hold_spans": 9432, "hold_with_a_false_cutoff": 0.2424724342663274}, "v2@tau85": {"tau": 0.7243981397923976, "eot_spans": 6798, "eot_answered_before_timer": 0.8609885260370698, "eot_answer_ms_median": 200, "hold_spans": 9432, "hold_with_a_false_cutoff": 0.37372773536895676}}

tau_c (conversation validation, false cut-off <= 0.10 at d = 0.2 s, reported, not stored): 0.6948428153991699

## Test (c): 30000 conversation test records (2061 speaker-unseen), reported

{"all": {"records": 30000, "v3": {"auc": 0.6878163410666565, "recall@fc0.1": 0.25434694702790134}, "v2": {"auc": 0.6030752499721959, "recall@fc0.1": 0.18075212292761828}, "smart_turn_recorded": {"auc": 0.6430074954456728, "recall@fc0.1": 0.20501415285078853, "agreement_with_labels@0.5": 0.7013}, "smart_turn_v3_agreement": 0.7033333333333334}, "speaker_unseen": {"records": 2061, "v3": {"auc": 0.6663204867558661, "recall@fc0.1": 0.18309859154929578}, "v2": {"auc": 0.6137366145806574, "recall@fc0.1": 0.22535211267605634}, "smart_turn_recorded": {"auc": 0.5988961386872564, "recall@fc0.1": 0.19014084507042253, "agreement_with_labels@0.5": 0.6889859291606016}, "smart_turn_v3_agreement": 0.7045123726346434}}

## Test (a): Smart Turn's test (31527 clips)

{"auc": {"value": 0.9586223285474864, "lo": 0.9566056887232373, "hi": 0.9606384964740196}, "ece": {"value": 0.006751277765742015, "lo": 0.005260629186320517, "hi": 0.010211426088217438}}

Extended by 0.2 and 0.6 s of their own background (reported): {"v3@200": {"n": 31527, "accuracy": 0.8958987534494243, "auc": 0.9584812126184146}, "v3@600": {"n": 31527, "accuracy": 0.8934564024486947, "auc": 0.9572397204319791}, "v2@200": {"n": 31527, "accuracy": 0.8804833951850795, "auc": 0.9473457451979878}, "v2@600": {"n": 31527, "accuracy": 0.851175183176325, "auc": 0.9219775541602855}}

## Reported (§10)

Pilot: {"arm": "P2", "start": "v2", "lr": 0.0003873, "minutes": 120, "stop": false, "v2_metric": 0.6175177069894198, "v2_metric_records": 20000, "arms": {"P1": {"start": "v2", "lr": 0.0001936, "rate_multiplier": 0.25, "st_val_acc_f32": 0.895166629883028, "conv_val_auc_d200": 0.7412835636845744, "eligible": true}, "P2": {"start": "v2", "lr": 0.0003873, "rate_multiplier": 0.5, "st_val_acc_f32": 0.8948723607739277, "conv_val_auc_d200": 0.7428866016123399, "eligible": true}, "P3": {"start": "v2", "lr": 0.0007745, "rate_multiplier": 1.0, "st_val_acc_f32": 0.8939159861693519, "conv_val_auc_d200": 0.731434355242243, "eligible": true}, "P4": {"start": "scratch", "lr": 0.0007745, "rate_multiplier": 1.0, "st_val_acc_f32": 0.8291032148900169, "conv_val_auc_d200": 0.7337842942983613, "eligible": false}}, "rules": {"metric": "AUC on the conversation validation records cut at d = 0.2 s (\u00a76.4)", "eligible": "Smart Turn validation accuracy at tau 0.5, d = 0, f32 >= 0.88", "choice": "the eligible from-v2 arm with the highest metric; P4 instead if eligible and higher by more than 0.005 (amendment 1: or the only eligible arm)", "stop": "no eligible arm, or the chosen arm below v2's file on the same cuts + 0.01 (\u00a76.5)"}}

Label cross-check (the instant label of amendment 2): {"agreement": {"value": 0.9509199451947543, "wilson95": [0.948782837974909, 0.9529722986657312], "agree": 38866, "labelled": 40872, "bar": 0.85, "pass": true}, "drop_share": {"value": 0.1015958148327252, "records": 4622, "drops": {"backchannel": 1545, "no_follow": 774, "overlap": 2255, "tie": 48}, "bar": 0.3, "pass": true}, "agreement_by_class": {"hold": 0.9502691617533966, "eot": 0.9645542427497314}, "pass": true}

Bleed probe: {"max_auc": 0.4396195319106588, "pass": true}

Data: {"sessions": 1125, "absent_from_splits": 0, "silent_channel": 0, "bleed": 17, "kept_sessions": 1108, "candidates": 674669, "in_dropped_sessions": 8931, "backchannel": 171926, "overlap": 15714, "no_follow": 19859, "tie": 100, "redaction": 1769, "eligible": {"test": [44009, 2473], "train": [343858, 20536], "val": [43104, 2390]}, "written": {"test": 43947, "train": 266829, "val": 37734}, "decoded_seconds": 1040448.4316249996}

Training: {"steps": 155300, "working_minutes": 120.00051838475, "epochs": 77.25519422738418}

Calibration: {"temperature": 1.2521543075546715, "tau50": 0.5, "tau85": 0.6655501782133095}

**Disclosure (§11).** v3 was designed after v2's eot-bench reading: the overall, per-language and per-cohort numbers of runs/s6-ear2-nano/GATE.json were seen, and that reading motivated making eot-bench the headline, the no-harm bar outside English and the expected per-language effects. eot-bench has never been used for training or for any choice of v2's or v3's training: no eot-bench file was opened for v3's design, and the cut point (0.2 s) comes from the demo's policy and LiveKit's card, not from a reading. The variable trailing silence was motivated by the microphone demo (turn.ts), through the code. v3's study measured the trailing silence of 3,160 of v2's training clips. Smart Turn's test is still partly in-sample for the design, as v2's disclosure says; v3 adds no reading of it before the gate. The label cross-check and the caps were amended after a measurement on five training sessions, before any validation or test session was read (amendment 2: Silero's label read at each energy record's instant; every end-of-turn record kept). **What it means:** the training corpus is human-to-human and eot-bench is human-to-agent (the labels of the training conversations are the listener's behaviour, not an agent's need); the training conversations are English and eot-bench has 14 languages; the conversation test is session-disjoint, not speaker-disjoint.
