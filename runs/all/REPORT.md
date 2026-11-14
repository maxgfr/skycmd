# The big run: the gates of the s6 models

| model | gate | detail | file read |
|---|---|---|---|
| s6-ear-nano | **FAIL** | turn end 0.5681 (bar >= 0.8518); keywords (12) 0.8885 (bar >= 0.9000); zero-shot keywords en 0.2100 (bar >= 0.2000); zero-shot keywords fr 0.1330 (bar >= 0.2000) | `runs/s6-ear-nano/eval/gate.json` |
| s6-secrets-pico | **PASS** | secret 0.9990 (bar > 0.977); service 0.9795 (bar > 0.787) | `runs/s6-secrets-pico/report.md` |
| s6-nano | **PASS** | PASS: publish at full depth; gate at τ = 1: holds; gate at CRC: fails (CRC thresholds not kept) | `runs/s6-nano/eval/results.md` |
| s6-micro | **FAIL** | FAIL: not published; gate at τ = 1: fails (ECE 0.0821 > 0.08); gate at CRC: fails (ECE 0.0804 > 0.08) | `runs/s6-micro/eval/results.md` |
| s6-cast-pico | **FAIL** | nGMASE 0.9471 (bar <= 0.838); nWQL 0.6646 (bar <= 0.587) | `data/gift-eval-out/s6-cast-pico/summary.json` |

A failed gate means the model is not published; its result stays in this report.
