# s6-route-nano: gate not passed (val, seat `potion-cos`)

**validation: a rehearsal of the gate, not its verdict.**

| condition | value | bar | pass |
|---|---:|---:|:---:|
| G1 accuracy vs potion-cos + 0.03 | 0.7255 | 0.7555 | no |
| G1 vs static-ft | 0.7255 | > 0.7351 | |
| G1 vs pair-lr | 0.7255 | > 0.4843 | |
| G1 vs bm25 (amendment 1) | 0.7255 | > 0.8549 | |
| G2 none answers | 0.1792 | 0.10–0.30 | yes |
| G2 largest name excess | -0.0056 | ≤ 0 | yes |
| G2 answer changes, gold removed | 0.9193 | ≥ 0.80 | yes |
| G3 ECE | 0.0198 | ≤ 0.05 | yes |
| G4 WASM median, tool list cached (µs) | — | ≤ 2000 | no |
| G4 max abs p gap | — | ≤ 1e-4 | |

Paired bootstrap of the accuracy differences (1,000 resamples of the slates, 95%):

| vs | difference | interval |
|---|---:|---|
| potion-cos | 0.0000 | [0.0000, 0.0000] |
| static-ft | -0.0096 | [-0.0219, 0.0017] |
| pair-lr | 0.2413 | [0.2037, 0.2780] |
| bm25 | -0.1294 | [-0.1565, -0.1023] |

## Every system, every source

| system | source | n | accuracy | tool gold | none gold | none P | none R | none F1 | ECE | NLL | mean conf | none answers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-cos | v_tools | 1144 | 0.7255 | 0.7781 | 0.5153 | 0.5756 | 0.5153 | 0.5438 | 0.0198 | 0.7940 | 0.7223 | 0.1792 |
| static-ft | v_tools | 1144 | 0.7351 | 0.7934 | 0.5022 | 0.6216 | 0.5022 | 0.5556 | 0.0378 | 0.7679 | 0.7325 | 0.1617 |
| pair-lr | v_tools | 1144 | 0.4843 | 0.5497 | 0.2227 | 0.4146 | 0.2227 | 0.2898 | 0.0617 | 1.4720 | 0.4568 | 0.1075 |
| bm25 | v_tools | 1144 | 0.8549 | 0.9311 | 0.5502 | 0.7500 | 0.5502 | 0.6348 | 0.0712 | 0.5498 | 0.8260 | 0.1469 |
| potion-cos-4bit | v_tools | 1144 | 0.7229 | 0.7836 | 0.4803 | 0.6077 | 0.4803 | 0.5366 | 0.0297 | 0.8030 | 0.7207 | 0.1582 |
| minilm-cos | v_tools | 1144 | 0.9126 | 0.9279 | 0.8515 | 0.7800 | 0.8515 | 0.8142 | 0.0209 | 0.2807 | 0.9057 | 0.2185 |
| toolrouter | v_tools | 1144 | 0.7832 | 0.9377 | 0.1659 | 0.5758 | 0.1659 | 0.2576 | 0.0860 | 1.0545 | 0.8687 | 0.0577 |
| name-prior | v_tools | 1144 | 0.2002 | 0.0000 | 1.0000 | 0.2002 | 1.0000 | 0.3336 | 0.7998 | 22.1000 | 1.0000 | 1.0000 |
| s6-nano-v1 | v_tools | 1144 | 0.0848 | 0.1060 | 0.0000 | 0.0000 | 0.0000 | — | 0.1115 | 2.2094 | 0.1963 | 0.0070 |

## Reading checks

| system | answer changes, gold removed | names masked: accuracy | swapped: follows description | follows name |
|---|---:|---:|---:|---:|
| potion-cos | 0.9193 | 0.7325 | 0.6700 | 0.0960 |
| static-ft | 0.9156 | 0.7517 | 0.6900 | 0.1040 |
| pair-lr | 0.8417 | 0.4126 | 0.4020 | 0.0940 |
| bm25 | 0.9440 | 0.8444 | 0.9080 | 0.0180 |
| potion-cos-4bit | 0.8935 | 0.7351 | 0.6960 | 0.0940 |
| minilm-cos | 0.9851 | 0.9003 | 0.8500 | 0.0360 |
| toolrouter | 0.9066 | 0.7622 | 0.8240 | 0.0700 |
| name-prior | — | 0.1101 | 0.0000 | 0.0000 |
| s6-nano-v1 | 0.7551 | 0.0900 | 0.1460 | 0.0800 |
