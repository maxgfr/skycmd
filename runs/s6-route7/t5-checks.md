# s6-route v5: the data checks (docs/s6-route5-prereg.md 4.4)

exit 0; C6 fails 0; C1 fails 0; C3 fails 0; C2 fails 0; N fails 0

Every judged check holds: C6, C1, C3, C2 and N1 to N6.

| cell | slates | tool-gold | judged | C6 L5/L6 | C1 diff | best rule (C2, N) | accuracy | tolerance | failing |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| fresh t5_tools.jsonl | 2000 | 1600 | no | 0.0000 |  | B: (name style, period, first word form), minority | 0.1929 | 0.1804 | 0 |
| fresh t5_tools_masked.jsonl | 2000 | 1600 | no | 0.0000 |  | N5 inverse: the lowest mean overlap | 0.1673 | 0.1804 | 0 |
| fresh t5_tools_rw.jsonl | 1600 | 1600 | no | 0.0000 |  | B: name style, minority | 0.1888 | 0.1804 | 0 |
| fresh t5_tools_masked_rw.jsonl | 1600 | 1600 | no | 0.0000 |  | N5 centrality: the highest mean word overlap with the other options | 0.1635 | 0.1804 | 0 |

## N6 (amendment 2)

| family | slates | none share | median spread | rule | answered none | accuracy | tolerance | fails |
|---|---:|---:|---:|---|---:|---:|---:|---|
| fresh t5_tools.jsonl | 2000 | 0.2000 | 0.7143 | N6 none when the options spread above the median | 473 | 0.1734 | 0.2568 | False |
| fresh t5_tools.jsonl | 2000 | 0.2000 | 0.7143 | N6 inverse: none when the options spread below the median | 771 | 0.2075 | 0.2488 | False |
| fresh t5_tools_masked.jsonl | 2000 | 0.2000 | 0.5714 | N6 none when the options spread above the median | 556 | 0.2086 | 0.2539 | False |
| fresh t5_tools_masked.jsonl | 2000 | 0.2000 | 0.5714 | N6 inverse: none when the options spread below the median | 878 | 0.2107 | 0.2470 | False |
| fresh t5_tools_rw.jsonl | 1600 | 0.0000 | 0.7143 | N6 none when the options spread above the median | 476 | 0.0000 | 0.0200 | False |
| fresh t5_tools_rw.jsonl | 1600 | 0.0000 | 0.7143 | N6 inverse: none when the options spread below the median | 503 | 0.0000 | 0.0200 | False |
| fresh t5_tools_masked_rw.jsonl | 1600 | 0.0000 | 0.5714 | N6 none when the options spread above the median | 578 | 0.0000 | 0.0200 | False |
| fresh t5_tools_masked_rw.jsonl | 1600 | 0.0000 | 0.5714 | N6 inverse: none when the options spread below the median | 553 | 0.0000 | 0.0200 | False |

## Removed slates against their sources (reported)

| group split | pairs | removed: modal share | source: modal share | difference | SE |
|---|---:|---:|---:|---:|---:|

## C3

| cell | slates | none | share | fails |
|---|---:|---:|---:|---|

### fresh t5_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| B: (name style, period, first word form), minority | 0.1929 | 0.1429 | 0.1804 |
| B: name style, minority | 0.1888 | 0.1429 | 0.1804 |
| B: (name style, period), minority | 0.1869 | 0.1429 | 0.1804 |
| R8 name style (4 classes), the slate's minority | 0.1864 | 0.1429 | 0.1804 |
| R9 (name style, R1), the slate's minority | 0.1845 | 0.1429 | 0.1804 |

### fresh t5_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N5 inverse: the lowest mean overlap | 0.1673 | 0.1429 | 0.1804 |
| R13 inverse | 0.1629 | 0.1429 | 0.1804 |
| N4 inverse: the lowest share | 0.1566 | 0.1429 | 0.1804 |
| B: (name style, period, first word form), minority | 0.1561 | 0.1429 | 0.1804 |
| R12 the fewest schema-like words | 0.1557 | 0.1429 | 0.1804 |

### fresh t5_tools_rw.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| B: name style, minority | 0.1888 | 0.1429 | 0.1804 |
| R8 name style (4 classes), the slate's minority | 0.1864 | 0.1429 | 0.1804 |
| N5 centrality: the highest mean word overlap with the other options | 0.1635 | 0.1429 | 0.1804 |
| N5 inverse: the lowest mean overlap | 0.1602 | 0.1429 | 0.1804 |
| R12 the fewest schema-like words | 0.1595 | 0.1429 | 0.1804 |

### fresh t5_tools_masked_rw.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N5 centrality: the highest mean word overlap with the other options | 0.1635 | 0.1429 | 0.1804 |
| N5 inverse: the lowest mean overlap | 0.1602 | 0.1429 | 0.1804 |
| R12 the fewest schema-like words | 0.1595 | 0.1429 | 0.1804 |
| B: (name style, period, first word form), majority | 0.1567 | 0.1429 | 0.1804 |
| R9 inverse: the majority | 0.1550 | 0.1429 | 0.1804 |
