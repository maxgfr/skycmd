# s6-route v5: the data checks (docs/s6-route5-prereg.md 4.4)

exit 1; C6 fails 0; C1 fails 0; C3 fails 1; C2 fails 18; N fails 2

C1 or C3 fails: the build has a defect and is fixed before anything else (4.4; C1: 4.3).
C2 fails: the surface cell does not make the slates blind. No model is trained, and the table is published (11).
N5 fails: even with the hard distractors anchored on the request, the right tool is the slate's centre in word overlap. No model is trained, and the table is published (11).

| cell | slates | tool-gold | judged | C6 L5/L6 | C1 diff | best rule (C2, N) | accuracy | tolerance | failing |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| val v_tools.jsonl | 1144 | 915 | yes | 0.0000 |  | R4 inverse: fewest words | 0.2550 | 0.1860 | 10 |
| val v_tools_masked.jsonl | 1144 | 915 | yes | 0.0000 |  | R4 inverse: fewest words | 0.2550 | 0.1860 | 10 |
| val v4_tools.jsonl | 1144 | 915 | yes | 0.0000 |  | R11 inverse: the longest name | 0.1774 | 0.1860 | 0 |
| val v4_tools_masked.jsonl | 1144 | 915 | yes | 0.0000 |  | R7 position 2 | 0.1661 | 0.1860 | 0 |
| val r_tools_renamed.model.jsonl | 1126 | 901 | yes | 0.0009 | +0.0233 | R5 most characters | 0.1803 | 0.1862 | 0 |
| val r_tools_natural.model.jsonl | 1126 | 1126 | yes | 0.0000 | -0.0167 | R12 inverse: the most | 0.3993 | 0.4185 | 0 |
| fresh f_tools.jsonl | 945 | 756 | no | 0.0000 |  | R5 inverse: fewest characters | 0.3849 | 0.1883 | 0 |
| fresh f_tools_masked.jsonl | 945 | 756 | no | 0.0000 |  | R5 inverse: fewest characters | 0.3849 | 0.1883 | 0 |

## N6 (amendment 2)

| family | slates | none share | median spread | rule | answered none | accuracy | tolerance | fails |
|---|---:|---:|---:|---|---:|---:|---:|---|
| val v_tools.jsonl | 1144 | 0.2002 | 0.8571 | N6 none when the options spread above the median | 0 | 0.0000 | 1.0000 | False |
| val v_tools.jsonl | 1144 | 0.2002 | 0.8571 | N6 inverse: none when the options spread below the median | 123 | 0.1789 | 0.2923 | False |
| val v_tools_masked.jsonl | 1144 | 0.2002 | 0.7143 | N6 none when the options spread above the median | 379 | 0.2058 | 0.2613 | False |
| val v_tools_masked.jsonl | 1144 | 0.2002 | 0.7143 | N6 inverse: none when the options spread below the median | 137 | 0.1460 | 0.2885 | False |
| val v4_tools.jsonl | 1144 | 0.2002 | 0.8571 | N6 none when the options spread above the median | 0 | 0.0000 | 1.0000 | False |
| val v4_tools.jsonl | 1144 | 0.2002 | 0.8571 | N6 inverse: none when the options spread below the median | 567 | 0.2134 | 0.2538 | False |
| val v4_tools_masked.jsonl | 1144 | 0.2002 | 0.7143 | N6 none when the options spread above the median | 238 | 0.1807 | 0.2720 | False |
| val v4_tools_masked.jsonl | 1144 | 0.2002 | 0.7143 | N6 inverse: none when the options spread below the median | 558 | 0.2151 | 0.2541 | False |
| val r_tools_renamed.model.jsonl | 1126 | 0.1998 | 0.0000 | N6 none when the options spread above the median | 123 | 0.2195 | 0.2919 | False |
| val r_tools_renamed.model.jsonl | 1126 | 0.1998 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| val r_tools_natural.model.jsonl | 1126 | 0.0000 | 0.5000 | N6 none when the options spread above the median | 467 | 0.0000 | 0.0200 | False |
| val r_tools_natural.model.jsonl | 1126 | 0.0000 | 0.5000 | N6 inverse: none when the options spread below the median | 402 | 0.0000 | 0.0200 | False |

## Removed slates against their sources (reported)

| group split | pairs | removed: modal share | source: modal share | difference | SE |
|---|---:|---:|---:|---:|---:|

## C3

| cell | slates | none | share | fails |
|---|---:|---:|---:|---|
| val v_tools.jsonl | 1144 | 229 | 0.2002 | False |
| val v_tools_masked.jsonl | 1144 | 229 | 0.2002 | False |
| val v4_tools.jsonl | 1144 | 229 | 0.2002 | False |
| val v4_tools_masked.jsonl | 1144 | 229 | 0.2002 | False |
| val r_tools_renamed.model.jsonl | 1126 | 225 | 0.1998 | False |
| val r_tools_natural.model.jsonl | 1126 | 0 | 0.0000 | True |

### val v_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R4 inverse: fewest words | 0.2550 | 0.1429 | 0.1860 |
| R3 inverse | 0.2468 | 0.1429 | 0.1860 |
| R1 inverse: no final period | 0.2415 | 0.1429 | 0.1860 |
| R5 inverse: fewest characters | 0.2158 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.2155 | 0.1429 | 0.1860 |

### val v_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R4 inverse: fewest words | 0.2550 | 0.1429 | 0.1860 |
| R3 inverse | 0.2468 | 0.1429 | 0.1860 |
| R1 inverse: no final period | 0.2415 | 0.1429 | 0.1860 |
| R5 inverse: fewest characters | 0.2158 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.2155 | 0.1429 | 0.1860 |

### val v4_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 inverse: the longest name | 0.1774 | 0.1429 | 0.1860 |
| R7 position 2 | 0.1661 | 0.1429 | 0.1860 |
| N3 the first name in alphabetical order | 0.1639 | 0.1429 | 0.1860 |
| R5 most characters | 0.1616 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.1585 | 0.1429 | 0.1860 |

### val v4_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R7 position 2 | 0.1661 | 0.1429 | 0.1860 |
| R5 most characters | 0.1616 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.1585 | 0.1429 | 0.1860 |
| N5 inverse: the lowest mean overlap | 0.1545 | 0.1429 | 0.1860 |
| R4 most words | 0.1543 | 0.1429 | 0.1860 |

### val r_tools_renamed.model.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 most characters | 0.1803 | 0.1429 | 0.1862 |
| N5 inverse: the lowest mean overlap | 0.1647 | 0.1429 | 0.1862 |
| N3 inverse: the last name | 0.1632 | 0.1429 | 0.1862 |
| N2 the rarest word (lowest frequency class) | 0.1617 | 0.1429 | 0.1862 |
| R7 position 7 | 0.1565 | 0.1429 | 0.1862 |

### val r_tools_natural.model.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.3993 | 0.3697 | 0.4185 |
| N3 the first name in alphabetical order | 0.3970 | 0.3697 | 0.4185 |
| R5 most characters | 0.3921 | 0.3697 | 0.4185 |
| R7 position 2 | 0.3845 | 0.3697 | 0.4185 |
| R2 inverse | 0.3800 | 0.3697 | 0.4185 |

### fresh f_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 inverse: fewest characters | 0.3849 | 0.1429 | 0.1883 |
| R4 inverse: fewest words | 0.3633 | 0.1429 | 0.1883 |
| R3 inverse | 0.3216 | 0.1429 | 0.1883 |
| R1 inverse: no final period | 0.3070 | 0.1429 | 0.1883 |
| B: S: source, minority (sources not visible to any system) | 0.2209 | 0.1429 | 0.1883 |

### fresh f_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 inverse: fewest characters | 0.3849 | 0.1429 | 0.1883 |
| R4 inverse: fewest words | 0.3633 | 0.1429 | 0.1883 |
| R3 inverse | 0.3216 | 0.1429 | 0.1883 |
| R1 inverse: no final period | 0.3070 | 0.1429 | 0.1883 |
| R9 (name style, R1), the slate's minority | 0.2495 | 0.1429 | 0.1883 |
