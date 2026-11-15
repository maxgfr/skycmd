# Amendment 1: the write-rule sweep on C1's synthetic states

d = 96, one head, Wqk = Wv = I, word vectors drawn once with seed 0 (mean cosine between words 0.401); sleep every 8 frames with N passes (the wake pass included). Each cell: the colour decoded at the ask's door (R = red, right; G, B, Y wrong) and the margin, red's projection minus the best wrong one's. `average` is v1's rule, for reference.

## Value form `frame`

| rule | N | d = 1 | d = 4 | d = 8 | d = 16 | d = 32 |
|---|---:|---|---|---|---|---|
| average | 1 | R +0.058 | Y -0.193 | Y -0.286 | Y -0.257 | Y -0.311 |
| average | 2 | R +0.058 | Y -0.193 | Y -0.229 | Y -0.182 | Y -0.254 |
| average | 4 | R +0.058 | Y -0.193 | Y -0.107 | Y -0.067 | Y -0.186 |
| average | 8 | R +0.058 | Y -0.193 | B -0.085 | B -0.030 | B -0.181 |
| delta | 1 | B -0.206 | B -0.348 | B -0.461 | B -0.184 | B -0.629 |
| delta | 2 | B -0.206 | B -0.348 | B -0.515 | B -0.318 | B -0.704 |
| delta | 4 | B -0.206 | B -0.348 | B -0.572 | B -0.384 | B -0.732 |
| delta | 8 | B -0.206 | B -0.348 | B -0.607 | B -0.379 | B -0.731 |
| exact | 1 | B -0.613 | B -0.613 | B -0.717 | B -0.626 | B -0.748 |
| exact | 2 | B -0.613 | B -0.613 | B -0.654 | B -0.576 | B -0.687 |
| exact | 4 | B -0.613 | B -0.613 | B -0.656 | B -0.579 | B -0.688 |
| exact | 8 | B -0.613 | B -0.613 | B -0.658 | B -0.581 | B -0.690 |
| centred | 1 | B -0.040 | B -0.033 | B -0.031 | B -0.029 | B -0.027 |
| centred | 2 | B -0.040 | B -0.033 | B -0.064 | B -0.060 | B -0.059 |
| centred | 4 | B -0.040 | B -0.033 | B -0.122 | B -0.111 | B -0.101 |
| centred | 8 | B -0.040 | B -0.033 | B -0.209 | B -0.172 | B -0.131 |

## Value form `token`

| rule | N | d = 1 | d = 4 | d = 8 | d = 16 | d = 32 |
|---|---:|---|---|---|---|---|
| average | 1 | R +0.025 | Y -0.145 | Y -0.206 | Y -0.161 | Y -0.188 |
| average | 2 | R +0.025 | Y -0.145 | Y -0.127 | Y -0.060 | B -0.153 |
| average | 4 | R +0.025 | Y -0.145 | B -0.127 | B -0.074 | B -0.174 |
| average | 8 | R +0.025 | Y -0.145 | B -0.167 | B -0.100 | B -0.208 |
| delta | 1 | B -0.293 | B -0.476 | B -0.580 | B -0.341 | B -0.849 |
| delta | 2 | B -0.293 | B -0.476 | B -0.719 | B -0.595 | B -1.009 |
| delta | 4 | B -0.293 | B -0.476 | B -0.847 | B -0.725 | B -1.070 |
| delta | 8 | B -0.293 | B -0.476 | B -0.915 | B -0.713 | B -1.042 |
| exact | 1 | B -0.896 | B -0.809 | B -0.936 | B -0.946 | B -0.971 |
| exact | 2 | B -0.896 | B -0.809 | B -0.857 | B -0.886 | B -0.895 |
| exact | 4 | B -0.896 | B -0.809 | B -0.866 | B -0.896 | B -0.903 |
| exact | 8 | B -0.896 | B -0.809 | B -0.869 | B -0.900 | B -0.906 |
| centred | 1 | B -0.063 | B -0.050 | B -0.047 | B -0.046 | B -0.039 |
| centred | 2 | B -0.063 | B -0.050 | B -0.099 | B -0.095 | B -0.090 |
| centred | 4 | B -0.063 | B -0.050 | B -0.191 | B -0.175 | B -0.157 |
| centred | 8 | B -0.063 | B -0.050 | B -0.330 | B -0.273 | B -0.205 |

No rule decodes red at d = 1, 4 and 8 with at most 4 passes: v2 stops here (amendment 1).
