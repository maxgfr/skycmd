# Amendment 2: the write-rule sweep with C1 read contrastively

The setting of amendment 1's sweep (`sweep.md`, kept on record), over 20 draws of the word vectors (seeds 0 to 19). Each cell: the draws whose four colours, played in turn, were all matched to their own colour after removing their mean; R when at least 90% of the draws are (18 of 20), W otherwise.

## Value form `frame`

| rule | N | d = 1 | d = 4 | d = 8 | d = 16 | d = 32 |
|---|---:|---|---|---|---|---|
| average | 1 | R 20/20 | R 20/20 | R 20/20 | R 20/20 | R 18/20 |
| average | 2 | R 20/20 | R 20/20 | R 20/20 | R 19/20 | R 18/20 |
| average | 4 | R 20/20 | R 20/20 | R 20/20 | R 18/20 | R 18/20 |
| average | 8 | R 20/20 | R 20/20 | R 19/20 | R 18/20 | R 18/20 |
| delta | 1 | W 13/20 | W 12/20 | W 11/20 | W 8/20 | W 4/20 |
| delta | 2 | W 13/20 | W 12/20 | W 14/20 | W 11/20 | W 5/20 |
| delta | 4 | W 13/20 | W 12/20 | W 14/20 | W 15/20 | W 5/20 |
| delta | 8 | W 13/20 | W 12/20 | W 15/20 | W 12/20 | W 4/20 |
| exact | 1 | W 13/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| exact | 2 | W 13/20 | W 0/20 | W 7/20 | W 2/20 | W 1/20 |
| exact | 4 | W 13/20 | W 0/20 | W 14/20 | W 2/20 | W 2/20 |
| exact | 8 | W 13/20 | W 0/20 | W 15/20 | W 2/20 | W 2/20 |
| centred | 1 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 2 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 4 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 8 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |

## Value form `token`

| rule | N | d = 1 | d = 4 | d = 8 | d = 16 | d = 32 |
|---|---:|---|---|---|---|---|
| average | 1 | R 20/20 | R 20/20 | W 17/20 | W 13/20 | W 10/20 |
| average | 2 | R 20/20 | R 20/20 | W 16/20 | W 10/20 | W 9/20 |
| average | 4 | R 20/20 | R 20/20 | W 13/20 | W 9/20 | W 5/20 |
| average | 8 | R 20/20 | R 20/20 | W 11/20 | W 6/20 | W 3/20 |
| delta | 1 | W 7/20 | W 5/20 | W 3/20 | W 3/20 | W 0/20 |
| delta | 2 | W 7/20 | W 5/20 | W 1/20 | W 2/20 | W 0/20 |
| delta | 4 | W 7/20 | W 5/20 | W 1/20 | W 1/20 | W 1/20 |
| delta | 8 | W 7/20 | W 5/20 | W 1/20 | W 1/20 | W 0/20 |
| exact | 1 | W 2/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| exact | 2 | W 2/20 | W 0/20 | W 0/20 | W 1/20 | W 0/20 |
| exact | 4 | W 2/20 | W 0/20 | W 0/20 | W 1/20 | W 0/20 |
| exact | 8 | W 2/20 | W 0/20 | W 0/20 | W 1/20 | W 0/20 |
| centred | 1 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 2 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 4 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |
| centred | 8 | W 0/20 | W 0/20 | W 0/20 | W 0/20 | W 0/20 |

No rule is right at d = 1, 4 and 8 with at most 4 passes: v2 stops here (amendments 1 and 2).
