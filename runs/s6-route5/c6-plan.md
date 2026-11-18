# s6-route v5: C6 at plan time (docs/s6-route5-prereg.md 4.4, amendment 1)

exit 0; C6 fails in 0 judged cells

C6 holds at plan time in every judged cell: the false-negative job may run (amendment 1, decision 1).

| cell | slates | tool-gold | drawn | judged | L5 or L6 | L6 | L0 ... L6 (drawn) | fails |
|---|---:|---:|---:|---|---:|---:|---|---|
| records intents hard train | 28759 | 28759 | 24795 | yes | 0.0133 | 0.0065 | 3345 15504 3833 1782 0 169 162 | False |
| records intents hard val | 3511 | 3511 | 3008 | yes | 0.0100 | 0.0076 | 508 1767 502 201 0 7 23 | False |
| records intents natural train | 28759 | 28759 | 24708 | yes | 0.0131 | 0.0059 | 3315 15532 3795 1742 0 178 146 | False |
| records intents natural val | 3511 | 3511 | 3015 | yes | 0.0133 | 0.0096 | 544 1725 520 186 0 11 29 | False |
| records intents removed train | 14380 | 0 | 14380 | yes | 0.0160 | 0.0076 | 1652 8849 2333 1316 0 121 109 | False |
| records intents removed val | 1756 | 0 | 1756 | yes | 0.0171 | 0.0131 | 274 982 334 136 0 7 23 | False |
| records intents renamed train | 28759 | 23007 | 28759 | yes | 0.0186 | 0.0186 | 11372 9401 5163 2287 0 0 536 | False |
| records intents renamed val | 3511 | 2809 | 3511 | yes | 0.0248 | 0.0248 | 1288 1054 783 299 0 0 87 | False |
| records tools hard train | 22376 | 22376 | 19061 | yes | 0.0000 | 0.0000 | 15934 2767 243 107 10 0 0 | False |
| records tools hard val | 1144 | 1144 | 979 | yes | 0.0000 | 0.0000 | 24 485 329 73 68 0 0 | False |
| records tools natural train | 22376 | 22376 | 16891 | yes | 0.0000 | 0.0000 | 14220 2377 192 92 10 0 0 | False |
| records tools natural val | 1144 | 1144 | 978 | yes | 0.0000 | 0.0000 | 37 475 330 75 61 0 0 | False |
| records tools removed train | 11188 | 0 | 11188 | yes | 0.0000 | 0.0000 | 9382 1609 131 61 5 0 0 | False |
| records tools removed val | 572 | 0 | 572 | yes | 0.0000 | 0.0000 | 16 295 184 38 39 0 0 | False |
| records tools renamed train | 22376 | 17901 | 22356 | yes | 0.0000 | 0.0000 | 22259 50 42 5 0 0 0 | False |
| records tools renamed val | 1144 | 915 | 1144 | yes | 0.0009 | 0.0009 | 1046 66 21 10 0 0 1 | False |
| records when2call hard train | 5048 | 4054 | 4272 | yes | 0.0789 | 0.0789 | 3664 259 3 9 0 0 337 | False |
| records when2call hard val | 319 | 252 | 262 | yes | 0.0840 | 0.0840 | 226 14 0 0 0 0 22 | False |
| records when2call natural train | 5087 | 4054 | 3767 | yes | 0.0903 | 0.0903 | 3197 218 4 8 0 0 340 | False |
| records when2call natural val | 311 | 252 | 222 | yes | 0.0586 | 0.0586 | 197 12 0 0 0 0 13 | False |
| records when2call renamed train | 4054 | 3243 | 4016 | yes | 0.0000 | 0.0000 | 4012 1 3 0 0 0 0 | False |
| records when2call renamed val | 252 | 202 | 250 | yes | 0.0000 | 0.0000 | 250 0 0 0 0 0 0 | False |
| val v5_tools.jsonl | 1144 | 915 | 1144 | yes | 0.0009 | 0.0000 | 302 457 256 56 72 1 0 | False |
| val v5_tools_masked.jsonl | 1144 | 915 | 1144 | yes | 0.0009 | 0.0000 | 302 457 256 56 72 1 0 | False |
