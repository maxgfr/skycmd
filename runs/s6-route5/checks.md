# s6-route v5: the data checks (docs/s6-route5-prereg.md 4.4)

exit 1; C6 fails 0; C1 fails 6; C3 fails 0; C2 fails 8; N fails 9

C1 or C3 fails: the build has a defect and is fixed before anything else (4.4; C1: 4.3).
C2 fails: the surface cell does not make the slates blind. No model is trained, and the table is published (11).
N5 fails: even with the hard distractors anchored on the request, the right tool is the slate's centre in word overlap. No model is trained, and the table is published (11).
N6 fails: the spread of a slate's options over the cells tells whether none is right, without the request. No model is trained, and the table is published, naming the families; there is no remedy (amendment 2).
Another N rule fails: a request-blind cue that the cell does not see remains. No model is trained, and the table is published, naming the rule and the cells (11).

| cell | slates | tool-gold | judged | C6 L5/L6 | C1 diff | best rule (C2, N) | accuracy | tolerance | failing |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| records intents hard train | 26037 | 26037 | yes | 0.0141 | +0.0095 | R12 inverse: the most | 0.4197 | 0.3953 | 3 |
| records intents natural train | 26037 | 26037 | yes | 0.0138 | +0.0149 | R12 inverse: the most | 0.4206 | 0.3951 | 1 |
| records intents removed train | 13019 | 0 | no | 0.0187 |  |  |  |  | 0 |
| records intents renamed train | 26037 | 20830 | yes | 0.0197 | -0.0036 | R12 inverse: the most | 0.2033 | 0.1677 | 1 |
| records intents hard val | 3167 | 3167 | yes | 0.0103 | +0.0113 | R12 inverse: the most | 0.4257 | 0.4097 | 4 |
| records intents natural val | 3167 | 3167 | yes | 0.0143 | +0.0064 | R12 inverse: the most | 0.4204 | 0.4078 | 1 |
| records intents removed val | 1584 | 0 | no | 0.0170 |  |  |  |  | 0 |
| records intents renamed val | 3167 | 2534 | yes | 0.0249 | -0.0162 | R12 inverse: the most | 0.2040 | 0.1768 | 1 |
| records tools hard train | 22020 | 22020 | yes | 0.0000 | +0.0155 | N2 the rarest word (lowest frequency class) | 0.4182 | 0.4018 | 2 |
| records tools natural train | 22020 | 22020 | yes | 0.0000 | +0.0026 | N3 the first name in alphabetical order | 0.3955 | 0.3977 | 0 |
| records tools removed train | 11010 | 0 | no | 0.0000 |  |  |  |  | 0 |
| records tools renamed train | 22020 | 17616 | yes | 0.0000 | +0.0056 | N5 centrality: the highest mean word overlap with the other options | 0.1661 | 0.1681 | 0 |
| records tools hard val | 1126 | 1126 | yes | 0.0000 | +0.0022 | N2 the rarest word (lowest frequency class) | 0.4229 | 0.4242 | 0 |
| records tools natural val | 1126 | 1126 | yes | 0.0000 | -0.0167 | R12 inverse: the most | 0.3993 | 0.4185 | 0 |
| records tools removed val | 563 | 0 | no | 0.0000 |  |  |  |  | 0 |
| records tools renamed val | 1126 | 901 | yes | 0.0009 | +0.0233 | R5 most characters | 0.1803 | 0.1862 | 0 |
| records when2call hard train | 4947 | 3967 | yes | 0.0782 | -0.0215 | N4 the highest share of capitalised words after the first | 0.4453 | 0.4046 | 2 |
| records when2call natural train | 4971 | 3967 | yes | 0.0902 | +0.0013 | N2 the rarest word (lowest frequency class) | 0.3945 | 0.4069 | 0 |
| records when2call renamed train | 3967 | 3174 | yes | 0.0000 | +0.0083 | N2 the rarest word (lowest frequency class) | 0.1738 | 0.1753 | 0 |
| records when2call hard val | 305 | 242 | yes | 0.0794 | +0.0017 | N3 inverse: the last name | 0.4421 | 0.4716 | 0 |
| records when2call natural val | 300 | 242 | yes | 0.0602 | -0.0163 | R7 position 2 | 0.4339 | 0.4541 | 0 |
| records when2call renamed val | 242 | 194 | no | 0.0000 | -0.0610 | N4 inverse: the lowest share | 0.2130 | 0.2131 | 0 |
| val v5_tools.jsonl | 1144 | 915 | yes | 0.0009 |  | N5 inverse: the lowest mean overlap | 0.1730 | 0.1860 | 0 |
| val v5_tools_masked.jsonl | 1144 | 915 | yes | 0.0009 |  | N5 inverse: the lowest mean overlap | 0.1730 | 0.1860 | 0 |

## N6 (amendment 2)

| family | slates | none share | median spread | rule | answered none | accuracy | tolerance | fails |
|---|---:|---:|---:|---|---:|---:|---:|---|
| records intents train renamed | 26037 | 0.2000 | 0.2857 | N6 none when the options spread above the median | 11957 | 0.1853 | 0.2273 | False |
| records intents train renamed | 26037 | 0.2000 | 0.2857 | N6 inverse: none when the options spread below the median | 11306 | 0.2104 | 0.2275 | False |
| records intents train other | 65093 | 0.2000 | 0.5000 | N6 none when the options spread above the median | 24598 | 0.1888 | 0.2251 | False |
| records intents train other | 65093 | 0.2000 | 0.5000 | N6 inverse: none when the options spread below the median | 27185 | 0.2128 | 0.2249 | False |
| records intents val renamed | 3167 | 0.1999 | 0.2857 | N6 none when the options spread above the median | 1480 | 0.1905 | 0.2407 | False |
| records intents val renamed | 3167 | 0.1999 | 0.2857 | N6 inverse: none when the options spread below the median | 1311 | 0.2075 | 0.2420 | False |
| records intents val other | 7918 | 0.2001 | 0.5000 | N6 none when the options spread above the median | 2666 | 0.1947 | 0.2355 | False |
| records intents val other | 7918 | 0.2001 | 0.5000 | N6 inverse: none when the options spread below the median | 3720 | 0.2108 | 0.2332 | False |
| records tools train renamed | 22020 | 0.2000 | 0.0000 | N6 none when the options spread above the median | 6567 | 0.1993 | 0.2299 | False |
| records tools train renamed | 22020 | 0.2000 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records tools train other | 55050 | 0.2000 | 0.0000 | N6 none when the options spread above the median | 13374 | 0.1978 | 0.2269 | False |
| records tools train other | 55050 | 0.2000 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records tools val renamed | 1126 | 0.1998 | 0.0000 | N6 none when the options spread above the median | 123 | 0.2195 | 0.2919 | False |
| records tools val renamed | 1126 | 0.1998 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records tools val other | 2815 | 0.2000 | 0.5000 | N6 none when the options spread above the median | 1236 | 0.1990 | 0.2428 | False |
| records tools val other | 2815 | 0.2000 | 0.5000 | N6 inverse: none when the options spread below the median | 952 | 0.1943 | 0.2459 | False |
| records when2call train renamed | 3967 | 0.1999 | 0.0000 | N6 none when the options spread above the median | 1578 | 0.2060 | 0.2400 | False |
| records when2call train renamed | 3967 | 0.1999 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records when2call train other | 9918 | 0.2000 | 0.0000 | N6 none when the options spread above the median | 3052 | 0.4410 | 0.2345 | True |
| records when2call train other | 9918 | 0.2000 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records when2call val renamed | 242 | 0.1983 | 0.0000 | N6 none when the options spread above the median | 94 | 0.2128 | 0.3006 | False |
| records when2call val renamed | 242 | 0.1983 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| records when2call val other | 605 | 0.2000 | 0.0000 | N6 none when the options spread above the median | 178 | 0.4382 | 0.2800 | True |
| records when2call val other | 605 | 0.2000 | 0.0000 | N6 inverse: none when the options spread below the median | 0 | 0.0000 | 1.0000 | False |
| val v5_tools.jsonl | 1144 | 0.2002 | 0.5714 | N6 none when the options spread above the median | 429 | 0.1702 | 0.2588 | False |
| val v5_tools.jsonl | 1144 | 0.2002 | 0.5714 | N6 inverse: none when the options spread below the median | 500 | 0.2180 | 0.2560 | False |
| val v5_tools_masked.jsonl | 1144 | 0.2002 | 0.2857 | N6 none when the options spread above the median | 500 | 0.1740 | 0.2560 | False |
| val v5_tools_masked.jsonl | 1144 | 0.2002 | 0.2857 | N6 inverse: none when the options spread below the median | 542 | 0.2251 | 0.2545 | False |

## Removed slates against their sources (reported)

| group split | pairs | removed: modal share | source: modal share | difference | SE |
|---|---:|---:|---:|---:|---:|
| intents train | 13019 | 0.6130 | 0.5987 | +0.0143 | 0.0013 |
| intents val | 1584 | 0.6376 | 0.6199 | +0.0177 | 0.0036 |
| tools train | 11010 | 0.8774 | 0.8727 | +0.0047 | 0.0006 |
| tools val | 563 | 0.5423 | 0.5541 | -0.0117 | 0.0065 |

## C3

| cell | slates | none | share | fails |
|---|---:|---:|---:|---|
| intents train renamed | 26037 | 5207 | 0.2000 | False |
| intents train other | 65093 | 13019 | 0.2000 | False |
| intents val renamed | 3167 | 633 | 0.1999 | False |
| intents val other | 7918 | 1584 | 0.2001 | False |
| tools train renamed | 22020 | 4404 | 0.2000 | False |
| tools train other | 55050 | 11010 | 0.2000 | False |
| tools val renamed | 1126 | 225 | 0.1998 | False |
| tools val other | 2815 | 563 | 0.2000 | False |
| when2call train renamed | 3967 | 793 | 0.1999 | False |
| when2call train other | 9918 | 1984 | 0.2000 | False |
| when2call val renamed | 242 | 48 | 0.1983 | False |
| when2call val other | 605 | 121 | 0.2000 | False |
| val v5_tools.jsonl | 1144 | 229 | 0.2002 | False |
| val v5_tools_masked.jsonl | 1144 | 229 | 0.2002 | False |

### records intents hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.4197 | 0.3693 | 0.3953 |
| N4 inverse: the lowest share | 0.4050 | 0.3693 | 0.3953 |
| R5 most characters | 0.4007 | 0.3693 | 0.3953 |
| N5 inverse: the lowest mean overlap | 0.3923 | 0.3693 | 0.3953 |
| R11 the shortest name | 0.3851 | 0.3693 | 0.3953 |

### records intents natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.4206 | 0.3691 | 0.3951 |
| N4 inverse: the lowest share | 0.3846 | 0.3691 | 0.3951 |
| R5 most characters | 0.3797 | 0.3691 | 0.3951 |
| R2 first word a capitalised word in -s | 0.3774 | 0.3691 | 0.3951 |
| R3 the diagnosis's rule (R1, then R2) | 0.3774 | 0.3691 | 0.3951 |

### records intents removed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records intents renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.2033 | 0.1429 | 0.1677 |
| N5 inverse: the lowest mean overlap | 0.1579 | 0.1429 | 0.1677 |
| N4 inverse: the lowest share | 0.1572 | 0.1429 | 0.1677 |
| R2 first word a capitalised word in -s | 0.1546 | 0.1429 | 0.1677 |
| R3 the diagnosis's rule (R1, then R2) | 0.1546 | 0.1429 | 0.1677 |

### records intents hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.4257 | 0.3726 | 0.4097 |
| N4 inverse: the lowest share | 0.4199 | 0.3726 | 0.4097 |
| R5 most characters | 0.4119 | 0.3726 | 0.4097 |
| N5 inverse: the lowest mean overlap | 0.4119 | 0.3726 | 0.4097 |
| N3 inverse: the last name | 0.4039 | 0.3726 | 0.4097 |

### records intents natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.4204 | 0.3706 | 0.4078 |
| N3 inverse: the last name | 0.3947 | 0.3706 | 0.4078 |
| N4 inverse: the lowest share | 0.3911 | 0.3706 | 0.4078 |
| R5 most characters | 0.3884 | 0.3706 | 0.4078 |
| N5 inverse: the lowest mean overlap | 0.3855 | 0.3706 | 0.4078 |

### records intents removed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records intents renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.2040 | 0.1429 | 0.1768 |
| N5 inverse: the lowest mean overlap | 0.1679 | 0.1429 | 0.1768 |
| N4 inverse: the lowest share | 0.1632 | 0.1429 | 0.1768 |
| R5 most characters | 0.1604 | 0.1429 | 0.1768 |
| R2 first word a capitalised word in -s | 0.1602 | 0.1429 | 0.1768 |

### records tools hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N2 the rarest word (lowest frequency class) | 0.4182 | 0.3753 | 0.4018 |
| N3 the first name in alphabetical order | 0.4037 | 0.3753 | 0.4018 |
| R11 inverse: the longest name | 0.3864 | 0.3753 | 0.4018 |
| R5 inverse: fewest characters | 0.3854 | 0.3753 | 0.4018 |
| N4 inverse: the lowest share | 0.3830 | 0.3753 | 0.4018 |

### records tools natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N3 the first name in alphabetical order | 0.3955 | 0.3712 | 0.3977 |
| N5 centrality: the highest mean word overlap with the other options | 0.3848 | 0.3712 | 0.3977 |
| R5 most characters | 0.3813 | 0.3712 | 0.3977 |
| B: (name style, period, first word form), majority | 0.3794 | 0.3712 | 0.3977 |
| R4 most words | 0.3772 | 0.3712 | 0.3977 |

### records tools removed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records tools renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N5 centrality: the highest mean word overlap with the other options | 0.1661 | 0.1429 | 0.1681 |
| B: (name style, period, first word form), majority | 0.1518 | 0.1429 | 0.1681 |
| N2 the rarest word (lowest frequency class) | 0.1509 | 0.1429 | 0.1681 |
| R7 position 5 | 0.1483 | 0.1429 | 0.1681 |
| N4 inverse: the lowest share | 0.1482 | 0.1429 | 0.1681 |

### records tools hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N2 the rarest word (lowest frequency class) | 0.4229 | 0.3753 | 0.4242 |
| N3 the first name in alphabetical order | 0.4112 | 0.3753 | 0.4242 |
| R5 inverse: fewest characters | 0.3939 | 0.3753 | 0.4242 |
| B: (name style, period, first word form), minority | 0.3912 | 0.3753 | 0.4242 |
| R2 inverse | 0.3909 | 0.3753 | 0.4242 |

### records tools natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R12 inverse: the most | 0.3993 | 0.3697 | 0.4185 |
| N3 the first name in alphabetical order | 0.3970 | 0.3697 | 0.4185 |
| R5 most characters | 0.3921 | 0.3697 | 0.4185 |
| R7 position 2 | 0.3845 | 0.3697 | 0.4185 |
| R2 inverse | 0.3800 | 0.3697 | 0.4185 |

### records tools removed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records tools renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 most characters | 0.1803 | 0.1429 | 0.1862 |
| N5 inverse: the lowest mean overlap | 0.1647 | 0.1429 | 0.1862 |
| N3 inverse: the last name | 0.1632 | 0.1429 | 0.1862 |
| N2 the rarest word (lowest frequency class) | 0.1617 | 0.1429 | 0.1862 |
| R7 position 7 | 0.1565 | 0.1429 | 0.1862 |

### records when2call hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N4 the highest share of capitalised words after the first | 0.4453 | 0.3693 | 0.4046 |
| N2 the rarest word (lowest frequency class) | 0.4076 | 0.3693 | 0.4046 |
| N3 inverse: the last name | 0.4026 | 0.3693 | 0.4046 |
| R6 inverse: an option not rewritten | 0.3877 | 0.3693 | 0.4046 |
| R5 inverse: fewest characters | 0.3849 | 0.3693 | 0.4046 |

### records when2call natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N2 the rarest word (lowest frequency class) | 0.3945 | 0.3716 | 0.4069 |
| N4 inverse: the lowest share | 0.3904 | 0.3716 | 0.4069 |
| B: (name style, period, first word form), majority | 0.3813 | 0.3716 | 0.4069 |
| R7 position 1 | 0.3809 | 0.3716 | 0.4069 |
| R10 a digit in the name | 0.3780 | 0.3716 | 0.4069 |

### records when2call renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N2 the rarest word (lowest frequency class) | 0.1738 | 0.1429 | 0.1753 |
| N4 inverse: the lowest share | 0.1691 | 0.1429 | 0.1753 |
| N1 the highest share of digits in the description | 0.1560 | 0.1429 | 0.1753 |
| R7 position 4 | 0.1553 | 0.1429 | 0.1753 |
| R7 position 2 | 0.1525 | 0.1429 | 0.1753 |

### records when2call hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N3 inverse: the last name | 0.4421 | 0.3889 | 0.4716 |
| N4 the highest share of capitalised words after the first | 0.4418 | 0.3889 | 0.4716 |
| R7 position 2 | 0.4132 | 0.3889 | 0.4716 |
| R7 position 3 | 0.4091 | 0.3889 | 0.4716 |
| N2 the rarest word (lowest frequency class) | 0.4070 | 0.3889 | 0.4716 |

### records when2call natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R7 position 2 | 0.4339 | 0.3719 | 0.4541 |
| N4 inverse: the lowest share | 0.4171 | 0.3719 | 0.4541 |
| R6 inverse: an option not rewritten | 0.3951 | 0.3719 | 0.4541 |
| R7 inverse: not position 1 | 0.3940 | 0.3719 | 0.4541 |
| N2 inverse: the rarest word most frequent | 0.3888 | 0.3719 | 0.4541 |

### records when2call renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N4 inverse: the lowest share | 0.2130 | 0.1429 | 0.2131 |
| N3 the first name in alphabetical order | 0.2062 | 0.1429 | 0.2131 |
| N2 inverse: the rarest word most frequent | 0.1804 | 0.1429 | 0.2131 |
| R7 position 3 | 0.1753 | 0.1429 | 0.2131 |
| R7 position 7 | 0.1701 | 0.1429 | 0.2131 |

### val v5_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N5 inverse: the lowest mean overlap | 0.1730 | 0.1429 | 0.1860 |
| R12 inverse: the most | 0.1649 | 0.1429 | 0.1860 |
| N2 inverse: the rarest word most frequent | 0.1633 | 0.1429 | 0.1860 |
| B: (name style, period, first word form), minority | 0.1610 | 0.1429 | 0.1860 |
| R13 inverse | 0.1541 | 0.1429 | 0.1860 |

### val v5_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| N5 inverse: the lowest mean overlap | 0.1730 | 0.1429 | 0.1860 |
| R12 inverse: the most | 0.1649 | 0.1429 | 0.1860 |
| N2 inverse: the rarest word most frequent | 0.1633 | 0.1429 | 0.1860 |
| N3 inverse: the last name | 0.1607 | 0.1429 | 0.1860 |
| B: (name style, period, first word form), minority | 0.1553 | 0.1429 | 0.1860 |
