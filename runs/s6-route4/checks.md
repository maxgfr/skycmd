# s6-route v4: the data checks (docs/s6-route4-prereg.md 4.4)

exit 2; C1 fails 0; C3 fails 0; C2 fails 17; remedy open: False

| cell | slates | tool-gold | judged | C1 diff | best rule | accuracy | tolerance | failing rules |
|---|---:|---:|---|---:|---|---:|---:|---:|
| records intents hard train | 25443 | 25443 | yes | -0.0061 | R11 the shortest name | 0.4219 | 0.3981 | 1 |
| records intents natural train | 25443 | 25443 | yes | +0.0016 | R11 the shortest name | 0.3929 | 0.3951 | 0 |
| records intents removed train | 12722 | 0 | no |  |  |  |  | 0 |
| records intents renamed train | 25443 | 20354 | yes | +0.0021 | R2 first word a capitalised word in -s | 0.1556 | 0.1678 | 0 |
| records intents hard val | 3078 | 3078 | yes | +0.0191 | R11 the shortest name | 0.4453 | 0.4130 | 1 |
| records intents natural val | 3078 | 3078 | yes | +0.0096 | R11 the shortest name | 0.3984 | 0.4063 | 0 |
| records intents removed val | 1539 | 0 | no |  |  |  |  | 0 |
| records intents renamed val | 3078 | 2462 | yes | -0.0169 | R7 position 5 | 0.1600 | 0.1770 | 0 |
| records tools hard train | 21759 | 21759 | yes | -0.0001 | R3 inverse | 0.4220 | 0.4035 | 7 |
| records tools natural train | 21759 | 21759 | yes | +0.0003 | R11 inverse: the longest name | 0.4015 | 0.3974 | 1 |
| records tools removed train | 10880 | 0 | no |  |  |  |  | 0 |
| records tools renamed train | 21759 | 17407 | yes | +0.0014 | R5 most characters | 0.1531 | 0.1682 | 0 |
| records tools hard val | 1116 | 1116 | yes | -0.0275 | R5 inverse: fewest characters | 0.4809 | 0.4253 | 7 |
| records tools natural val | 1116 | 1116 | yes | -0.0062 | R11 inverse: the longest name | 0.4137 | 0.4199 | 0 |
| records tools removed val | 558 | 0 | no |  |  |  |  | 0 |
| records tools renamed val | 1116 | 893 | yes | +0.0181 | R5 most characters | 0.1706 | 0.1863 | 0 |
| records when2call hard train | 4868 | 3899 | yes | +0.0126 | R4 most words | 0.4023 | 0.4068 | 0 |
| records when2call natural train | 4880 | 3899 | yes | -0.0023 | R11 inverse: the longest name | 0.4004 | 0.4066 | 0 |
| records when2call renamed train | 3899 | 3119 | yes | +0.0211 | R2 inverse | 0.1584 | 0.1754 | 0 |
| records when2call hard val | 301 | 239 | yes | +0.0108 | R11 the shortest name | 0.4351 | 0.4742 | 0 |
| records when2call natural val | 297 | 239 | yes | +0.0015 | R5 inverse: fewest characters | 0.4142 | 0.4557 | 0 |
| records when2call renamed val | 239 | 191 | no | -0.0576 | B: (name style, period, first word form), minority | 0.1770 | 0.2135 | 0 |
| val v4_tools.jsonl | 1144 | 915 | yes |  | R11 inverse: the longest name | 0.1774 | 0.1860 | 0 |
| val v4_tools_masked.jsonl | 1144 | 915 | yes |  | R7 position 2 | 0.1661 | 0.1860 | 0 |

## C3

| cell | slates | none | share | fails |
|---|---:|---:|---:|---|
| intents train renamed | 25443 | 5089 | 0.2000 | False |
| intents train other | 63608 | 12722 | 0.2000 | False |
| intents val renamed | 3078 | 616 | 0.2001 | False |
| intents val other | 7695 | 1539 | 0.2000 | False |
| tools train renamed | 21759 | 4352 | 0.2000 | False |
| tools train other | 54398 | 10880 | 0.2000 | False |
| tools val renamed | 1116 | 223 | 0.1998 | False |
| tools val other | 2790 | 558 | 0.2000 | False |
| when2call train renamed | 3899 | 780 | 0.2001 | False |
| when2call train other | 9748 | 1950 | 0.2000 | False |
| when2call val renamed | 239 | 48 | 0.2008 | False |
| when2call val other | 598 | 120 | 0.2007 | False |
| val v4_tools.jsonl | 1144 | 229 | 0.2002 | False |
| val v4_tools_masked.jsonl | 1144 | 229 | 0.2002 | False |

### records intents hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 the shortest name | 0.4219 | 0.3721 | 0.3981 |
| R12 the fewest schema-like words | 0.3930 | 0.3721 | 0.3981 |
| R2 first word a capitalised word in -s | 0.3751 | 0.3721 | 0.3981 |
| R3 the diagnosis's rule (R1, then R2) | 0.3751 | 0.3721 | 0.3981 |
| R13 inverse | 0.3751 | 0.3721 | 0.3981 |

### records intents natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 the shortest name | 0.3929 | 0.3690 | 0.3951 |
| R2 first word a capitalised word in -s | 0.3826 | 0.3690 | 0.3951 |
| R3 the diagnosis's rule (R1, then R2) | 0.3826 | 0.3690 | 0.3951 |
| R13 inverse | 0.3826 | 0.3690 | 0.3951 |
| B: (name style, period, first word form), minority | 0.3757 | 0.3690 | 0.3951 |

### records intents removed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records intents renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R2 first word a capitalised word in -s | 0.1556 | 0.1429 | 0.1678 |
| R3 the diagnosis's rule (R1, then R2) | 0.1556 | 0.1429 | 0.1678 |
| R13 inverse | 0.1556 | 0.1429 | 0.1678 |
| B: (name style, period, first word form), minority | 0.1524 | 0.1429 | 0.1678 |
| R7 position 7 | 0.1459 | 0.1429 | 0.1678 |

### records intents hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 the shortest name | 0.4453 | 0.3756 | 0.4130 |
| R12 the fewest schema-like words | 0.4007 | 0.3756 | 0.4130 |
| R2 first word a capitalised word in -s | 0.3950 | 0.3756 | 0.4130 |
| R3 the diagnosis's rule (R1, then R2) | 0.3950 | 0.3756 | 0.4130 |
| R13 inverse | 0.3950 | 0.3756 | 0.4130 |

### records intents natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 the shortest name | 0.3984 | 0.3689 | 0.4063 |
| R2 first word a capitalised word in -s | 0.3865 | 0.3689 | 0.4063 |
| R3 the diagnosis's rule (R1, then R2) | 0.3865 | 0.3689 | 0.4063 |
| R13 inverse | 0.3865 | 0.3689 | 0.4063 |
| B: (name style, period, first word form), minority | 0.3791 | 0.3689 | 0.4063 |

### records intents removed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records intents renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R7 position 5 | 0.1600 | 0.1429 | 0.1770 |
| R7 position 7 | 0.1580 | 0.1429 | 0.1770 |
| R2 first word a capitalised word in -s | 0.1572 | 0.1429 | 0.1770 |
| R3 the diagnosis's rule (R1, then R2) | 0.1572 | 0.1429 | 0.1770 |
| R13 inverse | 0.1572 | 0.1429 | 0.1770 |

### records tools hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R3 inverse | 0.4220 | 0.3769 | 0.4035 |
| R5 inverse: fewest characters | 0.4151 | 0.3769 | 0.4035 |
| R1 inverse: no final period | 0.4150 | 0.3769 | 0.4035 |
| R4 inverse: fewest words | 0.4132 | 0.3769 | 0.4035 |
| R12 the fewest schema-like words | 0.4127 | 0.3769 | 0.4035 |

### records tools natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 inverse: the longest name | 0.4015 | 0.3709 | 0.3974 |
| R5 most characters | 0.3818 | 0.3709 | 0.3974 |
| R13 an imperative first word | 0.3781 | 0.3709 | 0.3974 |
| R2 inverse | 0.3778 | 0.3709 | 0.3974 |
| B: S: source, minority (sources not visible to any system) | 0.3764 | 0.3709 | 0.3974 |

### records tools removed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records tools renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 most characters | 0.1531 | 0.1429 | 0.1682 |
| B: S: source, minority (sources not visible to any system) | 0.1497 | 0.1429 | 0.1682 |
| R4 most words | 0.1490 | 0.1429 | 0.1682 |
| R7 position 2 | 0.1460 | 0.1429 | 0.1682 |
| R7 position 1 | 0.1460 | 0.1429 | 0.1682 |

### records tools hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 inverse: fewest characters | 0.4809 | 0.3763 | 0.4253 |
| R4 inverse: fewest words | 0.4802 | 0.3763 | 0.4253 |
| R3 inverse | 0.4634 | 0.3763 | 0.4253 |
| R1 inverse: no final period | 0.4544 | 0.3763 | 0.4253 |
| R12 the fewest schema-like words | 0.4489 | 0.3763 | 0.4253 |

### records tools natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 inverse: the longest name | 0.4137 | 0.3710 | 0.4199 |
| R5 most characters | 0.3947 | 0.3710 | 0.4199 |
| R2 inverse | 0.3824 | 0.3710 | 0.4199 |
| R13 an imperative first word | 0.3818 | 0.3710 | 0.4199 |
| B: S: source, minority (sources not visible to any system) | 0.3808 | 0.3710 | 0.4199 |

### records tools removed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|

### records tools renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 most characters | 0.1706 | 0.1429 | 0.1863 |
| R12 inverse: the most | 0.1605 | 0.1429 | 0.1863 |
| B: S: source, minority (sources not visible to any system) | 0.1574 | 0.1429 | 0.1863 |
| R7 position 3 | 0.1545 | 0.1429 | 0.1863 |
| B: (name style, period, first word form), minority | 0.1538 | 0.1429 | 0.1863 |

### records when2call hard train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R4 most words | 0.4023 | 0.3713 | 0.4068 |
| R5 most characters | 0.3996 | 0.3713 | 0.4068 |
| B: S: source, minority (sources not visible to any system) | 0.3967 | 0.3713 | 0.4068 |
| R3 the diagnosis's rule (R1, then R2) | 0.3963 | 0.3713 | 0.4068 |
| R1 ends with a period | 0.3925 | 0.3713 | 0.4068 |

### records when2call natural train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 inverse: the longest name | 0.4004 | 0.3711 | 0.4066 |
| R13 an imperative first word | 0.3823 | 0.3711 | 0.4066 |
| R2 inverse | 0.3819 | 0.3711 | 0.4066 |
| R3 inverse | 0.3810 | 0.3711 | 0.4066 |
| R10 a digit in the name | 0.3805 | 0.3711 | 0.4066 |

### records when2call renamed train: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R2 inverse | 0.1584 | 0.1429 | 0.1754 |
| R13 an imperative first word | 0.1580 | 0.1429 | 0.1754 |
| R3 inverse | 0.1577 | 0.1429 | 0.1754 |
| B: (name style, period, first word form), minority | 0.1562 | 0.1429 | 0.1754 |
| R12 the fewest schema-like words | 0.1540 | 0.1429 | 0.1754 |

### records when2call hard val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 the shortest name | 0.4351 | 0.3911 | 0.4742 |
| R7 position 6 | 0.4169 | 0.3911 | 0.4742 |
| R1 ends with a period | 0.4128 | 0.3911 | 0.4742 |
| B: S: source, minority (sources not visible to any system) | 0.4120 | 0.3911 | 0.4742 |
| R3 the diagnosis's rule (R1, then R2) | 0.4116 | 0.3911 | 0.4742 |

### records when2call natural val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R5 inverse: fewest characters | 0.4142 | 0.3731 | 0.4557 |
| R7 inverse: not position 2 | 0.4050 | 0.3731 | 0.4557 |
| R7 position 4 | 0.3982 | 0.3731 | 0.4557 |
| R4 inverse: fewest words | 0.3961 | 0.3731 | 0.4557 |
| R6 inverse: an option not rewritten | 0.3948 | 0.3731 | 0.4557 |

### records when2call renamed val: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| B: (name style, period, first word form), minority | 0.1770 | 0.1429 | 0.2135 |
| R2 inverse | 0.1746 | 0.1429 | 0.2135 |
| R3 inverse | 0.1746 | 0.1429 | 0.2135 |
| R13 an imperative first word | 0.1746 | 0.1429 | 0.2135 |
| R5 most characters | 0.1728 | 0.1429 | 0.2135 |

### val v4_tools.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R11 inverse: the longest name | 0.1774 | 0.1429 | 0.1860 |
| R7 position 2 | 0.1661 | 0.1429 | 0.1860 |
| R5 most characters | 0.1616 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.1585 | 0.1429 | 0.1860 |
| R4 most words | 0.1543 | 0.1429 | 0.1860 |

### val v4_tools_masked.jsonl: the 5 highest rules

| rule | accuracy | chance | tolerance |
|---|---:|---:|---:|
| R7 position 2 | 0.1661 | 0.1429 | 0.1860 |
| R5 most characters | 0.1616 | 0.1429 | 0.1860 |
| B: S: source, minority (sources not visible to any system) | 0.1585 | 0.1429 | 0.1860 |
| R4 most words | 0.1543 | 0.1429 | 0.1860 |
| R4 inverse: fewest words | 0.1517 | 0.1429 | 0.1860 |
