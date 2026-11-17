# s6-sleep3-nano: report (docs/s6-sleep3-prereg.md)

This is the third version of the same idea on the same games, and its two changes (the gates held, an auxiliary recall loss in training) were found by looking at training curves on train sequences. The auxiliary loss gives the model the colour asked as a training target at the ask frames; it is not available at inference. The gates are constants in this configuration. v2 was designed after reading v1's pilot: its logs, its dev scores (all at chance), its checkpoints and optimiser states, and its training sequences. The changes answer an optimisation failure seen in training and in the weights, not a test score. v1's test seeds of recall and rule110, and the arena seeds of this experiment, have never been read. v1's dev seeds were read; v2 decides on a fresh dev set (docs/s6-sleep2-prereg.md, section 10).

Pilot: **works** ({"m_n1_minus_c": 0.74575, "ci95": [0.73225, 0.75925], "needs": ">= 0.1 with the lower bound above 0", "m_zeroed_minus_c": 0.0025000000000000022, "zeroed_needs": "<= 0.03"}).

Chosen N (dev2 recall, mean of 3 seeds): **1** ({"1": 0.48233333333333334, "2": 0.4910833333333333, "4": 0.4856666666666667}).

## Probes (test seeds, mean of 3 training seeds)

| probe | control | N=1 | N=2 | N=4 |
|---|---:|---:|---:|---:|
| recall | 0.2440 | 0.4892 | 0.4953 | 0.4922 |
| rule110 | 0.5290 | 0.5175 | 0.5175 | 0.5174 |

Recall by delay and by the colours the asked door was shown with (pooled over the 3 seeds):

| reading | 1-4 | 5-8 | 9-16 | 17-32 | one colour | several |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.2420 | 0.2398 | 0.2518 | 0.2423 | 0.2433 | 0.2446 |
| N = 1 | 0.6147 | 0.5590 | 0.4530 | 0.3302 | 0.4717 | 0.5052 |
| N = 2 | 0.6213 | 0.5823 | 0.4693 | 0.3083 | 0.4657 | 0.5224 |
| N = 4 | 0.6290 | 0.5910 | 0.4620 | 0.2867 | 0.4602 | 0.5214 |

## Hypotheses

- **H0, the memory is used**: recall at N = 1 minus the control = +0.2452 (95% interval +0.2370 to +0.2532), needs at least 0.1: **holds**.
- **H1, sleep helps**: recall at N = 4 minus N = 1 = +0.0030 (95% interval -0.0010 to +0.0068), needs a lower bound above 0: **not shown**.
- H1b (secondary, no gate): rule110 at N = 4 minus N = 1 = -0.0001 (95% interval -0.0007 to +0.0005).
- **H2, the games do not lose** (N = 1, sleep vs control, at most 2% under): flappy 15.87 vs 71.05 (-77.7%); rescue 3.00 vs 3.00 (-0.1%); dodge 5.89 vs 7.94 (-25.8%); doom 3.37 vs 3.37 (+0.1%): **not shown**.
- **H3, the language tasks do not lose**: typed_decisions_bench 0.4897 vs 0.4808 for the control, at most 0.01 under: **holds**.
