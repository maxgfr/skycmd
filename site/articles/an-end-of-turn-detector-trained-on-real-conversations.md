---
title: "An end-of-turn detector trained on real conversations: it learned them, and did not get better with voice agents"
description: "The third version of a 632 KB end-of-turn detector: trained on real two-speaker conversations, labelled from the other speaker's channel and checked before training, it improved on held-out human conversations and fell behind its predecessor on real speech addressed to an agent. Its pre-registration had said in advance what that pattern would mean."
date: 2026-11-20
outline: [2, 2]
---

# An end-of-turn detector trained on real conversations: it learned them, and did not get better with voice agents

*20 November 2026*

The second version of this project's end-of-turn detector, [s6-ear2-nano](/results/#s6-ear2-nano-passed), passed its gate on the clips of the reference model's own test set and was clearly weaker on real conversations. The third version was an attempt to close that gap with real conversations as training data. It was built and checked with care, trained once, read once, and it failed: it got better at the kind of conversation it was trained on and worse at the kind a voice agent faces. This article tells that version; the numbers are on [Results](/results/#s6-ear3-nano-failed), and every claim of the project, with its evidence, is on [What we bring](/novelty/).

## The context

skycmd builds very small "decision models" with ternary weights (each weight is −1, 0 or +1), written in Rust, run natively or in a browser through WebAssembly, and trained on one MacBook (Apple M5) under an automatic brake that keeps the machine usable, so training budgets count working minutes, not wall-clock time.

::: info Pre-registration and the gate
Before an experiment runs, the test, the bar to clear and what each kind of failure would mean are committed to the repository. The bar is the "gate": a set of conditions that must all hold. Each test is read once, a failure is published like a success, and the rules cannot be adjusted after a score is seen. Any change before the run is a dated amendment.
:::

An end-of-turn detector answers one question for a voice agent, a program you talk to: has the speaker finished their turn, or only paused? The agent asks it during a pause. A **false cut-off** is answering "finished" while the speaker was about to go on, the mistake a user notices most. Two measures recur below. The **AUC** (area under the ROC curve) is the probability that the model gives a higher score to a random real end of turn than to a random pause the speaker went on from: 0.5 is chance, 1.0 is perfect, and it does not depend on any threshold. **Recall at a false cut-off of 0.10** is the share of real ends of turn the model catches at the threshold where it cuts off 10% of the other pauses; it describes the region a voice agent works in, and it is a ranking measure, not a threshold you could deploy as is.

## Where v2 left off

v2 is a 632 KB ternary model, a student of [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) (Pipecat's 8.7 MB open model, BSD-2-Clause), trained on Smart Turn's public training clips. On Smart Turn's 31,527 test clips it is right on 0.8993 of them, against 0.9260 for Smart Turn through the same harness. On [eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data), LiveKit's set of real speech addressed to voice agents in 14 languages, which v2's design never read, it was much weaker: AUC 0.7299 against Smart Turn's 0.7838, read at the start of each labelled pause ([`runs/s6-ear2-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md)).

Two things about v2's training looked like causes, and the design study wrote them down with their evidence ([`docs/s6-ear3-study.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-study.md)):

- **It never trained on the silences it is asked about.** The demo's energy VAD (a voice-activity detector that calls each 10 ms frame speech or silence from its loudness) asks the model 0.2 s into a pause, then every 0.2 s up to 2.4 s. v2's training clips end where the speech ends: on 3,160 of them, the median trailing silence is 0 ms, and only 7.2% of the unfinished and 8.9% of the finished clips have 200 ms or more.
- **Its data are mostly not conversations.** Most of Smart Turn's training clips are marked synthetic, and the teacher itself is weak on eot-bench.

## The bet

Change only the training. The graph, the front end, the file format and the engine stay exactly v2's, so that v3 would be a drop-in replacement file. Two changes:

- **Real two-speaker conversations, labelled for free.** otoSpeech ([`otoearth/otoSpeech-full-duplex-280h`](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h)) is about 280 hours of spontaneous English between two people, each recorded on their own channel, under CC BY 4.0 and gated: the project's owner accepted its conditions, among them not to try to identify speakers and to respect the segments it has muted. The demo's VAD and turn policy, ported to Rust and checked bit for bit against the browser's code, are replayed on one speaker's channel; every pause the policy would ask about becomes a record. Its label comes from the other channel: **end of turn** if the other speaker takes the floor within 4 s (speaks for at least 1 s and is not just saying "mm-hm"), **hold** if the same speaker resumes first. Short acknowledgements of that kind, called backchannels, do not count as taking the floor. Pauses where both people talk at once, where nobody speaks within 4 s, or that lie near a muted segment are dropped and counted. No model writes a label.
- **The silences it is asked about.** The conversation records are cut where the policy asks, with half the weight on the first ask at 0.2 s; Smart Turn's clips are extended by their own quietest sound, so the model also sees them 0.2 to 2.4 s after the speech ends.

The model starts from v2's weights and trains for 120 working minutes on both sources, half and half. The gate was set on eot-bench at the policy's first question, 0.2 s into each pause, against v2 on the same cuts: an AUC higher by at least 0.02, with the lower bound of a 95% interval above 0 (E1); a better recall at a false cut-off of 0.10 (E2); no loss of more than 0.02 on the 13 languages other than English (E3), since the new data are English. It also had to keep v2's level on Smart Turn's clips (A1, A2) and be a drop-in file (G3).

## What was built and checked before training

Labels derived from a detector are only as good as the detector, so three checks were written into the pre-registration, each able to stop the line before any GPU time. Their results were committed before the first training step.

- **A second opinion on every label.** On the 109 validation sessions, each record was labelled again from a different, neural detector, Silero VAD (MIT licence), with the same rule. The two agree on **0.9509** of the 40,872 records Silero labels (95% interval [0.9488, 0.9530]; the bar was 0.85), and Silero's path drops 0.1016 of the 45,494 records (bar 0.30). Silero is used for this check only: it never labels, selects or drops a training record.
- **No leak from the other speaker.** If the other person's voice bled into this channel, the silence after a pause would itself reveal the label. On the 120,000 training records, the loudness of that silence alone predicts the label with an AUC of at most **0.4396** over the twelve asks (the bar was 0.60).
- **The harness reproduces the past.** Before any v3 score, the new evaluation code had to give v2's known numbers: 0.8993 on Smart Turn's test, and on eot-bench at the start of each pause v2's 0.7299 and Smart Turn's 0.7838, on the same 17,934 cuts. It did. The number of cuts for the gate, 16,230, was recorded before any model read them.

One part of this was changed after a look at data, and it is disclosed wherever v3 is reported. The cross-check first matched each pause's end to the nearest pause end Silero found. On five training sessions, half the records had no match: the energy VAD hears less speech than Silero and asks during speech that Silero hears as continuing. The rule was amended, before any validation or test session was read, to read Silero's label at the record's own instant instead ([amendment 2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-prereg.md)), and to keep every end-of-turn record, which are rare (about one pause in twenty).

## The run

A pilot of four arms of 12 working minutes compared three learning rates from v2's weights and one start from scratch, scored on the conversation validation records at 0.2 s. The middle rate won (AUC 0.7429, where v2's file reads 0.6175 on the same records); the scratch arm lost too much on Smart Turn's clips to be eligible. The main run then trained for 120 working minutes, 155,300 steps, seed 1, once, with no restart. Every stage of the chain, from the first smoke test at 17:38 to the gate at 23:39 on 19 November, ran once and exited cleanly.

## The result

Read once, at the end ([`runs/s6-ear3-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear3-nano/GATE.md)). On eot-bench, the intervals come from a paired bootstrap over turns: its 7,342 turns are resampled 2,000 times, all the cuts of a turn together, and v3 and v2 are compared on each resample.

| | s6-ear3-nano (v3) | s6-ear2-nano (v2) | Smart Turn v3.2 |
|---|---:|---:|---:|
| eot-bench, 0.2 s into each pause, AUC (16,230 cuts, 14 languages) | 0.7197 | 0.7448 | 0.7873 |
| the same, English only (3,902 cuts) | 0.7375 | 0.7737 | 0.8172 |
| held-out human-to-human conversations, AUC (30,000 records, reported) | 0.6878 | 0.6031 | 0.6430 |
| Smart Turn's test clips, accuracy (31,527 clips) | 0.8978 | 0.8993 | 0.9260 |

- **E1 fails.** On eot-bench v3 is 0.0252 below v2, with a 95% interval of [−0.0336, −0.0167]: not the gain of at least 0.02 that was asked, and not a gain at all.
- **E2 and E3 fail.** Its recall at a false cut-off of 0.10 is 0.0563 lower ([−0.0832, −0.0315]), and outside English it loses 0.0209 ([−0.0308, −0.0111]), past the allowed 0.02. English, the language of the new data, did not gain either.
- **What held.** On Smart Turn's clips it keeps v2's level (0.8978, bar 0.8893), and it is a drop-in file: 2.39 ms per decision in WebAssembly, 631,936 bytes, v2's exact tensor list. On those clips extended by 0.2 s and 0.6 s of their own background (reported, not gated) it keeps 0.8959 and 0.8935, where v2 drops to 0.8805 and 0.8512.
- **What it learned.** On the 113 conversation sessions held out of training, the same kind of human-to-human speech it trained on, its AUC is 0.6878 against v2's 0.6031.

## What it means

The pre-registration wrote, before the run, how to read exactly this case: if E1 fails on its lower bound, no improvement on real human-to-agent speech is shown, and the held-out conversations separate two causes. If they improve over v2, "the human-to-human signal does not transfer to agents"; if not, the model did not learn the signal at all. They improved. So the reading is the first one: the labels were learnable, the model learned them, and what it learned did not carry over to people talking to a machine.

The pre-registration had also named why this could happen, as limits of the design rather than as a forecast. The training labels describe what a human listener did next, not what an agent needs. The conversations are English, and eot-bench has 14 languages. And the test sessions share speakers with training, since the corpus allows no clean speaker split. The run does not measure which of these, if any, explains the gap, and this article does not choose one.

The file is published as evaluated, with a [card](https://huggingface.co/maxgfr/s6-ear3-nano) and a [Space](https://huggingface.co/spaces/maxgfr/s6-ear3-nano) that run the same microphone demo as v2, each opening with the failure. The site's [Listen page](/ear/) keeps v2, and v2 stays the file to take, with its fallback timer; when real conversations matter most, Smart Turn's larger file is still ahead of both.

## The limits of the reading

- **One training run, one seed.** Every interval is a bootstrap over test items (clips, or turns of eot-bench), not over training runs: it says how much a number would move on other test items, not how much a second training would move it.
- **No interval on the conversation test.** The held-out conversations were reported, not gated, and their numbers are point estimates. On the 2,061 records whose speaker never appears in training, v3 still ranks better than v2 (AUC 0.6663 against 0.6137), but catches fewer ends of turn at a false cut-off of 0.10 (0.1831 against 0.2254).
- **A second look at v2's weak spot.** v3 was designed after v2's eot-bench numbers were seen, and that reading chose the headline test, the bar outside English and the expected per-language effects. No eot-bench file was opened for the design and none was used for training or any choice; the 0.2 s cut point comes from the demo's policy and LiveKit's card. Smart Turn's test is still partly in-sample for the design, as v2's disclosure says.
- **An amended cross-check.** The label check was changed after a measurement on five training sessions, before any validation or test session was read (above).
- **Nothing of the corpus is published.** The per-record files of the conversation records stay on the machine that ran it, as the corpus's conditions and the pre-registration require; the repository holds counts and statistics only. The run's outputs on Smart Turn's and eot-bench's clips are in the release [`archive-2026-11-20`](https://github.com/maxgfr/skycmd/releases/tag/archive-2026-11-20).

## Check it

Every number of this version: [Results](/results/#s6-ear3-nano-failed); every claim of the project, with its evidence: [What we bring](/novelty/). The pre-registration, its three amendments and the result block (§17): [`docs/s6-ear3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-prereg.md); the design study: [`docs/s6-ear3-study.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-study.md); the result files, the checks made before training included: [`runs/s6-ear3-nano/`](https://github.com/maxgfr/skycmd/tree/main/runs/s6-ear3-nano) and the pilot's [`runs/s6-ear3-pilot/choice.json`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear3-pilot/choice.json). The earlier versions are told in [A 632 KB model that knows when you have finished speaking](/articles/a-632-kb-end-of-turn-detector).
