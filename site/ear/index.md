---
title: Listen
outline: [2, 2]
---

# Listen: a decision model that hears

## End of turn: s6-ear2-nano (passed its gate)

**s6-ear2-nano** is an end-of-turn detector: it listens through the microphone and, when the speaker pauses, says whether the turn is finished, so that a voice agent knows when to answer. It is right on 0.8993 of the 31,527 test clips of Smart Turn v3.2, against 0.9260 for Smart Turn itself, in a file of 632 KB and about 2.4 ms per decision in WebAssembly. Its limit is real conversations: at the cautious threshold this demo uses by default (the file's τ₈₅, fewer cut-offs) it answers "finished" on about half of the real ends of turn of eot-bench, which is why the demo also runs a fallback timer that ends a turn after 2.5 s of silence whatever the model says. The numbers are in [its section of Results](/results/#s6-ear2-nano-passed), and [the article](/articles/a-632-kb-end-of-turn-detector) tells the story of the run.

<ClientOnly>
  <TurnDemo />
</ClientOnly>

## The third version: s6-ear3-nano (failed its gate)

::: danger Failed its pre-registered gate: published as a result, not for use
**s6-ear3-nano** is the same graph trained again from v2 on real two-speaker conversations and on the silences this demo asks at. On eot-bench, real speech addressed to an agent, read 0.2 s into each pause, its AUC is 0.7197 where v2 reads 0.7448 on the same cuts and Smart Turn 0.7873, so it failed its gate; on held-out human-to-human conversations it reads 0.6878 against v2's 0.6031. It was designed after v2's eot-bench reading, and its label cross-check was amended after a measurement on five training sessions; both are disclosed with the result. This page keeps v2. The third version runs with the same demo in [its Space](https://huggingface.co/spaces/maxgfr/s6-ear3-nano); the numbers are in [its section of Results](/results/#s6-ear3-nano-failed).
:::

## The first version: s6-ear-nano (failed its gate)

The demo below is the first version, which failed its gate: with a microphone it almost never says that the speaker has finished their turn, because its end-of-turn accuracy is 0.5681, barely above the 0.4979 of always answering "finished".

**s6-ear** is an s6 whose state is sound. It listens to the last 10 seconds of audio and answers typed questions about them, every 80 ms:

- has the speaker finished their turn?
- which word was said?

::: danger Failed its pre-registered gate: published as a result, not for use
The first s6-ear-nano failed three of its four pre-registered gates: end of turn 0.5681 against 0.9263 for Smart Turn v3.2 (Pipecat's report; the v1 harness read 0.8518), keywords 0.8885 against 0.90, French zero-shot keywords 0.1330 against 0.20; its control without the Mimi pretraining did better on all four sets. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), and the demo below loads it so that the result can be heard. It answers at full depth, where the gates were read. The numbers are on the [results page](/results/#s6-ear-nano-failed).
:::

It has 1.5M parameters, about a fifth of Smart Turn v3.2. It runs here in WebAssembly, log-mel front-end included, and the audio never leaves your browser. Below a hand-off threshold (conformal for a model that passes its gates; 0.5 for this one), it says it would hand the clip to a larger model: that is the System 1 of a voice agent ([how it works](/tutorial/26-s6-ear)).

<ClientOnly>
  <EarDemo />
</ClientOnly>

## What you are looking at

- **The input.** The microphone is resampled to 16 kHz into a sliding buffer of 10.24 s: 128 frames of 80 ms, each of 8 log-mel hops of 40 bands.
- **The questions are text.** The options are rendered as text in the static stream and cached, so only the audio is encoded each time. The same model can be asked about other words without retraining.
- **The exits.** "Exit 2 / 4" means the model answered after two of its four block applications, because it was already confident enough. For a model that passes its gates the thresholds would come from conformal risk control on held-out clips; the failed s6-ear-nano answers at full depth, "exit 4 / 4", where its gates were read.
- **The hand-off.** Below a second threshold the answer is "not sure". An agent would then pass the clip to Whisper or to Smart Turn. For a model that passes, that threshold bounds the error among the answers s6-ear keeps; the published file has none, so the demo uses 0.5.

The sample clips are under CC-BY-4.0:

- **“yes”:** from Speech Commands v0.02 (Warden, 2018), test split.
- **French, mid-turn and end of turn:** from `pipecat-ai/smart-turn-data-v3.2-test` (Pipecat), test split.

The model never trains on either split.
