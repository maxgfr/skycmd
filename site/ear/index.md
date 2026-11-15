---
title: Listen
outline: [2, 2]
---

# Listen: a decision model that hears

**s6-ear** is an s6 whose state is sound. It listens to the last 10 seconds of audio and answers typed questions about them, every 80 ms:

- has the speaker finished their turn?
- which word was said?

::: danger Failed its pre-registered gate: published as a result, not for use
The first s6-ear-nano failed three of its four pre-registered gates: end of turn 0.5681 against 0.8518 for Smart Turn v3.2, keywords 0.8885 against 0.90, French zero-shot keywords 0.1330 against 0.20; its control without the Mimi pretraining did better on all four sets. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), and the demo below loads it so that the result can be heard. It answers at full depth, where the gates were read. The numbers are on the [results page](/results/#s6-ear-nano-failed).
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
