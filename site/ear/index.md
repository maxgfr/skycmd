---
title: Listen
outline: [2, 2]
---

# Listen: a decision model that hears

**s6-ear** is an s6 whose state is sound. It listens to the last 10 seconds of audio and answers typed questions about them, every 80 ms:

- has the speaker finished their turn?
- which word was said?

It has 1.6M parameters, a fifth of Smart Turn v3.2. It runs here in WebAssembly, log-mel front-end included, and the audio never leaves your browser. Below its conformal threshold, it says it would hand the clip to a larger model: that is the System 1 of a voice agent ([how it works](/tutorial/26-s6-ear)).

<ClientOnly>
  <EarDemo />
</ClientOnly>

## What you are looking at

- **The input.** The microphone is resampled to 16 kHz into a sliding buffer of 10.24 s: 128 frames of 80 ms, each of 8 log-mel hops of 40 bands.
- **The questions are text.** The options are rendered as text in the static stream and cached, so only the audio is encoded each time. The same model can be asked about other words without retraining.
- **The exits.** "Exit 2 / 4" means the model answered after two of its four block applications, because it was already confident enough. The thresholds come from conformal risk control on held-out clips.
- **The hand-off.** Below a second threshold the answer is "not sure". An agent would then pass the clip to Whisper or to Smart Turn. That threshold bounds the error among the answers s6-ear keeps.

The sample clips are under CC-BY-4.0:

- **“yes”:** from Speech Commands v0.02 (Warden, 2018), test split.
- **French, mid-turn and end of turn:** from `pipecat-ai/smart-turn-data-v3.2-test` (Pipecat), test split.

The model never trains on either split.
