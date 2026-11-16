---
license: apache-2.0
language:
  - en
  - fr
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - tool-routing
  - rust
  - skycmd
  - failed-gate
  - negative-result
---

# s6-route-nano

> **Failed its pre-registered gate: published as a result, not for use.** On T-tools, 2,000
> slates of tools that no training item contains, it picks the right option on **0.8395**. That is
> above potion-base-2M with cosine + 0.03 (0.6975) and above both trained baselines of its size,
> but not above **BM25 (0.8500)**, the lexical score it reads in its own input, which gate G1 asks
> it to beat (amendment 1 of
> [`docs/s6-nano2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-prereg.md)):
> **it failed G1** (difference −0.0105, 95% interval [−0.0265, +0.0055]). It also **failed G4**:
> **38.9 ms** per request in WASM with the tool list cached, where the bar is 2 ms. G2 and G3 held.
> It is not published as a router. The file is published so that the result can be checked and run
> (owner's decision of 15/11/2026, §11 of the pre-registration); the verdict does not change, and
> this run is not repeated. Numbers:
> [BENCH.md, section s6-route-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-route-nano-s6-nano-v2-a-tool-router-pre-registered-gate-failed)
> and [`runs/s6-nano2/eval-test/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-nano2/eval-test/results.md).
>
> **The follow-up:** this file, unchanged, fused with a 4-bit static embedding, passed its own
> gate on fresh slates as [s6-route](https://huggingface.co/{{USER}}/s6-route) (a system defined
> after this model's test was read); this model's verdict does not change.

**A 2.4 MB ternary tool router.** 5,587,201 parameters. Given a request and a list of up to 7
tools, each a name and one sentence, plus the option "none of these tools fits the request", it
gives a probability to each option in one forward pass. It is the second version of
[s6-nano](https://huggingface.co/{{USER}}/s6-nano) as a decision model, narrowed to one job. The
name was fixed before training; the pre-registered pilot rule then chose the micro size (0.6390
against 0.5778 for nano on validation, at least 0.04 asked), so it has the shape of
[s6-micro](https://huggingface.co/{{USER}}/s6-micro), not of s6-nano.

- **No demo:** the pre-registration publishes the playground only if the gate passes (§11).
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-nano2-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-prereg.md)
  (the gate in §8, amendments 1 and 2 before training, the result block)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## The gate, as read

T-tools, 2,000 slates (up to 7 tools and *none*; *none* is right on 20% of them), ternary, full
depth:

| gate | condition | measured | verdict |
|---|---|---|---|
| G1 | ≥ potion-cos + 0.03 | 0.8395 against 0.6675 + 0.03 = 0.6975 | holds |
| G1 | > static-ft (size-matched) | 0.8395 against 0.6735 | holds |
| G1 | > pair-lr (size-matched) | 0.8395 against 0.4500 | holds |
| G1 | > BM25 (amendment 1) | 0.8395 against 0.8500 | **fails** |
| G2 | *none* answers in 0.10–0.30; no tool name above twice its gold share + 0.01; the answer changes on ≥ 0.80 of the gold-removed pairs | 0.1565; largest excess −0.0085; 0.9528 | holds |
| G3 | ECE ≤ 0.05 | 0.0244 | holds |
| G4 | WASM SIMD128 median ≤ 2 ms per request, tool list cached; max \|p_wasm − p_native\| ≤ 1e-4 | 38.9 ms; 2.2e-6 | **fails** |

Paired bootstrap of the G1 differences (1,000 resamples of the slates, 95%): +0.1720 [0.1490,
0.1965] over potion-cos, +0.1660 [0.1415, 0.1895] over static-ft, +0.3895 [0.3625, 0.4150] over
pair-lr, **−0.0105 [−0.0265, +0.0055] against BM25**: a tie within the interval, where the gate
asks for strictly above.

## Results

T-tools, every system (accuracy; ECE; *none* F1):

| system | size | accuracy | ECE | *none* F1 |
|---|---:|---:|---:|---:|
| **s6-route-nano** | 2.4 MB | **0.8395** | **0.0244** | 0.7041 |
| BM25 (bm25s, over the slate's own tools) | no parameters | 0.8500 | 0.0610 | 0.6486 |
| all-MiniLM-L6-v2, cosine | 90.9 MB | 0.8840 | 0.0231 | 0.7516 |
| toolrouter (stanley-nv) | 7.66 MB | 0.7855 | 0.0886 | 0.3333 |
| static-ft (potion fine-tuned, reduced to the model's bytes) | ≤ 2.4 MB | 0.6735 | 0.0430 | 0.4715 |
| potion-base-2M, cosine | 7.56 MB | 0.6675 | 0.0393 | 0.4724 |
| potion-base-2M, cosine, 4 bits | about 1 MB | 0.6665 | 0.0398 | 0.4661 |
| pair-lr (size-matched logistic regression) | ≤ 2.4 MB | 0.4500 | 0.0426 | 0.3114 |
| name-prior | | 0.2000 | 0.8000 | 0.3333 |
| s6-nano (v1), same prompt | 715 KB | 0.0885 | 0.1152 | 0.0049 |

By source (accuracy):

| source | n | s6-route-nano | potion-cos | BM25 | MiniLM | toolrouter |
|---|---:|---:|---:|---:|---:|---:|
| T-tools (unseen tools; gated) | 2,000 | **0.8395** | 0.6675 | 0.8500 | 0.8840 | 0.7855 |
| T-metatool (never trained) | 2,000 | 0.3260 | 0.5245 | 0.4525 | 0.6610 | 0.5855 |
| T-clinc (intents by description) | 2,000 | 0.3980 | 0.7740 | 0.4775 | 0.8605 | 0.7235 |
| tool-routing-data `tools/test` | 100 | 0.3700 | 0.6100 | 0.3800 | 0.6000 | 0.7900 |
| BFCL `multiple` and `irrelevance` | 440 | 0.7205 | 0.7250 | 0.8295 | 0.7136 | 0.5250 |
| When2Call `test/mcq` | 1,863 | 0.5153 | 0.5722 | 0.4895 | 0.6720 | 0.5915 |
| MASSIVE, 4 held-out scenarios, en and fr | 1,657 | 0.3645 | 0.4762 | 0.3343 | 0.5836 | 0.6035 |

The announced goals: T-tools ≥ 0.80 is met; T-metatool and T-clinc ≥ potion-cos are not met; 0.95
among answered requests at τ_ask with coverage ≥ 0.70 is not met (0.9433 at coverage 0.7325).

## What it does well, and badly

- **Well:** on tools it never saw, far above its first version (0.8395 against 0.0885) and above
  potion-base-2M with cosine, a static embedding three times its size (0.6675), and above both
  trained baselines of its size. Calibrated (ECE 0.0244, against 0.0610 for BM25), and it abstains
  sensibly: accuracy 0.991 / 0.956 / 0.885 on the 50% / 70% / 90% of the slates it is most sure
  of (BM25 0.979 / 0.945 / 0.878, MiniLM 0.993 / 0.974 / 0.928). Its answers depend on the list:
  with the right tool removed, the answer changes on 0.9528 of the requests it had right.
- **Badly:** it ties BM25, the score it is handed, so it adds nothing measurable to it on the
  gated test. It is slow: 38.9 ms per request in WASM (SIMD128), 23.5 ms with relaxed SIMD, 24.0
  ms natively. **With every tool renamed `tool_<k>` it is at chance (0.1050)**, below the
  name-prior (0.1180), where BM25 keeps 0.8445 and potion-cos 0.7015 on the same slates; one
  reading, not tested, is that it ties each BM25 entry of its input to an option through a
  distinctive name. With the descriptions of the right tool and of a distractor swapped, it
  follows the description on 0.6700 and the name on 0.0780. Outside its training families it is
  weak: below potion-cos on every other source (MetaTool, CLINC150, tool-routing-data, When2Call,
  the held-out MASSIVE scenarios, and BFCL by little, 0.7205 against 0.7250, where BM25 is at
  0.8295).

## How to ask it

One `choice` question (`skycmd-s2` engine, the prompt of §3 and amendment 1 of the
pre-registration):

- **instruction**, word for word: `Which tool should handle the user's request?`
- **options**: up to 7 tools as `name: description`, the description cut to its first sentence
  (up to the first `.`, `!` or `?` followed by a space or the end) and to 30 words, as
  `sky_data::route::cut_description` does; then `none of these tools fits the request`, always
  last;
- **state**: `User: <request>` (one turn), then a line with each tool's BM25 score against the
  request and its rank in the slate, computed over the slate's own tools (not *none*), in the
  options' order: `BM25: <name> score <s> rank <r>; …`. `sky_infer::route::state` writes it
  (bm25s 0.3.16, Lucene variant, k1 1.5, b 0.75, its English stopwords; two decimals; rank 1 for
  the highest score, ties to the earlier tool), and
  `crates/sky-infer/tests/fixtures/route_bm25.json` holds requests, options and the expected
  states that any other implementation must reproduce.

For example, the body of `POST /v1/systemone` (or `skycmd decide --request`):

```json
{
  "state": "User: turn on the lights in the hall\nBM25: lights_on score 0.38 rank 1; weather score 0.00 rank 2; play_music score 0.00 rank 3",
  "questions": {
    "tool": {
      "type": "choice",
      "instructions": "Which tool should handle the user's request?",
      "criteria": {
        "lights_on": "Turns the lights on in a room.",
        "weather": "Gives the weather of a city.",
        "play_music": "Plays a song.",
        "none of these tools fits the request": ""
      }
    }
  }
}
```

A key with an empty description is shown as the key alone, so the last option is the *none* text
as trained. The engine reads at most 256 head tokens and caps each option at 48 tokens
(amendment 2), so a long description loses its tail.

**When to ask the user.** The file stores `skycmd-s2.route.tau_ask` = 0.72938, fixed on the 1,144
validation slates before the test was opened: answer when the top probability is at least τ_ask,
otherwise ask. On validation it gave 0.9501 accuracy among answered requests at coverage 0.7360;
on T-tools 0.9433 at coverage 0.7325. The engine does not expose this key; read it with any GGUF
reader.

## Full depth and latency

The file answers at full depth: its three exit thresholds are set to 1, and every reading above
was made on it. On 200 T-tools requests under Node, WASM matches the native probabilities to
2.2e-6 (relaxed SIMD 2.8e-6), with the same answer on 200 of 200. Median per request with the tool
list cached: 38,929 µs (SIMD128), 23,532 µs (relaxed SIMD); the first request, which encodes the
list: 50,922 and 33,204 µs. Natively, median 24,031 µs (p95 48,782 µs). At the first exit (2 of 6
block passes) the accuracy on T-tools is 0.8340 at a native median of 13,340 µs. Apple M5.

## Training

From scratch in Rust (mlx-rs) on an Apple M5, one run, seed 0, no restart, in a throttled mode,
so the budget is working minutes. The s6-micro graph (2 weight-shared blocks looped three times,
exits after 2, 4 and 6 block passes, d 256, 4 heads, vocabulary 4,096, windows of 384 state tokens
and 256 head tokens), initialised from s6-micro's masked-language pretraining (SimpleStories,
TinyStories, FineWeb and the non-web parts of Dolma 3 and TxT360), then decision training for
120 working minutes (20,504 steps; ternary QAT after the first 12), gold one-hot targets, on routing data only,
drawn 0.60 / 0.25 / 0.15:

- **Tools** (81,596 training records): glaive-function-calling-v2, hermes-function-calling-v1,
  ToolACE and Seal-Tools (Apache-2.0) and API-Bank (MIT). Tools are clustered by name and by
  description (all-MiniLM-L6-v2), and every cluster goes to training, validation or test as a
  whole: no test tool has a training tool in its cluster. Four slates per request: natural, hard
  (the nearest tools by potion-base-2M, cleaned by the teacher of distractors that also serve the
  request), the right tool removed (the answer becomes *none*), every tool renamed.
- **Intents** (95,411): MASSIVE in English and French, 14 of its 18 scenarios (CC-BY-4.0), and
  Banking77 (CC-BY-4.0), each intent a tool with one description written by the teacher from its
  name.
- **When2Call** (14,621; CC-BY-4.0): its train split, a call to the called tool, no call to
  *none*.

The teacher, GLM-5.3-flash (open weights, MIT, called through the z.ai API), wrote the intent
descriptions, filtered the hard distractors and rewrote the training tools' descriptions; it was
never sent a test request or slate, and it gives no training target. The teacher's weights are open
(MIT); how its outputs are used is described in
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md), with the licence
and commit of every source. MetaTool, CLINC150, BFCL, tool-routing-data and the When2Call test are
test only.

**Disclosure** (§10 of the pre-registration). This version was designed after s6-nano's results
were read, and after a study that read s6-nano's test dumps. Its gated test, T-tools, is new:
built by a rule fixed before training, hashed and read once. The When2Call and MASSIVE test
utterances were read in s6-nano's evaluation and are reported only. The clustering that defines
"unseen tool" uses all-MiniLM-L6-v2, which is also a reported baseline.

**Read after the gate, not a result.** On the already-read test, with nothing fitted, the
equal-weight geometric mean of this model's probabilities and potion-base-2M's (4 bits) reaches
0.8885 on T-tools, and 0.5290 on MetaTool and 0.7935 on CLINC150. These are hypotheses, not
results. That fusion is pre-registered as s6-nano v3
([`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md)),
to be read on fresh slates. It was read once on 16/11/2026 and passed: see
[s6-route](https://huggingface.co/{{USER}}/s6-route).

## Files

- `s6-route-nano.gguf`: ternary weights, tokenizer, temperatures, exit thresholds set to 1 (full
  depth) and `skycmd-s2.route.tau_ask`. 2,418,912 bytes, sha256
  `a124dfe86f76e9547db98edc03351f66908617ae2be3f5a4f423a876b7956c82`.
