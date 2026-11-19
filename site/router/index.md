---
title: Router
outline: [2, 2]
---

# Router: pick a tool from a list you write

::: info Passed its pre-registered gate on fresh slates; the system was defined after v2's test was read
v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.

On 945 fresh requests about tools never seen in training, **s6-route** picks the right option on **0.8868**, against **0.8169 for BM25** and **0.9048 for all-MiniLM-L6-v2** (91.6 MB). It is **3.7 MB** in two files and takes **about 31 ms** per request in WebAssembly with the tool list cached (measured under Node on a machine in use). Without the three largest clusters of the test its advantage over BM25 is +0.0202 [−0.0062, +0.0481], an interval that holds zero, and with every tool renamed it falls to 0.2497. Every number: [results](/results/#s6-route-passed).
:::

**s6-route** is the file of [s6-route-nano](/results/#s6-route-nano-failed) (which failed its own gate), unchanged, and the static embedding potion-base-2M (Minish Lab, MIT) packed to 4 bits. Each scores every option; the router multiplies the two probabilities option by option, raises the product to 0.81 and renormalises: `p(i) ∝ (p_model(i) · p_embedding(i))^0.81`. When the top probability is below τ_ask3 = 0.671047 it does not pick: it asks the user. Nothing was trained for it.

<ClientOnly>
  <RouterDemo />
</ClientOnly>

## What you are looking at

- **The input.** A request and up to 7 tools, each `name: description`. The description is cut to its first sentence and 30 words, as every training and test slate was. The router adds the option *none of these tools fits the request* itself.
- **The two parts.** Under each bar, the probability that the model part (the s6-route-nano file, 2.4 MB, which also reads each tool's BM25 score against the request) and the embedding part (cosine between the request and each tool, 1.3 MB) give that option. The bar is their fusion.
- **Ask the user.** Below τ_ask3 the router answers "ask the user". On the fresh test it answered 86% of the requests at that threshold and was right on 94.3% of them; the announced goal was 95% at 80% coverage, so that goal was not met.
- **Fast mode.** The model part stops at its first exit (2 of 6 block passes): 0.8836 on the same test, about 10.5 ms per request.
- **Times.** The first request on a new tool list encodes the list in both parts; the following requests reuse it. The page shows both. Measured under Node for the gate: 30.9 ms cached and 45.3 ms for the first request (SIMD128), 23.5 and 34.3 ms with relaxed SIMD, which this page loads where the browser supports it.
- **Mask names.** The button renames every tool `tool_<k>`, the names-masked reading of the pre-registration, announced as a failure before the reading. With the names masked the model part is confidently wrong and pulls the fusion with it: 0.2497, where the embedding part alone keeps 0.7280. Use the router only on tool lists whose names say what the tools do.
- **The presets** are tool lists and requests written for this page, not rows of any dataset. The mark ✓ is the tool each request was written for; one request per list should get *none*. What the router answers is shown as it comes: of the 12 preset requests, written before the router was run on them, at full depth it routes 8 as written, asks the user on 2 and answers *none* on 2 that a developer tool serves (`search_code`, `run_tests`). With the names masked it picks a tool for all three *none* requests.

## What the result does not say

- **The tools of the test are v2's held-out tools.** Its requests are fresh, but the project had already read how every system does on these tools. On 2,000 fresh requests about the tools of MCP servers, a family unlike its training data, read on 19 November 2026 for its retrained successor and reported here, it picks the right option on 0.5530, where BM25 reads 0.5675 (no interval was computed for that difference). That successor, [s6-route7-nano](/results/#s6-route7-nano-failed), failed its own gate on the same requests, and the line is closed.
- **The advantage over BM25 is fragile.** It holds on the slate bootstrap that the gate asked for (+0.0698 [0.0455, 0.0952]), but not when the three largest clusters are left out (+0.0202 [−0.0062, +0.0481]) nor on a bootstrap over clusters (+0.0698 [−0.0079, +0.1404]).
- **MiniLM is more accurate on every test file.** With 91.6 MB to spend, all-MiniLM-L6-v2 with cosine is the better router on these tests.
- **Outside the tool lists of its training families** (MetaTool, CLINC150 intents, MASSIVE held-out scenarios) it stays near its embedding part: 0.5175 against 0.5340, 0.8040 against 0.7855 and 0.5340 against 0.4755.

The pre-registration, with the system, the gate and the reading: [`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md). The engine: `ToolRouter` in `crates/sky-web/src/router.rs`, and natively `sky-route fuse` ([install](/install/)).

The story of the line, two failures, then this pass, then four versions that tried to retrain its model part and closed the line, is told in an article: [A 3.7 MB tool router built on a MacBook](/articles/a-3-7-mb-tool-router).
