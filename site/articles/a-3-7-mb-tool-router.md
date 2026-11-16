---
title: "A 3.7 MB tool router built on a MacBook: two failures, then a pass and its limits"
description: "A small model that picks an assistant’s tool: two failed versions, published as failures, a method written down before every test, and a third version that passes, without hiding what that pass does not prove."
date: 2026-11-16
outline: [2, 2]
---

# A 3.7 MB tool router built on a MacBook: two failures, then a pass and its limits

*16 November 2026*

An assistant that can use tools first has to pick one. That choice is usually left to a large language model. I wanted to see how far a tiny model can go: one that fits in a web page and trains on a laptop. This article covers three versions: two failures, then one that passes the test set in advance. The failures matter as much as the pass, and the pass has limits, which I spell out.

## The context: skycmd

A short reminder, for readers who arrive here directly: skycmd gathers very small "decision models", from a few hundred kilobytes to a few megabytes. Their weights are ternary: each one is −1, 0 or +1, which keeps them compact and fast to compute. They are written in Rust and run in the browser through WebAssembly. Everything is trained on a single MacBook (Apple M5), in a throttled mode that keeps the machine usable in the meantime.

One rule frames every experiment: it is pre-registered.

::: info Pre-registration, in practice
Before an experiment runs, I write down and commit three things: the test, the bar to clear (which I call the "gate"), and what a failure would mean. The test is read once. A failure is published like a success. The idea comes from scientific research: fixing the rule before seeing the results stops you from adjusting it afterwards.
:::

## The problem: picking the right tool

The task is called tool routing. At run time, an assistant receives a list of tools, each with a name and a one-sentence description. Given a user request, it has to name the right tool, answer "none of these", or ask for more detail when it is too unsure.

The hard part is that the list changes. The router must work with tools it never saw during training. So it cannot learn the list by heart: it has to connect a request to a description.

## v1: a gate that a constant answer could pass

The first version, s6-nano, was 715 KB. It was a small general-purpose model, and it passed its own gate. A later study found the problem: on 17 tasks out of 20, it gave the same answer to at least 95% of the items. A program that always gives the same answer would have passed the gate too.

On the new routing test it scores 0.09. With 8 options per question, chance is about 0.125.

First lesson: a gate must compare the model against trivial baselines, not only against a target number.

## v2: a specialised model, beaten by a formula

The second version, [s6-route-nano](https://huggingface.co/maxgfr/s6-route-nano), has one job only: routing. It weighs 2.4 MB for 5.6 million parameters.

Its main idea comes from the project's best result, a secrets detector where rules propose candidates and the model judges them. The principle: whatever a cheap algorithm can compute, compute it and hand it to the model. Here, that algorithm is BM25. BM25 is a classic search score: it measures the words a request shares with each tool's description, giving more weight to rare words. It learns nothing and weighs nothing. Each tool's BM25 score and rank were written into the model's input.

Training took 120 "working minutes" on the M5, about two hours of GPU time spread over a morning because of the throttling: 20,504 steps, on about 190,000 examples built from public tool-calling datasets. A larger open-weight model (GLM-5.3-Flash, MIT licence) served as a teacher, only to clean the training labels (4,174 calls).

The gate compared the model with a static embedding, [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M). An embedding turns a text into a list of numbers, so that texts with close meanings get close lists; you then pick the tool whose description is closest to the request. "Static" means each word has a fixed list, which makes the whole thing very light.

The test: 2,000 requests about tools from families never seen in training. Throughout this article, the number measured is the share of requests for which the system gives the right answer.

| System | Size | Right answers (v2's test, 2,000 requests) |
|---|---:|---:|
| potion-base-2M, static embedding (the gate's baseline) | 7.6 MB | 0.6675 |
| s6-route-nano (v2) | 2.4 MB | 0.8395 |
| BM25 alone | 0 bytes | 0.8500 |
| all-MiniLM-L6-v2, sentence encoder | 91.6 MB | 0.8840 |

The model beats its gate's baseline by a wide margin. But BM25 alone, without a single byte of model, does better, and a 91.6 MB sentence encoder does better still. On speed, the model takes 39 ms per request in the browser, where the bar was 2 ms.

The gate failed, on two conditions. First, the model does not beat the score it was handed: the difference with BM25 is −0.011, with a 95% interval of [−0.027, +0.006]. That interval is the range where the true difference most likely lies, given the luck of the sample; since it contains zero, this is a tie. Second, the model is 19 times too slow. I published it as a failure, with its file.

## Reading the failure

A failure read carefully teaches something. I renamed every tool `tool_1`, `tool_2`, and so on. The model then falls to chance (0.105), while BM25, on the same text, keeps 0.84. The model had not learned to understand requests: it had learned to read the lexical score through distinctive tool names.

Outside its training families, it is weak on meaning. On the intents of CLINC150, a public benchmark, it scores 0.40, against 0.77 for the embedding.

One detail caught my attention, though: the small model and the small embedding are wrong on different requests.

## A lead found by looking at the test

When two systems make mistakes in different places, combining them can help. I multiplied, option by option, the model's probabilities by those of the same embedding packed to 4 bits (1 MB). On that test, the product reaches 0.8885: above BM25 and above MiniLM. For comparison, BM25 multiplied by the embedding gives only 0.8475.

That number is tempting, but it was found by looking at the test. If you try combinations until one works on a dataset, you often end up with a coincidence. A number obtained that way is a hypothesis, not a result. Turning it into a result required data nobody had looked at yet.

## v3: no training, fresh data, one reading

The third version, [s6-route](https://huggingface.co/maxgfr/s6-route), required no training at all. It is the pair "v2 model + 4-bit embedding", frozen. It has two settings only: an exponent of 0.81, which makes the probabilities more or less sharp, and a threshold below which the system asks the user instead of choosing. Both were read on validation data, never on a test.

Then, in this order:

1. the pre-registration is committed ([`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md));
2. 945 fresh requests are built: they are about the same held-out tools, and no system has ever scored them. On top of that come 2,000 requests from each of three other benchmarks;
3. the fingerprints (hashes) of these files are committed before any system reads them, so anyone can check they did not change;
4. the scoring code is rehearsed on validation data;
5. the whole thing is read once.

<figure class="article-figure">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 300" width="440" height="300" role="img" aria-labelledby="fig1-title fig1-desc" fill="currentColor" font-size="14" style="display:block;width:100%;max-width:440px;height:auto;margin:0 auto">
<title id="fig1-title">Right answers on the 945 fresh requests</title>
<desc id="fig1-desc">Horizontal bar chart, from 0 to 1. MiniLM, 91.6 MB: 0.9048. s6-route, 3.7 MB: 0.8868. BM25 alone, 0 bytes: 0.8169. v2 model alone, 2.4 MB: 0.7926. 4-bit embedding alone, 1 MB: 0.7386.</desc>
<g><title>MiniLM (all-MiniLM-L6-v2) · 91.6 MB: 0.9048</title><text x="0" y="19">MiniLM (all-MiniLM-L6-v2) · 91.6 MB</text><path d="M0,26h285.5a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-285.5z" fill-opacity=".6"/><text x="297.5" y="39">0.9048</text></g>
<g><title>s6-route · 3.7 MB: 0.8868</title><text x="0" y="73" font-weight="700">s6-route · 3.7 MB</text><path d="M0,80h279.8a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-279.8z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="291.8" y="93" font-weight="700">0.8868</text></g>
<g><title>BM25 alone · 0 bytes: 0.8169</title><text x="0" y="127">BM25 alone · 0 bytes</text><path d="M0,134h257.4a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-257.4z" fill-opacity=".6"/><text x="269.4" y="147">0.8169</text></g>
<g><title>v2 model alone · 2.4 MB: 0.7926</title><text x="0" y="181">v2 model alone · 2.4 MB</text><path d="M0,188h249.6a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-249.6z" fill-opacity=".6"/><text x="261.6" y="201">0.7926</text></g>
<g><title>4-bit embedding alone · 1 MB: 0.7386</title><text x="0" y="235">4-bit embedding alone · 1 MB</text><path d="M0,242h232.4a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-232.4z" fill-opacity=".6"/><text x="244.4" y="255">0.7386</text></g>
<g stroke="currentColor" stroke-opacity=".5"><line x1="0" y1="274.5" x2="321" y2="274.5"/><line x1="0.5" y1="274" x2="0.5" y2="279"/><line x1="80.5" y1="274" x2="80.5" y2="279"/><line x1="160.5" y1="274" x2="160.5" y2="279"/><line x1="240.5" y1="274" x2="240.5" y2="279"/><line x1="320.5" y1="274" x2="320.5" y2="279"/></g>
<g fill-opacity=".8" font-size="13" text-anchor="middle"><text x="0" y="297" text-anchor="start">0</text><text x="80" y="297">0.25</text><text x="160" y="297">0.5</text><text x="240" y="297">0.75</text><text x="320" y="297">1</text></g>
</svg>
<figcaption>Figure 1. Share of right answers on the 945 fresh requests, read once (the axis starts at zero). s6-route, in colour, is 0.018 below MiniLM for 25 times fewer bytes, and above BM25 and above each of its two parts.</figcaption>
</figure>

On the fresh requests, s6-route scores 0.8868, with 3.7 MB in two files. The difference with BM25 is +0.070, with a 95% interval of [0.046, 0.095]: this time, zero is far away. Against MiniLM, it loses 0.018, for 25 times fewer bytes; the gate allowed −0.03. It also does better than either of its two parts on its own.

Its probabilities are well calibrated. A system is well calibrated when its probabilities mean what they say: among the answers given with 80% confidence, about 80% are right. The calibration error (ECE), which measures the average gap between stated confidence and actual accuracy, is 0.016.

In the browser, it takes 31 ms per request with the tool list cached, and 10.5 ms in a fast mode that loses 0.003. On 200 requests, it gives the same answers as the Python reference, 200 times out of 200.

The gate passed.

## What the pass does not say

These limits matter as much as the result:

- The system was defined after v2's test was read. Fresh data protects against the luck of a choice made by looking, not against everything.
- The fresh requests are about the same held-out tools as v2's test.
- Without the three largest tool families, the advantage over BM25 shrinks to +0.020, with an interval that contains zero.
- With tool names masked, it falls to 0.25. It needs distinctive tool names.
- One announced goal was missed: when it chooses to answer on its own (86% of requests), it is right 94.3% of the time. I had hoped for 95%.
- MiniLM is still more accurate.
- On tool lists unlike its training families, it is about as good as its embedding part: slightly below on MetaTool (0.52 against 0.53), slightly above on CLINC150 (0.80 against 0.79) and on MASSIVE (0.53 against 0.48).

## What I take from it

**Put the zero-byte baseline in the gate.** BM25 was a harder judge than a neural baseline three times the model's size. Had v2's gate compared the model with the embedding only, accuracy would have looked settled. Before training anything, it is worth measuring what the simplest possible method does: a constant answer, a word search.

**Hand the small model what simple algorithms already compute.** A model of a few megabytes has no room to rediscover BM25. Giving it the score frees its capacity for something else. The risk, as v2 showed, is that it just copies what it is given: you have to check what it adds.

**When a small model plateaus, look for a complementary part before growing it.** The two parts of s6-route are wrong on different requests, and their product costs only 0.02 ms more. Growing the model would have meant a new training run, with no guarantee of a result.

**Pre-registration is what turns a lucky number into a result.** The 0.8885 of the lead was only a hint. Writing the rule, building fresh data, reading once: that is what lets me say the 0.8868 is not a coincidence picked after the fact. It is also what forces the limits to be published with the result.

**All of this fits on a laptop.** The only training run in this story took about two hours of throttled GPU time. The rest, meaning the combination, the settings and the evaluation, took a few minutes of CPU.

## Try it and check it

::: tip Try it
The demo runs in the browser: give it a list of tools and a request, and it shows the probability of each tool, along with "none" and "ask the user". It is on this site's [Router](/router/) page, and as a [live demo on Hugging Face](https://huggingface.co/spaces/maxgfr/s6-route).
:::

Everything else is public: [the model files](https://huggingface.co/maxgfr/s6-route), [the failed v2](https://huggingface.co/maxgfr/s6-route-nano), both pre-registrations ([v2's](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-prereg.md) and [v3's](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md)), the result file of the fresh reading ([`runs/s6-nano3/eval-fresh/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-nano3/eval-fresh/results.md)), [all the results](/results/) and [the code](https://github.com/maxgfr/skycmd). The potion-base-2M embedding is the work of Minish Lab, under the [MIT licence](https://huggingface.co/minishlab/potion-base-2M).
