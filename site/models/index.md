---
title: Models
pageClass: wide-page
outline: [2, 3]
---

<script setup>
import ModelCatalogue from '../.vitepress/theme/components/ModelCatalogue.vue'
</script>

# Models

Every model skycmd publishes, one card each: what it is for, its file size, how it was judged, the one fact it brings and where to try it. Most were judged by a gate, a bar written down in a pre-registration before the model was trained; a model that misses its gate is published anyway, as a result. The numbers on the cards come from [Results](/results/), [What we bring](/novelty/) and the Hugging Face model cards. Download links, formats and the engines each file runs in are on [Install](/install/#models).

Each model also has a demo on Hugging Face, and all of them are gathered in one place: [Every demo in one Space](https://huggingface.co/spaces/maxgfr/skycmd), where each demo opens with its model's verdict, passed or failed.

<ModelCatalogue legend />

## Works: use it {#works}

These models do what their card says, on the tests their card names. Four passed a pre-registered gate. Two, s3-pico and s5-lite, were measured without one, and their status says so.

<ModelCatalogue group="works" />

## Useful as a part, with a failed gate {#part}

It missed its gate, and what it does measure makes it worth using behind a rule-based scanner.

<ModelCatalogue group="part" />

## Published as results, not for use {#results}

Models that failed their gate, or that a later model replaced. Each file stays published as the result of its experiment, and each one taught something a later experiment used.

<ModelCatalogue group="results" />

## In progress {#in-progress}

Pre-registered and not yet read on a test set: no score until the gate is read. Each pre-registration says what is tested, how, and what would count as a pass. No card is here at the moment: the last ones to be read, s6-cast v2 on 18 November 2026 and s6-route v7 on 19 November, are now [s6-cast2-pico](#s6-cast2-pico) and [s6-route7-nano](#s6-route7-nano), under Published as results. Two lines go on without a model yet (19 November 2026): s6-sleep v4, whose next check, K1, is decided and not run ([`docs/s6-sleep4-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep4-prereg.md)), and s6-ear v3, whose pre-registration is being frozen ([`docs/s6-ear3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-prereg.md)).

<ModelCatalogue group="progress" />
