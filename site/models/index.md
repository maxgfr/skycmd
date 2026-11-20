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

## Published as results, not for use {#results}

Models that failed their gate, that passed only on generated data, or that a later model replaced. Each file stays published as the result of its experiment, and each one taught something a later experiment used.

<ModelCatalogue group="results" />

## Closed without a model {#closed}

No model is in progress: the experiments ended on 20 November 2026, and every pre-registration has been read or closed. Nine versions closed without a model, so they have no card here. s6-sleep v1, v2 and v4 and s6-route v4, v5 and v6 stopped at a check their pre-registration named, before any model was published. The fine-tuning of LFM3-1.7B trained nothing, because no method fitted the machine's memory. The flexible-weights experiments were closed without running after a literature review, and a possible s6-secrets v4 was closed by a diagnosis. Each one is in [Results](/results/), with its reason.
