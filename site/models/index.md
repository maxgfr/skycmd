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

Pre-registered and not yet read on a test set: no score until the gate is read. Each pre-registration says what is tested, how, and what would count as a pass. No card is here at the moment: the last one, s6-cast v2, had its gate read on 18 November 2026 and is now [s6-cast2-pico](#s6-cast2-pico), under Published as results.

<ModelCatalogue group="progress" />
