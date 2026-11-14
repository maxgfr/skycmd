---
layout: home
title: skycmd
titleTemplate: Tiny decision models in the browser

hero:
  name: skycmd
  text: Tiny typed decision models that play live
  tagline: "Typed questions in, calibrated probabilities out, in one forward pass. s3-pico, 251 KB, plays all four games at the oracle's level in about 50 µs per frame in WASM. s5-lite, 20.7 MB, answers questions you write yourself, and option order or key names cannot change its answers by construction. Everything is Rust, from training to the browser."
  actions:
    - theme: brand
      text: Play
      link: /play/
    - theme: brand
      text: Live
      link: /live/
    - theme: alt
      text: Forecast
      link: /forecast/
    - theme: alt
      text: What we bring
      link: /novelty/
    - theme: alt
      text: Results
      link: /results/
    - theme: alt
      text: Install
      link: /install/
    - theme: alt
      text: Tutorial
      link: /tutorial/
    - theme: alt
      text: Bench
      link: /bench/
    - theme: alt
      text: GitHub
      link: https://github.com/maxgfr/skycmd
---

<div class="home-section">

<LiveStats />

<FeatureCards />

<div class="home-s6">

**For researchers.** What is new here, the measurement behind each claim, failures included, and a demo for each, in one table: [What we bring, at a glance](./novelty/#at-a-glance). Every gate was fixed before training and every amendment is dated ([Results](./results/)); the methods and data are credited with links on [Sources](./sources/).

**New, November 2026: s6-nano.** One 715 KB ternary model plays the four games at the oracle's level and answers typed questions in free text: 0.480 on `typed_decisions_bench`, above the majority class (0.398) where the earlier small models stayed below it, with an ECE of 0.0479. It passed a gate fixed before training, and its two siblings that missed theirs are reported too: s6-micro on calibration, s6-ear-nano on three of four gates. In WASM it gives the native answers to 9.5e-7 on 200 bench questions, at 436 µs per Flappy frame under Node (SIMD128 build; about 1.2x faster with relaxed SIMD, which the site loads where the browser supports it). [Try it on Live](./live/#s6-nano-the-questions-it-was-trained-for) · [Results](./results/#s6-november-2026) · [Chapter 27](./tutorial/27-s6-nano) · [What it shows](./novelty/#_8-what-s6-adds-trained-in-november-2026)

**Also new: s6-secrets-pico.** A 250 KB ternary model trained to say whether a short text exposes a secret, which service issued it and what kind of value it is. On a synthetic bench of file types it never saw it beats a gitleaks-style rule scanner (secret 0.9990 against 0.977, service 0.9795 against 0.787); its line answer is weak (0.6859). **On real files it fails:** measured after publication, it flags 59.8% of clean files at p ≥ 0.9 and 49.4% at p ≥ 0.999 ([measurement](https://github.com/maxgfr/secretgate/pull/19)). It is published as the result of its pre-registration, not as a scanner. The scanner's score was written down before training; in WASM the model takes 1,420 to 1,427 µs per decision under Node. [Try it on Live](./live/#s6-secrets-pico-is-a-secret-exposed-in-this-text) · [Results](./results/#s6-secrets-pico-passed-published-at-full-depth) · [Real text](./results/#s6-secrets-real-text) · [Chapter 28](./tutorial/28-s6-secrets-pico)

</div>

<div class="links-row">
  <a href="./play/">Play the games</a>
  <a href="./play/?game=rescue">Rescue Run</a>
  <a href="./play/?game=dodge">Obstacle Dodge</a>
  <a href="./play/?game=doom">Doom-like</a>
  <a href="./live/">Live: ask s5-lite anything</a>
  <a href="./forecast/">Forecast: TinyCast and Timestral-5M in the browser</a>
  <a href="./live/#s6-nano-the-questions-it-was-trained-for">s6-nano: games and typed questions, 715 KB</a>
  <a href="./live/#s6-secrets-pico-is-a-secret-exposed-in-this-text">s6-secrets-pico: secrets on its bench, not a scanner, 250 KB</a>
  <a href="./results/">s6 and s5 results, failures included</a>
  <a href="./novelty/#at-a-glance">What is new, with its evidence</a>
  <a href="./sources/">Sources: papers, data and credits</a>
  <a href="https://huggingface.co/maxgfr">Models on Hugging Face</a>
  <a href="./install/">Install</a>
  <a href="./tutorial/01-why-tiny-decision-models">Start the tutorial</a>
  <a href="./tutorial/27-s6-nano">s6-nano: games and questions in 715 KB</a>
  <a href="./tutorial/28-s6-secrets-pico">s6-secrets-pico: a secret detector that fails on real files</a>
  <a href="./tutorial/23-s5-lite">s5-lite: invariant by construction</a>
  <a href="./tutorial/24-reproduce-s5-lite">Reproduce s5-lite</a>
  <a href="./tutorial/21-s3-recursive-tiny">s3: 251 KB, four games</a>
  <a href="./tutorial/19-s2-architecture">The s2 architecture</a>
  <a href="./tutorial/20-dagger-lessons">What DAgger taught us</a>
  <a href="./bench/">Size vs accuracy vs latency</a>
</div>

</div>
