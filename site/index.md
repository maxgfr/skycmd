---
layout: home
title: skycmd
titleTemplate: Tiny decision models in the browser

hero:
  name: skycmd
  text: Tiny typed decision models that play live
  tagline: "0.2M to 1.5M parameters in a 0.5 to 0.7 MB file. Typed questions in, calibrated probabilities out, in one forward pass and under a millisecond per drone-game decision in WASM. Trained from scratch in Rust; s2-nano matches the oracle in three drone games and a fine-tuned copy plays a Doom-like shooter. Runs in the browser, Node, Python and the CLI; the s1 files also run in llama.cpp."
  actions:
    - theme: brand
      text: Play
      link: /play/
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

<div class="links-row">
  <a href="./play/">Play the games</a>
  <a href="./play/?game=rescue">Rescue Run</a>
  <a href="./play/?game=dodge">Obstacle Dodge</a>
  <a href="./play/?game=doom">Doom-like</a>
  <a href="./install/">Install</a>
  <a href="./tutorial/01-why-tiny-decision-models">Start the tutorial</a>
  <a href="./tutorial/19-s2-architecture">The s2 architecture</a>
  <a href="./tutorial/20-dagger-lessons">What DAgger taught us</a>
  <a href="./bench/">Size vs accuracy vs latency</a>
</div>

</div>
