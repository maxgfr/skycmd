---
layout: home
title: skycmd
titleTemplate: Tiny decision models in the browser

hero:
  name: skycmd
  text: Tiny decision models that play live
  tagline: A bidirectional encoder with a typed decision head scores every option in one forward pass. Trained from scratch in Rust, running in your browser through WASM.
  actions:
    - theme: brand
      text: Play
      link: /play/
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
  <a href="./play/">Play the three games</a>
  <a href="./play/?game=rescue">Rescue Run</a>
  <a href="./play/?game=dodge">Obstacle Dodge</a>
  <a href="./tutorial/01-why-tiny-decision-models">Start the tutorial</a>
  <a href="./bench/">Size vs accuracy vs latency</a>
</div>

</div>
