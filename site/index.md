---
layout: home
title: skycmd
titleTemplate: Tiny decision models in the browser

hero:
  name: skycmd
  text: Tiny typed decision models that play live
  tagline: "Typed questions in, calibrated probabilities out, in one forward pass. The default model, s3-pico, has 445,345 parameters in a 251 KB ternary file, plays all four games at the oracle's level and answers a drone-game frame in about 50 µs in WASM. Trained from scratch in Rust. Runs in the browser, Node, Python and the CLI; only the older s1 files also run in llama.cpp."
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
  <a href="./tutorial/21-s3-recursive-tiny">s3: 251 KB, four games</a>
  <a href="./tutorial/19-s2-architecture">The s2 architecture</a>
  <a href="./tutorial/20-dagger-lessons">What DAgger taught us</a>
  <a href="./bench/">Size vs accuracy vs latency</a>
</div>

</div>
