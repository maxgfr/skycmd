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

<div class="links-row">
  <a href="./play/">Play the games</a>
  <a href="./play/?game=rescue">Rescue Run</a>
  <a href="./play/?game=dodge">Obstacle Dodge</a>
  <a href="./play/?game=doom">Doom-like</a>
  <a href="./live/">Live: ask s5-lite anything</a>
  <a href="./forecast/">Forecast: TinyCast and Timestral-5M in the browser</a>
  <a href="./results/">s5 results, failures included</a>
  <a href="https://huggingface.co/maxgfr">Models on Hugging Face</a>
  <a href="./install/">Install</a>
  <a href="./tutorial/01-why-tiny-decision-models">Start the tutorial</a>
  <a href="./tutorial/23-s5-lite">s5-lite: invariant by construction</a>
  <a href="./tutorial/24-reproduce-s5-lite">Reproduce s5-lite</a>
  <a href="./tutorial/21-s3-recursive-tiny">s3: 251 KB, four games</a>
  <a href="./tutorial/19-s2-architecture">The s2 architecture</a>
  <a href="./tutorial/20-dagger-lessons">What DAgger taught us</a>
  <a href="./bench/">Size vs accuracy vs latency</a>
</div>

</div>
