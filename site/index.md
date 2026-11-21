---
layout: home
title: skycmd
titleTemplate: Tiny decision models in the browser

hero:
  name: skycmd
  text: Tiny decision models in your browser
  tagline: "Model files from 143 KB to 20.7 MB, built on one MacBook and run on your device through WebAssembly. From s5 on, every experiment was pre-registered before training, and every result was published, failures included."
  actions:
    - theme: brand
      text: See the models
      link: /models/
    - theme: alt
      text: Play a game
      link: /play/
    - theme: alt
      text: Results
      link: /results/
---

<script setup>
import ModelCatalogue from './.vitepress/theme/components/ModelCatalogue.vue'
</script>

<div class="home-section">

<LiveStats />

<section class="home-block" aria-labelledby="project-state">

<h2 id="project-state">The state of the project</h2>

<p class="home-lead">The experiments are finished, as of 20 November 2026. Every line is closed or published: 20 model files (5 passed their gate, 10 failed it and are published as results, 5 had no gate), 16 data sets on Hugging Face and 24 pre-registrations. The repository stays as the record: the models, the data sets, the pre-registrations and the <a href="./paper/">technical report</a>.</p>

</section>

<section class="home-block" aria-labelledby="works-today">

<h2 id="works-today">What works</h2>

<p class="home-lead">Four of the models you can use, each with its key number and a demo on this site. The first two, an end-of-turn detector and a tool router, can go into a real project: <a href="./use/">Use it for real</a> shows how, and when not to. The other two are for the games and the playground. Their cards say how each was judged and what to know before using it.</p>

<ModelCatalogue home />

<p class="home-more"><a href="./models/">All models, with their status and what each one brings</a> · <a href="https://huggingface.co/spaces/maxgfr/skycmd">Every demo in one Space</a> · <a href="https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603">Every model, Space and data set in one Hugging Face collection</a></p>

</section>

<section class="home-block" aria-labelledby="how-it-works">

<h2 id="how-it-works">How the project works</h2>

<ul class="home-facts">
  <li><strong>A bar fixed before training.</strong> From s5 on, each experiment had a pre-registration with its gate, committed before the model was trained; amendments are dated, and a missed gate is reported as missed.</li>
  <li><strong>One laptop.</strong> The models were trained on one Apple M5 MacBook, in Rust; s3-pico was trained from scratch in about 47 minutes.</li>
  <li><strong>Runs in your browser.</strong> The demos run the models in WebAssembly on your device, from the files their numbers were measured on; nothing is sent to a server.</li>
</ul>

<p class="home-more"><a href="./novelty/">What we bring: every claim, with its measurement and its evidence</a></p>

</section>

<section class="home-block" aria-labelledby="what-failed">

<h2 id="what-failed">What failed, and what it taught</h2>

<p class="home-lead">Most experiments here missed their gate. Each failure is published with its numbers, and the version after it started from what it showed:</p>

<ModelCatalogue lessons />

<p class="home-more"><a href="./models/#results">Every failed gate and its lesson</a> · <a href="./articles/">Articles: five of these stories told in full, and an overview</a></p>

</section>

<section class="home-block" aria-labelledby="where-to-go">

<h2 id="where-to-go">Where to go</h2>

<ul class="home-entries">
  <li><a href="./models/">Models</a><span>Every model, its status, its size and where to try it.</span></li>
  <li><a href="./play/">Demos</a><span><a href="./play/">Play</a> four games, ask your own questions on <a href="./live/">Live</a>, pick a tool on <a href="./router/">Router</a>, forecast a series on <a href="./forecast/">Forecast</a>, test audio on <a href="./ear/">Listen</a>.</span></li>
  <li><a href="./results/">Results</a><span>Every gate with its numbers, failures included, and the <a href="./bench/">Bench</a> of sizes and latencies.</span></li>
  <li><a href="./paper/">Technical report</a><span>The whole project as a paper: the protocol, every gate as read, and what the readings show across lines. Not peer-reviewed; <a href="https://zenodo.org/records/24421273">on Zenodo</a>.</span></li>
  <li><a href="./paper/methods/">Methods paper</a><span>How the models are built, trained, calibrated and run, precisely enough to reimplement them: the ternary graph with its looped exits, the training recipe, exit thresholds chosen by playing, and the engine. Not peer-reviewed; <a href="https://zenodo.org/records/24420817">on Zenodo</a>.</span></li>
  <li><a href="./tutorial/">Tutorial</a><span>28 chapters: setup, data, training, export to WebAssembly, and the story of each model.</span></li>
  <li><a href="./use/">Use it for real</a><span>Which files fit a real project, the code to call the end-of-turn detector and the tool router, and their limits.</span></li>
  <li><a href="./install/">Install</a><span>The files, and the engine for the browser, Node, Python, the command line and an HTTP server.</span></li>
  <li><a href="https://github.com/maxgfr/skycmd">GitHub</a><span>The code, the pre-registrations and the result files.</span></li>
</ul>

</section>

</div>
