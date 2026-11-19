# Site and Spaces check, 18 November 2026

*Completed on 19/11/2026: the site's demos and the eight Spaces listed under "Not tested" below were checked in Chrome that day, in [`docs/site-check-2026-11-19.md`](site-check-2026-11-19.md).*

A check of the public site (<https://maxgfr.github.io/skycmd/>), of the 18 Hugging Face Spaces
listed in `hf/spaces.json`, of the 18 model pages and of the 15 data sets named in `README.md`.
Nothing was changed: this file records what was tested and what was seen.

**Coverage, stated first.** The browser extension used to drive Chrome disconnected after the
tenth Space and did not come back on three attempts, so the check stopped using the browser
there, as planned. What follows marks each cell with how it was obtained:

- **Chrome**: opened in a real Chrome tab, resources read from `performance.getEntriesByType('resource')`,
  console read with console capture on before the page loaded, demo exercised by clicking.
- **HTTP**: fetched without a browser (status code, `<title>`, `<h1>`, first paragraph, every
  same-site link and asset of the served HTML), or read from the Hub API.

The deployed site was at main's latest commit when the site was fetched: the Pages run for
`67563f3` had completed, and the Results page served the per-window readings sentence that only
that commit added ("computed later the same day"). It was not yet deployed at the start of the
check (the run was in progress) and was waited for.

## The site

Pages fetched over HTTP. 108 distinct same-site links and assets found in the served HTML of the
24 pages (navigation, sidebar, in-page links, scripts, styles, images) were each fetched once:
**all 108 answered 200**. Console, broken images rendered at run time and the demos were **not**
tested on the site (see "Not tested").

| Page | Loads (status, title, h1) | Console | Links | Demo exercised, number recorded | Verdict |
|---|---|---|---|---|---|
| `/` (home) | 200, "skycmd", "skycmd Tiny decision models in your browser" | not tested | all 200 | none on this page | OK (HTTP only) |
| `/models/` | 200, "Models", "Models" | not tested | all 200 | — | OK (HTTP only) |
| `/results/` | 200, "Results", "Results: what works, what does not" | not tested | all 200 | — | OK (HTTP only); serves the 67563f3 sentence |
| `/novelty/` | 200, "What we bring", "What skycmd brings" | not tested | all 200 | — | OK (HTTP only) |
| `/bench/` | 200, "Bench", "Bench" | not tested | all 200 | not tested | OK (HTTP only) |
| `/install/` | 200, "Install", "Install and use" | not tested | all 200 | — | OK (HTTP only) |
| `/live/` | 200, "Live", "Live: a decision model in your browser" | not tested | all 200 | not tested | OK (HTTP only) |
| `/play/` | 200, "Play", "Play" | not tested | all 200 | not tested | OK (HTTP only) |
| `/router/` | 200, "Router", "Router: pick a tool from a list you write" | not tested | all 200 | not tested | OK (HTTP only) |
| `/forecast/` | 200, "Forecast", "Forecast: time-series models in your browser" | not tested | all 200 | not tested | OK (HTTP only) |
| `/ear/` (Listen) | 200, "Listen", "Listen: a decision model that hears" | not tested | all 200 | not tested | OK (HTTP only) |
| `/secrets-verify/` | 200, "Secrets verifier", "Secrets verifier: is this value a real credential?" | not tested | all 200 | not tested | OK (HTTP only) |
| `/memory/` | 200, "Memory: s6-sleep3-nano (failed gate)", same h1 | not tested | all 200 | not tested | OK (HTTP only) |
| `/idle-training/` | 200, "Train while you work", same h1 | not tested | all 200 | — | OK (HTTP only) |
| `/sources/` | 200, "Sources & credits", same h1 | not tested | all 200 | — | OK (HTTP only) |
| `/articles/` | 200, "Articles", "Articles"; newest first | not tested | all 200 | — | OK (HTTP only) |
| `/articles/a-memory-that-works-and-a-model-that-does-not` | 200, title = h1, dated 17 November 2026 | not tested | all 200 | — | OK (HTTP only) |
| `/articles/a-632-kb-end-of-turn-detector` | 200, title = h1, dated 17 November 2026 | not tested | all 200 | — | OK (HTTP only) |
| `/articles/a-250-kb-secrets-verifier` | 200, title = h1, dated 16 November 2026 | not tested | all 200 | — | OK (HTTP only) |
| `/articles/a-3-7-mb-tool-router` | 200, title = h1, dated 16 November 2026 | not tested | all 200 | — | OK (HTTP only) |
| `/tutorial/` | 200, "Tutorial", "Tutorial" | not tested | all 200 | — | OK (HTTP only) |
| `/tutorial/03-typed-decisions` | 200, title = h1 | not tested | all 200 | — | OK (HTTP only) |
| `/tutorial/14-wasm-inference-simd128` | 200, title = h1 | not tested | all 200 | — | OK (HTTP only) |
| `/tutorial/28-s6-secrets-pico` | 200, title = h1 | not tested | all 200 | — | OK (HTTP only) |

The navigation has no separate "Secrets" page: the s6-secrets-pico demo was checked on its Space.
The first paragraph of every page says what the page is; the Router page opens with a callout
whose title is its verdict, then the disclosure, which reads as intended.

Sizes spot-checked against `site/public/models/manifest.json` (bytes) and the catalogue
(`site/.vitepress/theme/lib/catalogue.ts`): every tagline size in `hf/spaces.json` is the
manifest's byte count in decimal units (s6-route 3,692,224 B = 3.7 MB, s6-secrets-pico 250,272 B
= 250 KB, s6-ear-nano 899 KB, s6-ear2-nano 632 KB, s6-cast-pico 143 KB, s6-cast2-pico 149 KB,
s6-sleep3-nano 752 KB, s6-nano 715 KB, s6-micro 2.4 MB, s3-pico 251 KB, s1-pico 545 KB, s2-nano
702 KB). The numbers printed by most Space demos do not agree (defect 1).

## The Spaces

State from the Hub API (`runtime.stage`) for all 18: **RUNNING**, no build error, none sleeping.
Every direct static URL `https://maxgfr-<model>.static.hf.space/index.html` answered 200.

| Space | State | Verdict shown first | Files load | Demo exercised, number recorded | Console | OK or defect |
|---|---|---|---|---|---|---|
| s6-nano | Running | No banner; "passed" only in the third note (defect 3) | Chrome: all 200 (bundle, wasm, `s6-nano.gguf`) | Flappy, AI: score 3 after 8 s, 2,000 µs per decision (median), 61 decisions/s | clean | Defects 1, 3 |
| s6-micro | Running | No banner; "passed" in the tagline's second sentence (defect 3) | Chrome: all 200 | Flappy, AI: score 3 after 10 s, 8,700 µs per decision; the page says it decides every 2 frames on this device | clean | Defects 1, 3 |
| s6-micro-v1 | Running | Yes: failed banner, agrees with spaces.json and card | Chrome: all 200 | Flappy, AI: score 4 after 10 s, 9,000 µs per decision | clean | Defects 1, 4 |
| s6-secrets-pico | Running | Tagline: "passes … on a synthetic bench and fails on real files" (no banner) | Chrome: all 200 | Preset "placeholder": 4 answers, all agree with the expected labels; 3.00 ms for the text, warm median 500 µs per decision; prints "244 KB" | clean | Defects 1, 3 |
| s6-secrets-verify-pico | Running | Yes: failed banner (G1) | Chrome: all 200, dictionary included | Preset "JSON config": 2 candidates, both accepted as real at τ_f1 = 0.2598 (P = 0.4492 and 0.3413; the page explains why fake values pass), median 6.10 ms per candidate; prints "244 KB" | clean | Defects 1, 4 |
| s6-ear2-nano | Running | Tagline: "It passed its pre-registered gate" (no banner) | Chrome: all 200 | Self-test without a microphone: events listed, fallback timer ends the turn after 2.5 s, 2.10 ms per decision; prints "617 KB" | clean | Defects 1, 3 |
| s6-ear-nano | Running | Yes: failed banner | Chrome: all 200, sample `yes.wav` 200 | Sample clip "yes": word "yes" at 98 %, turn "still talking" 73 %; prints "878 KB" | clean | Defects 1, 4 |
| s6-cast2-pico | Running | Yes: failed banner | Chrome: all 200 (`tinycast.gguf`, then `s6-cast2-pico.gguf` once ticked) | Trend + seasons + noise, 64 steps, backtest: s6-cast2-pico MASE 0.693, 9.9 ms; TinyCast 0.374, 48.2 ms; seasonal naive 1.054 | clean | Defect 4 |
| s6-cast-pico | Running | Yes: failed banner | Chrome: all 200 | Same series: "s6-cast" MASE 0.639, 8.3 ms; TinyCast 0.374, 36.8 ms; prints "140 KB" and TinyCast "582 KB" | clean | Defects 1, 4 |
| s6-route | Running | Tagline: "It passed its pre-registered gate on fresh slates" (no banner) | Chrome: all 200 (`s6-route-embed.gguf` on load) | Loaded; preset clicked when the browser connection was lost: no number recorded | clean at load | Defects 1, 3; demo not tested |
| s6-route-nano | Running | HTTP: failed notice in `space.json` | HTTP: page 200 | not tested | not tested | Defect 1 (bundle) |
| s6-sleep3-nano | Running | HTTP: failed notice in `space.json` | HTTP: page 200 | not tested | not tested | not tested in Chrome |
| s5-lite | Running | No verdict in the header (defect 3) | HTTP: page 200 | not tested | not tested | Defects 1, 3 |
| s4-micro | Running | No banner: `space.json` has `gate: null` although the card says it failed (defect 2) | HTTP: page 200 | not tested | not tested | Defects 1, 2 |
| s3-pico | Running | No verdict (ungated model) | HTTP: page 200 | not tested | not tested | Defect 1 (bundle) |
| s1-pico | Running | No verdict (ungated model) | HTTP: page 200 | not tested | not tested | Defect 1 (bundle) |
| s2-nano | Running | No verdict (ungated model) | HTTP: page 200 | not tested | not tested | Defect 1 (bundle) |
| s2-nano-doom | Running | No verdict (ungated model) | HTTP: page 200 | not tested | not tested | Defect 1 (bundle) |

## Model cards and data sets

Every `https://huggingface.co/maxgfr/<model>` answered 200. For every card, the published
`README.md` (front matter removed) was compared with `hf/cards/<model>.md` (front matter removed,
`{{USER}}` read as `maxgfr`): the first 300 characters are the same for all 18, and so is the
whole body.

| Model | Page | Published card |
|---|---|---|
| s1-pico | 200 | current |
| s2-nano | 200 | current |
| s2-nano-doom | 200 | current |
| s3-pico | 200 | current |
| s4-micro | 200 | current |
| s5-lite | 200 | current |
| s6-cast-pico | 200 | current |
| s6-cast2-pico | 200 | current |
| s6-ear-nano | 200 | current |
| s6-ear2-nano | 200 | current |
| s6-micro | 200 | current |
| s6-micro-v1 | 200 | current |
| s6-nano | 200 | current |
| s6-route | 200 | current |
| s6-route-nano | 200 | current |
| s6-secrets-pico | 200 | current |
| s6-secrets-verify-pico | 200 | current |
| s6-sleep3-nano | 200 | current |

| Data set | Page | `is-valid` |
|---|---|---|
| skycmd-cast-synth, skycmd-commands, skycmd-decisions, skycmd-ear, skycmd-evals, skycmd-games, skycmd-glm-labels, skycmd-lfm3-math, skycmd-numeric, skycmd-route, skycmd-secrets, skycmd-series, skycmd-text, skycmd-trajectories, skycmd-zero-shot | 200 each | preview, viewer, search, filter and statistics all true, for each of the 15 |

*Note of 19/11/2026: `skycmd-lfm3-math` was renamed `skycmd-math` on the Hub on 19/11/2026; the old address redirects. The table above records the name as checked on 18/11.*

## Defects

1. **Medium. 16 of the 18 Spaces serve a front-end built before the size fix of 17 November
   (commit 3cd9a35), so their demos print sizes that disagree with their own tagline and card.**
   URLs: every Space except s6-cast2-pico and s6-sleep3-nano (for example
   <https://maxgfr-s6-ear2-nano.static.hf.space/index.html>). What I did: ran each demo in Chrome,
   then searched each Space's served `assets/index-*.js` for a byte count divided by 1024 and
   labelled "KB". What I saw: the demos print kibibytes as "KB": s6-secrets-pico and
   s6-secrets-verify-pico "244 KB" (tagline 250 KB), s6-ear2-nano "617 KB" (632 KB), s6-ear-nano
   "878 KB" (899 KB), s6-cast-pico "140 KB" (143 KB) and TinyCast "582 KB" where the s6-cast2-pico
   Space prints "596 KB" for the same file; the s6-cast-pico Space also labels its model "s6-cast"
   where its own `models/forecast.json` says "s6-cast-pico". The 1024-and-"KB" formatter is in the
   bundle of all 16 and absent from the two Spaces published after 3cd9a35. Expected: one decimal
   unit everywhere, as `site/.vitepress/theme/lib/size.ts` now does. Change needed: republish the
   16 Spaces' front-end (`scripts/hf-publish.py`, which ships the build of `site/space/`); no source
   change.
2. **Medium. s4-micro failed its quality gate, but its Space does not show the failed notice.**
   URL: <https://maxgfr-s4-micro.static.hf.space/index.html> (its `space.json`). What I did: read
   the Space's `space.json` and the repository's manifest. What I saw: `"gate": null`,
   `"gate_note": null`, `"gate_url": null`, so the red notice that every other failed model's Space
   opens with is absent; the verdict is the last sentence of the tagline. The card ("failed its
   quality gate"), the catalogue (`status: 'failed'`) and the tagline all say it failed. Expected:
   the failed notice first, as on the other failed Spaces. File: `site/public/models/manifest.json`
   (the s4-micro entry's `gate`, `gate_note`, `gate_url`), then a republish.
3. **Low. On the Spaces of models that passed, the verdict is not the first thing shown.** URLs:
   s6-nano, s6-micro, s6-secrets-pico, s6-ear2-nano, s6-route, s5-lite Spaces. What I did: read the
   header of each. What I saw: only a failed gate gets a notice (`site/space/App.vue`, line 117,
   `v-if="cfg.gate === 'failed'"`); on s6-nano "passed" first appears in the third note, on
   s6-micro, s6-ear2-nano and s6-route in the tagline's second sentence, on s5-lite nowhere in the
   header. Their cards open with the verdict ("Passed its pre-registered gate, every condition.",
   "It passed its gate on a second look.", and so on). Expected: the verdict first, as on the
   cards. Files: `site/space/App.vue` (a notice for `gate === 'passed'`) or the taglines in
   `hf/spaces.json`.
4. **Low. A missing space after the bold notice and before "The result".** URLs: the s6-micro-v1,
   s6-ear-nano, s6-cast-pico, s6-cast2-pico and s6-secrets-verify-pico Spaces. What I saw, on screen
   and in the page text: "published as a result, not for use.It missed its calibration guard …
   at least 0.45).The result." Expected: "not for use. It missed … 0.45). The result." The
   s6-cast2-pico Space carries the current bundle and shows it too, so republishing does not fix
   it; the leading space inside `<template v-if="cfg.gate_note"> {{ cfg.gate_note }}.</template>`
   is dropped by the template compiler. Files: `site/space/App.vue` lines 117 to 121, and the same
   pattern in `site/.vitepress/theme/components/PlayGame.vue` (424 to 426),
   `ForecastView.vue` (345), `EarDemo.vue` (316) and `LiveDecide.vue` (362), which suggests the site's
   demo notices show it as well (not seen on the site: see "Not tested").

No defect found in the model cards (all current), the data sets (all valid), the Space states
(all running), the files the Spaces load (no 404 on the ten opened in Chrome) or the site's links
(108 of 108 answer 200).

## Not tested

- **Every site page in the browser**: console errors and unhandled rejections, images that fail
  at run time, and every site demo (Live, Play, Router, Forecast, Listen, Secrets verifier, Memory
  with its 200 episodes, Bench). The browser connection was lost before the site part began.
- **Eight Spaces in the browser**: s6-route-nano, s6-sleep3-nano, s5-lite (with s4-micro and its
  proofs), s4-micro, s3-pico, s1-pico, s2-nano, s2-nano-doom, and the s6-route demo beyond its
  load. Their state, static URL and `space.json` were checked over HTTP only.
- **The microphone paths**: the "Use the microphone" mode of s6-ear2-nano and s6-ear-nano (only
  the self-test and a sample clip were run).
- **The Space pages on huggingface.co in the browser**: their state comes from the Hub API.
- **Safari, Firefox, and mobile layouts**.
- **The other 25 tutorial chapters** (three were fetched: 03, 14 and 28).
