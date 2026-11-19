# Site and Spaces check in the browser, 19 November 2026

This check completes the "Not tested" list of [`docs/site-check-2026-11-18.md`](site-check-2026-11-18.md):
the site's demo pages, which the check of 18/11 could not open in a browser, and the eight Spaces it
checked over HTTP only. Nothing was changed: this file records what was tested and what was seen.

**How.** Chrome on macOS, on the owner's machine (Apple M5), on the deployed site
(<https://maxgfr.github.io/skycmd/>) and on the Spaces' direct static URLs
(`https://maxgfr-<name>.static.hf.space/index.html`). Each demo was exercised by clicking, and the
console was read on every page listed below: **clean on all of them** (no error, no unhandled
rejection).

## The site's demos

| Page | Demo exercised | What was seen | Verdict |
|---|---|---|---|
| `/play/` | Flappy Drone, the AI by default (s3-pico) | the game runs, the HUD updates | OK |
| `/live/` | preset "Database replication lag…" | s5-lite loads (20.7 MB) and prints the four probabilities: rollback 38.1 %, scale 18.7 %, wait 31.6 %, escalate 11.7 %; "The model abstains"; "Expected answer: escalate · the model disagrees". This decision 27.6 ms, first call 51.9 ms, engine relaxed SIMD | OK |
| `/router/` | three presets | "turn on the kitchen lights please" → `lights_on` 96.6 % (at or above τ_ask3 = 67.1 %); "it's freezing in here…" → `set_thermostat` 79.0 %; "put on some jazz in the living room" → the router asks the user (top option *none* 61.6 %, below τ_ask3), where the preset was written for `play_music`, and the page says so itself in red ("the router asks instead"). Model part 16.9 to 18.6 ms on the first requests | OK; the jazz preset shows a weakness of the model, stated by the page, not a page defect |
| `/forecast/` | "Trend + seasons + noise", 64 steps, Backtest | TinyCast MASE 0.374 (33.9 ms, 98 % of the points in the 10–90 % band); seasonal naive 1.054. The two s6-cast models are unticked by default and were not run | OK |
| `/ear/` (Listen) | s6-ear2-nano, "Self-test without a microphone"; s6-ear-nano, sample clip "French, end of turn (Smart Turn)" | s6-ear2-nano: events listed, "Finished by the fallback timer after 2.5 s of silence", p(finished) about 0.02 to 0.03, about 1.9 to 2.4 ms per decision. s6-ear-nano: "finished (58 %)", word "not sure (37 % < 50 %)" | OK (see the note below) |
| `/secrets-verify/` | preset "Python settings" | 3 candidates, 0 accepted as real at tau_f1 = 0.2598 (P(real) 0.1380 for the fake `SECRET_KEY`, 0.0557 for `os.environ.get`), about 12 ms per candidate | OK |
| `/memory/` | "Run 200 episodes" | read while the third pass, the control file, was at episode 56 of 200: memory on, N = 1, 0.487 over all delays (0.583 at 1–4 frames, 0.580 at 5–8, 0.450 at 9–16, 0.335 at 17–32); memory zeroed 0.264; the page's test columns read 0.4765 and 0.2451 | OK |
| `/bench/` | none | the page loads, no broken image; "Run benchmark in your browser" was not clicked | OK (load only) |

**Note on the Listen page's sample clips.** They need a real click: a scripted `element.click()`
leaves the audio waiting. That is the browser's autoplay rule, not a defect of the page.

## The eight Spaces not opened on 18/11

| Space | Demo exercised | What was seen | Verdict |
|---|---|---|---|
| s6-sleep3-nano | one episode, played "To the end" | the failed banner first; frame 64 answered green (0.741), right, 800 µs for the decision; 4 of 8 questions right on that episode | OK |
| s6-route-nano | preset "turn on the kitchen lights please" | the model alone picks `set_thermostat` 35.7 %; the page says "the model disagrees" (the pair s6-route answers `lights_on` 96.6 %); model part 43.4 ms | OK; a weakness of the failed model, shown as such |
| s5-lite | the Live preset "Database replication lag…" | the same four probabilities as on the site; 38.1 ms, first call 63.4 ms | OK |
| s4-micro | the same preset | the failed notice is now shown first: **defect 2 of 18/11 is fixed on the published Space**; 28.1 / 24.3 / 24.0 / 23.7 %, 33.0 ms | OK |
| s3-pico | Flappy Drone, the AI playing | score 3 after about 9 s, 60 to 61 decisions/s | OK |
| s1-pico | Flappy Drone, the AI playing | score 3 after about 9 s, 60 to 61 decisions/s | OK |
| s2-nano | Flappy Drone, the AI playing | score 3 after about 9 s, 60 to 61 decisions/s | OK |
| s2-nano-doom | the Doom-like game (the Space opens on `?game=doom`) | the game runs, the HUD and the minimap are drawn, the model chooses the actions | OK |

**The global Space** `maxgfr/skycmd`, opened on its direct static URL with a deep link
(`…/index.html#s6-cast2-pico`): it opens directly on
s6-cast2-pico, with "Back to all models", and shows the failed notice first. OK.

No new defect was found.

## Still not tested

- **The microphone paths** (the "Use the microphone" mode of s6-ear2-nano and s6-ear-nano), on the
  site and on the Spaces.
- **Safari and Firefox.** In Firefox the Listen page resamples the audio in JavaScript, a path that
  Chrome does not take; it has not been run.
- **The Space pages inside the huggingface.co frame**: only the direct static URLs were opened.
- **`/bench/`'s "Run benchmark in your browser" button.**
- **The console of the other site pages**: the articles and the tutorial chapters.
