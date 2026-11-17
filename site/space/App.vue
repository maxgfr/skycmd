<script setup lang="ts">
// A Hugging Face Space for one skycmd model. `space.json`, shipped next to the page, names the
// model and the demos to show; the model file and the wasm module are part of the Space, and
// everything runs in the browser.
import { onMounted, ref } from 'vue'
import EarDemo from '../.vitepress/theme/components/EarDemo.vue'
import ForecastView from '../.vitepress/theme/components/ForecastView.vue'
import LiveDecide from '../.vitepress/theme/components/LiveDecide.vue'
import LiveProof from '../.vitepress/theme/components/LiveProof.vue'
import PlayGame from '../.vitepress/theme/components/PlayGame.vue'
import RouterDemo from '../.vitepress/theme/components/RouterDemo.vue'
import TurnDemo from '../.vitepress/theme/components/TurnDemo.vue'
import { loadManifest } from '../.vitepress/theme/lib/wasm'

interface SpaceConfig {
  model: string
  title: string
  tagline: string
  /** `secrets`: the API-key demo of s6-secrets-pico (manifest `task: secrets`). `ear`: the Listen
   * demo (manifest `task: ear`). `forecast`: the Forecast demo (`models/forecast.json`). `router`:
   * the tool router of the Router page (manifest `task: router`, two files fused by `ToolRouter`).
   * `turn`: the end-of-turn demo of the Listen page (s6-ear2-nano through `TurnDetector`). */
  kind: 'decision' | 'game' | 'secrets' | 'ear' | 'forecast' | 'router' | 'turn'
  proofs?: boolean
  model_repo: string
  notes?: string[]
  /** Game shown first (game Spaces): flappy, rescue, dodge or doom. */
  game?: string
  /** The model's pre-registered gate (from the manifest). A failed model's Space opens with a
   * notice that says so, with the numbers and the link to the result. */
  gate?: 'passed' | 'failed'
  gate_note?: string
  gate_url?: string
}

const cfg = ref<SpaceConfig | null>(null)
const failed = ref('')
/** A game Space whose model also answers free-text questions (manifest `task: general`, s6-nano). */
const general = ref(false)
const SITE = 'https://maxgfr.github.io/skycmd'

const DECISION = [
  {
    mode: 'playground',
    title: 'Playground',
    text: 'Any question, any options, any type: choice, score (levels from lowest to highest) or noul (a true/false statement).',
  },
  {
    mode: 'tools',
    title: 'Tool router for an agent',
    text: 'Describe the tools in one line each; the model picks the tool for the request, or no_tool.',
  },
  {
    mode: 'triage',
    title: 'Inbox and ticket triage',
    text: 'Your own categories, with an "unsure" way out.',
  },
  {
    mode: 'guard',
    title: 'Guardrail on a written rule',
    text: 'A noul question: "The message violates this rule: …". The model returns p(true).',
  },
] as const

const PROOFS = [
  {
    mode: 'guarantee',
    title: 'Keys and order cannot matter (guaranteed)',
    text: 'Every key moves to another definition, then the order is reversed: the iso model gives each definition the same probability, up to float rounding. s4-micro, which reads the keys, moves.',
  },
  {
    mode: 'paraphrase',
    title: 'Other words for the same options (learned)',
    text: 'Robustness to paraphrased definitions is trained, not guaranteed; measured on held-out paraphrases it is far from perfect.',
  },
  {
    mode: 'dutch',
    title: 'A Dutch book on the model’s own beliefs',
    text: 'For each option the model is asked "The correct answer is: x" and "… is not: x" separately. A coherent model has s + n = 1 and Σ s = 1.',
  },
] as const

onMounted(async () => {
  if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) document.documentElement.classList.add('dark')
  try {
    const r = await fetch('./space.json')
    if (!r.ok) throw new Error(`space.json: HTTP ${r.status}`)
    const c: SpaceConfig = await r.json()
    // PlayGame reads the first game from `?game=`.
    const url = new URL(window.location.href)
    if (c.game && !url.searchParams.has('game')) {
      url.searchParams.set('game', c.game)
      window.history.replaceState(null, '', url)
    }
    cfg.value = c
    document.title = `${c.title} · skycmd`
    general.value = (await loadManifest()).some((m) => m.name === c.model && m.task === 'general')
  } catch (e) {
    failed.value = (e as Error).message
  }
})
</script>

<template>
  <main class="space">
    <p v-if="failed" class="space-error">{{ failed }}</p>
    <template v-else-if="cfg">
      <header>
        <h1>{{ cfg.title }}</h1>
        <p v-if="cfg.gate === 'failed'" class="space-failed">
          <strong>Failed its pre-registered gate: published as a result, not for use.</strong>
          <template v-if="cfg.gate_note"> {{ cfg.gate_note }}.</template>
          <template v-if="cfg.gate_url"> <a :href="cfg.gate_url">The result</a>.</template>
        </p>
        <p class="tagline">{{ cfg.tagline }}</p>
        <p class="links">
          <a :href="cfg.model_repo">Model card</a> ·
          <a :href="SITE">Project site</a> ·
          <a :href="`${SITE}/results/`">Results</a> ·
          <a href="https://github.com/maxgfr/skycmd">Source (Rust)</a>
        </p>
        <ul v-if="cfg.notes?.length" class="notes">
          <li v-for="n in cfg.notes" :key="n">{{ n }}</li>
        </ul>
      </header>

      <template v-if="cfg.kind === 'decision'">
        <section v-for="d in DECISION" :key="d.mode">
          <h2>{{ d.title }}</h2>
          <p>{{ d.text }}</p>
          <LiveDecide :mode="d.mode" />
        </section>
        <template v-if="cfg.proofs">
          <section v-for="p in PROOFS" :key="p.mode">
            <h2>{{ p.title }}</h2>
            <p>{{ p.text }}</p>
            <LiveProof :mode="p.mode" />
          </section>
        </template>
      </template>
      <template v-else-if="cfg.kind === 'secrets'">
        <section>
          <h2>Is a secret exposed in this text?</h2>
          <p>
            Paste a short text: a config file, a <code>.env</code>, a few lines of code. The model answers the four
            questions it was trained for, each in one forward pass: is a usable secret exposed, which service issued
            it, what kind of value it is, and on which line. The presets are texts of the test bench taken by a fixed
            rule (the first of each kind, the first where the rule scanner is wrong, the first where the model is
            wrong), kept whatever the model answers. They come from the model's own generator: on real files the model
            flags most clean files, so this is a demo of a pre-registered result, not a scanner.
          </p>
          <LiveDecide mode="secrets" :prefer="cfg.model" />
          <p class="space-note">
            The bench is synthetic: our generator's formats, file templates and class shares, with fake keys. Beating
            the rule scanner there shows that the model reads formats and context in file types it never saw; it does
            not carry over to real files, where the model flags 59.8% of clean files at p ≥ 0.9. The line answer is weak (0.686 on the bench, and right on
            383 of the 1,010 texts that do expose a secret): do not rely on it.
          </p>
        </section>
      </template>
      <template v-else-if="cfg.kind === 'ear'">
        <section>
          <h2>Has the speaker finished? Which word was said?</h2>
          <p>
            Use the microphone or a sample clip: every 80 ms the model hears the last 10.24 s and answers two typed
            questions, the end of a turn and a word among the 12 classes of Speech Commands. The log-mel front-end runs
            in the same WebAssembly module, and the audio never leaves your browser. The model answers at full depth,
            where its gates were read, and says "not sure" below 0.5.
          </p>
          <EarDemo />
          <p class="space-note">
            Sample clips, CC-BY-4.0: “yes” from Speech Commands v0.02 (Warden, 2018), test split; the two French clips
            from <code>pipecat-ai/smart-turn-data-v3.2-test</code> (Pipecat), test split. The model never trains on
            either split.
          </p>
        </section>
      </template>
      <template v-else-if="cfg.kind === 'turn'">
        <section>
          <h2>Has the speaker finished their turn?</h2>
          <p>
            Use the microphone: an energy detector finds each pause, and at 0.2 s of silence the model reads the
            last 4 s of audio and says whether the turn is finished, in about 2.4 ms. While it answers "not
            finished" it is asked again every 0.2 s, and a fallback timer ends the turn after 2.5 s of silence
            whatever it says. The log-mel front end runs in the same WebAssembly module, and the audio never
            leaves your browser.
          </p>
          <TurnDemo />
          <p class="space-note">
            On real conversations (eot-bench) the model catches about half of the real ends of turn at its
            cautious threshold: the fallback timer is part of how it should be used.
          </p>
        </section>
      </template>
      <template v-else-if="cfg.kind === 'router'">
        <section>
          <h2>Pick a tool from a list you write</h2>
          <p>
            Up to 7 tools, one per line as <code>name: description</code>, and a request. Two parts score every option:
            the s6-route-nano file (which also reads each tool's BM25 score against the request) and potion-base-2M
            packed to 4 bits (Minish Lab, MIT). The router multiplies their probabilities option by option, raises the
            product to 0.81 and renormalises, adds <em>none of these tools fits the request</em>, and asks the user when
            the top probability is below τ_ask3. The presets were written for this demo, not taken from any dataset, and
            are kept whatever the router answers.
          </p>
          <RouterDemo :prefer="cfg.model" />
          <p class="space-note">
            “Mask names” renames every tool <code>tool_&lt;k&gt;</code>: on the fresh test the router fell to 0.2497
            there, where BM25 kept 0.8138. Use it only on tool lists whose names say what the tools do.
          </p>
        </section>
      </template>
      <template v-else-if="cfg.kind === 'forecast'">
        <section>
          <h2>Forecast a series</h2>
          <p>
            Pick a series: each forecaster returns nine quantiles, and the chart shows its median and its 10–90 % band.
            <strong>Backtest</strong> hides the last steps and scores the forecasts (MASE below 1 beats repeating the
            last season). TinyCast (RAWS Labs, Apache-2.0,
            <a href="./models/ref/TINYCAST-NOTICE.txt">NOTICE</a>), ported to Rust, runs in the same WebAssembly
            module as a reference: s6-cast-pico was to beat FLAIR (its gate) and TinyCast (its goal), and did neither.
          </p>
          <ForecastView />
        </section>
      </template>
      <template v-else>
        <section>
          <PlayGame />
        </section>
        <section v-if="general">
          <h2>Typed questions in free text</h2>
          <p>
            The same model answers the game and numeric questions it was trained for, with your own instruction,
            options and state. The presets are the first test record of each source and question type, kept whatever
            the model answers.
          </p>
          <LiveDecide mode="s6" :prefer="cfg.model" />
        </section>
      </template>

      <footer>
        Everything runs in your browser: a Rust inference engine compiled to WebAssembly (SIMD128) reads the GGUF
        {{ cfg.kind === 'router' ? 'files' : 'file' }} of this Space. Nothing is sent to a server.
      </footer>
    </template>
    <p v-else class="space-note">Loading…</p>
  </main>
</template>
