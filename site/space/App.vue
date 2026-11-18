<script setup lang="ts">
// A Hugging Face Space for one skycmd model. `space.json`, shipped next to the page, names the
// model and the demos to show; the model file and the wasm module are part of the Space, and
// everything runs in the browser.
import { computed, onMounted, ref } from 'vue'
import EarDemo from '../.vitepress/theme/components/EarDemo.vue'
import ForecastView from '../.vitepress/theme/components/ForecastView.vue'
import GateNotice from '../.vitepress/theme/components/GateNotice.vue'
import LiveDecide from '../.vitepress/theme/components/LiveDecide.vue'
import LiveProof from '../.vitepress/theme/components/LiveProof.vue'
import MemoryDemo from '../.vitepress/theme/components/MemoryDemo.vue'
import PlayGame from '../.vitepress/theme/components/PlayGame.vue'
import RouterDemo from '../.vitepress/theme/components/RouterDemo.vue'
import SecretsVerifyDemo from '../.vitepress/theme/components/SecretsVerifyDemo.vue'
import TurnDemo from '../.vitepress/theme/components/TurnDemo.vue'
import { gateNotice } from '../.vitepress/theme/lib/gate'
import { loadManifest } from '../.vitepress/theme/lib/wasm'

interface SpaceConfig {
  model: string
  title: string
  tagline: string
  /** `secrets`: the API-key demo of s6-secrets-pico (manifest `task: secrets`). `ear`: the Listen
   * demo (manifest `task: ear`). `forecast`: the Forecast demo (`models/forecast.json`). `router`:
   * the tool router of the Router page (manifest `task: router`, two files fused by `ToolRouter`).
   * `turn`: the end-of-turn demo of the Listen page (s6-ear2-nano through `TurnDetector`).
   * `router-v2`: the same router demo showing the s6-route-nano file alone (it failed its gate; the
   * Space also carries s6-route's two files, which `ToolRouter` needs). `secrets-verify`: the
   * Secrets verifier demo (s6-secrets-verify-pico through `SecretVerifier`, with its word list).
   * `memory`: the Memory demo (s6-sleep3-nano and its control, two files of one manifest entry
   * `task: memory`, through `RecallProbe`; it failed its gate). */
  kind: 'decision' | 'game' | 'secrets' | 'secrets-verify' | 'ear' | 'forecast' | 'router' | 'router-v2' | 'turn' | 'memory'
  proofs?: boolean
  model_repo: string
  notes?: string[]
  /** Game shown first (game Spaces): flappy, rescue, dodge or doom. */
  game?: string
  /** The model's pre-registered gate (from the manifest). The Space of a model with a gate opens
   * with a notice that gives its verdict, passed or failed, with the numbers and the link to the
   * result; a model published without a gate has none. */
  gate?: 'passed' | 'failed'
  gate_note?: string
  gate_url?: string
}

const cfg = ref<SpaceConfig | null>(null)
const failed = ref('')
/** A game Space whose model also answers free-text questions (manifest `task: general`, s6-nano). */
const general = ref(false)
const SITE = 'https://maxgfr.github.io/skycmd'
/** The verdict notice, first in the header (lib/gate.ts). */
const verdict = computed(() => gateNotice(cfg.value))

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
        <GateNotice v-if="verdict" :notice="verdict" class="space-verdict" />
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
      <template v-else-if="cfg.kind === 'secrets-verify'">
        <section>
          <h2>Is this value a real credential?</h2>
          <p>
            s6-secrets-verify-pico failed gate G1: on the CredData test, 221 real repositories, its line F1 is 0.8561,
            where the gate asked for the best rule scanner plus 0.15, 0.7332 + 0.15 = 0.8832 (Betterleaks). The six
            other gates held. It is above every rule scanner tested, calibrated on real candidates (ECE 0.0073), and it
            accepts a candidate in 0.35% of 8,825 clean real files; since version 1.9.0, secretgate offers it as an
            experimental option, off by default, where it can only add redactions.
          </p>
          <p>
            A rule proposes a candidate, a value with its name, its line, the lines around it and its file, and the
            model judges that one value. This demo does not run secretgate's rule engine: the real generator is
            secretgate's, frozen as sg-cand-1, and this demo proposes with one simple rule so that the model's
            judgement can be seen in a browser. The presets were written for this demo, and every value in them is
            fake.
          </p>
          <SecretsVerifyDemo />
          <p class="space-note">
            The three thresholds are stored in the file, fixed on 1,589 CredData validation candidates before the test
            was opened: τ_f1 (F1-optimal, every test reading used it), τ_f05 (fewer alarms) and τ_f2 (fewer misses).
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
      <template v-else-if="cfg.kind === 'router-v2'">
        <section>
          <h2>The s6-route-nano file alone, which failed its gate</h2>
          <p>
            This Space runs the s6-route-nano file alone. It failed its pre-registered gate: 0.8395 against 0.8500
            for BM25 on 2,000 slates of tools never seen in training, and 38.9 ms per request in WebAssembly against
            a bar of 2 ms. The pair that passed, this file fused with a static embedding, is the Space
            <a href="https://huggingface.co/spaces/maxgfr/s6-route">s6-route</a>.
          </p>
          <p>
            Up to 7 tools, one per line as <code>name: description</code>, and a request. The bars are the
            probabilities of the s6-route-nano file alone, which also reads each tool's BM25 score against the
            request. The engine's router loads the two files of s6-route, so under each bar are the pair's fused
            probability and the embedding part's, and the line under the answer says what the pair answers. The
            presets were written for this demo, not taken from any dataset, and are kept whatever the model answers:
            on them the file alone is often wrong where the pair is right.
          </p>
          <RouterDemo prefer="s6-route" part="model" />
          <p class="space-note">
            “Mask names” renames every tool <code>tool_&lt;k&gt;</code>: on its test the file alone was at chance
            there (0.1050), where BM25 kept 0.8445.
          </p>
        </section>
      </template>
      <template v-else-if="cfg.kind === 'memory'">
        <section>
          <h2>Does the model remember a colour it saw once?</h2>
          <p>
            s6-sleep3-nano failed its pre-registered gate. Its memory is used: on the test episodes of the
            <code>recall</code> probe it is right on 0.4892 of the questions, against 0.2440 for the same training
            without a memory. But the memory fades with the delay, four sleep passes add +0.0030 where a gain above
            zero was asked, and it plays Flappy (15.87 against 71.05) and Dodge (5.89 against 7.94) much worse than the
            control. It is a research result, not a part to use.
          </p>
          <p>
            A door is shown its colour on one frame; some frames later the door is named, and the model must give
            the colour back. Below, one episode frame by frame, then 200 episodes with the memory on, with the memory
            zeroed and with the control file, next to the test reading.
          </p>
          <MemoryDemo />
          <p class="space-note">
            The episodes come from the evaluation's own generator and are scored by the game, as on the test; the 200
            here are other seeds (the validation split), so they differ from the test by sampling noise.
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
            module as a reference:
            <template v-if="cfg.model === 's6-cast2-pico'"
              >s6-cast2-pico was to beat FLAIR on GIFT-Eval (its gate G1) and did not; it stays behind TinyCast.</template
            >
            <template v-else>s6-cast-pico was to beat FLAIR (its gate) and TinyCast (its goal), and did neither.</template>
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
        {{ cfg.kind === 'router' || cfg.kind === 'router-v2' || cfg.kind === 'memory' ? 'files' : 'file' }} of this Space{{
          cfg.kind === 'secrets-verify' ? ', with its word list' : ''
        }}. Nothing is sent to a server.
      </footer>
    </template>
    <p v-else class="space-note">Loading…</p>
  </main>
</template>
