<script setup lang="ts">
// A Hugging Face Space for one skycmd model. `space.json`, shipped next to the page, names the
// model and the demos to show; the model file and the wasm module are part of the Space, and
// everything runs in the browser.
import { onMounted, ref } from 'vue'
import LiveDecide from '../.vitepress/theme/components/LiveDecide.vue'
import LiveProof from '../.vitepress/theme/components/LiveProof.vue'
import PlayGame from '../.vitepress/theme/components/PlayGame.vue'
import { loadManifest } from '../.vitepress/theme/lib/wasm'

interface SpaceConfig {
  model: string
  title: string
  tagline: string
  /** `secrets`: the API-key demo of s6-secrets-pico (manifest `task: secrets`). */
  kind: 'decision' | 'game' | 'secrets'
  proofs?: boolean
  model_repo: string
  notes?: string[]
  /** Game shown first (game Spaces): flappy, rescue, dodge or doom. */
  game?: string
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
        file of this Space. Nothing is sent to a server.
      </footer>
    </template>
    <p v-else class="space-note">Loading…</p>
  </main>
</template>
