<script setup lang="ts">
// A Hugging Face Space for one skycmd model. `space.json`, shipped next to the page, names the
// model and the demos to show; the model file and the wasm module are part of the Space, and
// everything runs in the browser.
import { onMounted, ref } from 'vue'
import LiveDecide from '../.vitepress/theme/components/LiveDecide.vue'
import LiveProof from '../.vitepress/theme/components/LiveProof.vue'
import PlayGame from '../.vitepress/theme/components/PlayGame.vue'

interface SpaceConfig {
  model: string
  title: string
  tagline: string
  kind: 'decision' | 'game'
  proofs?: boolean
  model_repo: string
  notes?: string[]
}

const cfg = ref<SpaceConfig | null>(null)
const failed = ref('')
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
    cfg.value = await r.json()
    document.title = `${cfg.value!.title} · skycmd`
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
      <section v-else>
        <PlayGame />
      </section>

      <footer>
        Everything runs in your browser: a Rust inference engine compiled to WebAssembly (SIMD128) reads the GGUF
        file of this Space. Nothing is sent to a server.
      </footer>
    </template>
    <p v-else class="space-note">Loading…</p>
  </main>
</template>
