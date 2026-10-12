<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { live } from '../lib/store'
import { formatInt } from '../lib/wasm'

const tiny = computed(() =>
  live.status === 'model'
    ? `${live.model}: ${formatInt(live.params)} parameters in a ${formatInt(live.bytes / 1024)} KB GGUF file.`
    : 's3-pico: 445K parameters in a 251 KB ternary GGUF file, and it plays all four games at oracle level.',
)
const fast = computed(() =>
  live.medianUs > 0 && live.status === 'model'
    ? `${formatInt(live.medianUs)} µs per decision right now, in WASM with SIMD128, on the main thread.`
    : 'One forward pass per question, no token generation: well under a millisecond per decision in WASM with SIMD128.',
)
</script>

<template>
  <div class="cards">
    <a class="card" :href="withBase('/tutorial/12-size-ladder')">
      <div class="icon" aria-hidden="true">◆</div>
      <h3>Tiny</h3>
      <p>{{ tiny }}</p>
    </a>
    <a class="card" :href="withBase('/bench/')">
      <div class="icon" aria-hidden="true">⚡</div>
      <h3>Fast</h3>
      <p>{{ fast }}</p>
    </a>
    <a class="card" :href="withBase('/tutorial/03-typed-decisions')">
      <div class="icon" aria-hidden="true">{ }</div>
      <h3>Typed</h3>
      <p>
        Questions are <code>choice</code>, <code>score</code> or <code>noul</code> (probabilistic yes/no). Every option
        gets a calibrated probability, <code>unknown</code> is a real answer, and a coded safety reflex has the last word.
      </p>
    </a>
  </div>
</template>

<style scoped>
.cards {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin: 28px 0 8px;
}
@media (max-width: 860px) {
  .cards {
    grid-template-columns: 1fr;
  }
}
.card {
  display: block;
  padding: 22px;
  border-radius: 16px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: inherit;
  text-decoration: none !important;
  transition: border-color 0.2s, transform 0.2s;
}
.card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}
.icon {
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  font-weight: 800;
  font-size: 16px;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
h3 {
  margin: 14px 0 6px !important;
  padding: 0 !important;
  border: 0 !important;
  font-size: 18px !important;
}
p {
  margin: 0 !important;
  font-size: 14px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}
</style>
