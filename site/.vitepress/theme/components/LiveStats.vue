<script setup lang="ts">
// Live counters fed by the hero game.
import { computed } from 'vue'
import { live } from '../lib/store'
import { formatInt } from '../lib/wasm'

const hasModel = computed(() => live.status === 'model')
const us = computed(() => (live.medianUs > 0 ? formatInt(live.medianUs) : '—'))
const capacity = computed(() => (live.medianUs > 0 ? formatInt(1e6 / live.medianUs) : '—'))
const kb = computed(() => (hasModel.value ? formatInt(live.bytes / 1024) : '—'))
const params = computed(() => (hasModel.value ? formatInt(live.params) : '—'))
</script>

<template>
  <div class="live-stats" aria-live="polite">
    <div class="stat">
      <div class="value">{{ us }}<span class="unit">µs</span></div>
      <div class="label">per decision in WASM</div>
    </div>
    <div class="stat">
      <div class="value">{{ capacity }}</div>
      <div class="label">decisions/s capacity <span class="muted">({{ formatInt(live.decisionsPerS) }} taken)</span></div>
    </div>
    <div class="stat">
      <div class="value">{{ kb }}<span class="unit">KB</span></div>
      <div class="label">model file</div>
    </div>
    <div class="stat">
      <div class="value">{{ params }}</div>
      <div class="label">parameters</div>
    </div>
  </div>
  <p class="live-note">
    <template v-if="hasModel">
      Measured live on this device: {{ live.model }} answers two questions (action + danger) per frame, then the physics
      steps, all inside one wasm module.
    </template>
    <template v-else-if="live.status === 'oracle'">
      No model file is published yet, so the hero runs the game's oracle. Latency shows the oracle's lookahead search,
      not a model.
    </template>
    <template v-else>&nbsp;</template>
  </p>
</template>

<style scoped>
.live-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 8px 0 4px;
}
@media (max-width: 720px) {
  .live-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.stat {
  padding: 16px;
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
}
.value {
  font-size: 28px;
  font-weight: 750;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  line-height: 1.15;
  color: var(--vp-c-text-1);
}
.unit {
  margin-left: 4px;
  font-size: 15px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.label {
  margin-top: 4px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.muted {
  color: var(--vp-c-text-3);
}
.live-note {
  margin: 6px 2px 0;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
</style>
