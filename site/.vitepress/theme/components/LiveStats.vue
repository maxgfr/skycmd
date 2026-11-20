<script setup lang="ts">
// Live counters fed by the hero game.
import { computed } from 'vue'
import { live } from '../lib/store'
import { formatInt, useClockStep } from '../lib/wasm'
import { decisionLatency } from '../lib/latency'
import { sizeParts } from '../lib/size'

const hasModel = computed(() => live.status === 'model')
/** The browser clock's step in µs, measured on the client at mount (0 until then), and again
 * with each new decision while it is still unknown. */
const clockStep = useClockStep(() => [live.timedN, live.meanUs, live.decisionsPerS])
const latency = computed(() =>
  decisionLatency({ medianUs: live.medianUs, meanUs: live.meanUs, n: live.timedN }, clockStep.value),
)
const size = computed(() => (hasModel.value ? sizeParts(live.bytes) : { value: '—', unit: 'KB' }))
const params = computed(() => (hasModel.value ? formatInt(live.params) : '—'))
</script>

<template>
  <div class="live-stats" aria-live="polite">
    <div class="stat">
      <div class="value">{{ latency.us }}<span class="unit">µs</span></div>
      <div class="label">
        per decision in WASM
        <span v-if="latency.detail" class="muted">({{ latency.detail }})</span>
      </div>
    </div>
    <div class="stat">
      <div class="value">{{ latency.capacity }}</div>
      <div class="label">decisions/s capacity <span class="muted">({{ formatInt(live.decisionsPerS) }} taken)</span></div>
    </div>
    <div class="stat">
      <div class="value">{{ size.value }}<span class="unit">{{ size.unit }}</span></div>
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
