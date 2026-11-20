<script setup lang="ts">
// A model's status: a word first, a shape second, a colour third, so that none depends on colour.
import { computed } from 'vue'
import { STATUS, type Status } from '../lib/catalogue'

const props = defineProps<{ status: Status }>()
const meta = computed(() => STATUS[props.status])
</script>

<template>
  <span class="status" :class="`tone-${meta.tone}`">
    <svg class="status-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
      <template v-if="status === 'passed'">
        <path d="M3 8.6l3.2 3.1L13 4.6" />
      </template>
      <template v-else-if="status === 'second-look'">
        <circle cx="8" cy="8" r="6.4" stroke-dasharray="3 2" />
        <path d="M5 8.3l2.1 2 3.9-4.2" />
      </template>
      <template v-else-if="status === 'ungated'">
        <circle cx="8" cy="8" r="6.4" />
        <path d="M5 8h6" />
      </template>
      <template v-else-if="status === 'bench-only'">
        <circle cx="8" cy="8" r="6.4" />
        <path class="fill" d="M8 1.6a6.4 6.4 0 0 1 0 12.8z" />
      </template>
      <template v-else>
        <path d="M4 4l8 8M12 4l-8 8" />
      </template>
    </svg>
    <span>{{ meta.label }}</span>
  </span>
</template>

<style scoped>
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 10px 2px 8px;
  border-radius: 999px;
  font-size: 12.5px;
  font-weight: 600;
  line-height: 20px;
  white-space: nowrap;
}
.status-icon {
  flex: none;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.status-icon .fill {
  fill: currentColor;
  stroke: none;
}
.tone-success {
  color: var(--vp-c-success-1);
  background: var(--vp-c-success-soft);
}
.tone-neutral {
  color: var(--vp-c-text-2);
  background: var(--vp-c-default-soft);
}
.tone-warning {
  color: var(--vp-c-warning-1);
  background: var(--vp-c-warning-soft);
}
.tone-danger {
  color: var(--vp-c-danger-1);
  background: var(--vp-c-danger-soft);
}
.tone-info {
  color: var(--vp-c-indigo-1);
  background: var(--vp-c-indigo-soft);
}
</style>
