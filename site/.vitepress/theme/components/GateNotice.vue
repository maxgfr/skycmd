<script setup lang="ts">
// The verdict notice of a model with a pre-registered gate: red for a failed gate, green for a
// passed one. lib/gate.ts builds its three parts with their spaces; keep the template on one line,
// so that no whitespace is added or dropped between them (tests/gate.test.ts renders it).
import type { GateNotice } from '../lib/gate'

defineProps<{ notice: GateNotice }>()
</script>

<template>
  <p class="gate-notice" :class="`gate-${notice.tone}`"><strong>{{ notice.lead }}</strong>{{ notice.note }}<template v-if="notice.url"><a :href="notice.url">The result</a>.</template></p>
</template>

<style scoped>
.gate-notice.gate-failed {
  background: var(--vp-c-danger-soft);
}
.gate-notice.gate-passed {
  background: var(--vp-c-success-soft);
}
</style>
