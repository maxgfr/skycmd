<script setup lang="ts">
// A Hugging Face Space of skycmd. `space.json`, shipped next to the page, says what it shows: one
// model and its demos (SpaceView.vue; the model files and the wasm module are part of the Space),
// or, with kind `hub`, every model of the project, each demo opened on demand (HubView.vue: the
// global Space, maxgfr/skycmd). Each view is a chunk of its own, loaded once the kind is known.
// Everything runs in the browser.
import { onMounted, ref, shallowRef, type Component } from 'vue'
import type { HubConfig, SpaceConfig } from './hub'

const cfg = ref<SpaceConfig | HubConfig | null>(null)
const view = shallowRef<Component | null>(null)
const failed = ref('')

onMounted(async () => {
  if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) document.documentElement.classList.add('dark')
  try {
    const r = await fetch('./space.json')
    if (!r.ok) throw new Error(`space.json: HTTP ${r.status}`)
    const c: SpaceConfig | HubConfig = await r.json()
    if (c.kind === 'hub') {
      view.value = (await import('./HubView.vue')).default
    } else {
      // PlayGame reads the first game from `?game=`.
      const url = new URL(window.location.href)
      if (c.game && !url.searchParams.has('game')) {
        url.searchParams.set('game', c.game)
        window.history.replaceState(null, '', url)
      }
      view.value = (await import('./SpaceView.vue')).default
    }
    cfg.value = c
    // the global Space names the page itself: the list, or the model whose demo is open
    if (c.kind !== 'hub') document.title = `${c.title} · skycmd`
  } catch (e) {
    failed.value = (e as Error).message
  }
})
</script>

<template>
  <main class="space">
    <p v-if="failed" class="space-error">{{ failed }}</p>
    <component :is="view" v-else-if="cfg && view" :cfg="cfg" />
    <p v-else class="space-note">Loading…</p>
  </main>
</template>
