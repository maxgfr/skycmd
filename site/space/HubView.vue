<script setup lang="ts">
// The global Space, maxgfr/skycmd: every model of the project in one place. The first screen
// lists the models under the groups of the site's Models catalogue (lib/catalogue.ts: the name,
// what the model is for, its size and its status badge). Picking one opens its demo exactly as its
// own Space shows it (SpaceView.vue with that Space's `space.json`), loaded on demand: the demo
// code is a chunk of its own, and the wasm module and the model files are fetched by the demo once
// it is open. The open model is in the URL hash (`#s6-route`), so the browser's back button and a
// link to one demo both work. While a demo is open, the manifest readers see only the models its
// own Space lists (lib/scope.ts); leaving it unmounts it, and each demo stops
// its own audio, timers and game loop when it is unmounted.
import { computed, defineAsyncComponent, h, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import StatusBadge from '../.vitepress/theme/components/StatusBadge.vue'
import { CATALOGUE, GROUPS } from '../.vitepress/theme/lib/catalogue'
import { setManifestScope } from '../.vitepress/theme/lib/scope'
import { hashOf, hubGroups, hubModels, spaceOfHash, type HubConfig, type HubSpace } from './hub'

const props = defineProps<{ cfg: HubConfig }>()

const SpaceView = defineAsyncComponent({
  loader: () => import('./SpaceView.vue'),
  loadingComponent: { render: () => h('p', { class: 'space-note' }, 'Loading…') },
  errorComponent: {
    props: { error: { type: Object, default: null } },
    setup: (p: { error: Error | null }) => () => h('p', { class: 'space-error' }, p.error?.message ?? 'The demo could not be loaded.'),
  },
  delay: 0,
})

const SITE = 'https://maxgfr.github.io/skycmd'
const groups = computed(() => hubGroups(props.cfg.spaces, CATALOGUE, GROUPS))
const universe = computed(() => hubModels(props.cfg.spaces))
/** The Space whose demo is open; null: the list. */
const open = shallowRef<HubSpace | null>(null)
const back = ref<HTMLAnchorElement | null>(null)
let first = true

/** Opens what the hash names: a model's demo, or the list. */
function route() {
  const s = spaceOfHash(window.location.hash, props.cfg.spaces)
  if (!first && s === open.value) return
  const left = open.value
  setManifestScope(s ? s.models : null, universe.value)
  // as a fresh visit of that model's Space: its first game, or none (PlayGame reads `?game=`); on
  // the first load a `?game=` given in the address is kept, as in a model's own Space
  const url = new URL(window.location.href)
  if (!(first && url.searchParams.has('game'))) {
    if (s?.config.game) url.searchParams.set('game', s.config.game)
    else url.searchParams.delete('game')
    if (url.href !== window.location.href) window.history.replaceState(window.history.state, '', url)
  }
  first = false
  open.value = s
  document.title = s ? `${s.config.title} · skycmd` : `${props.cfg.title} · every demo`
  void nextTick(() => {
    if (s) {
      window.scrollTo(0, 0)
      back.value?.focus({ preventScroll: true })
    } else if (left) {
      // back on the list, at the model just left
      document.getElementById(`hub-${left.config.model}`)?.focus()
    }
  })
}

onMounted(() => {
  window.addEventListener('hashchange', route)
  route()
})

onBeforeUnmount(() => {
  window.removeEventListener('hashchange', route)
  setManifestScope(null)
})
</script>

<template>
  <template v-if="open">
    <nav class="hub-bar" aria-label="skycmd demos">
      <a ref="back" href="#">Back to all models</a>
      <a :href="open.space_url">This demo in its own Space</a>
    </nav>
    <SpaceView :key="open.config.model" :cfg="open.config" />
  </template>
  <template v-else>
    <header>
      <h1>{{ cfg.title }}</h1>
      <p class="tagline">{{ cfg.tagline }}</p>
      <p class="links">
        <a :href="SITE">Project site</a> ·
        <a :href="`${SITE}/models/`">Models</a> ·
        <a :href="`${SITE}/results/`">Results</a> ·
        <a href="https://github.com/maxgfr/skycmd">Source (Rust)</a>
      </p>
      <ul v-if="cfg.notes?.length" class="notes">
        <li v-for="n in cfg.notes" :key="n">{{ n }}</li>
      </ul>
    </header>
    <section v-for="g in groups" :key="g.id" :aria-labelledby="`hub-group-${g.id}`">
      <h2 :id="`hub-group-${g.id}`">{{ g.title }}</h2>
      <ul class="hub-grid">
        <li v-for="it in g.items" :key="it.name" class="hub-card">
          <p class="hub-head">
            <a :id="`hub-${it.name}`" class="hub-name" :href="hashOf(it.space)">{{ it.name }}</a>
            <span v-if="it.card.size" class="hub-size">{{ it.card.size }}</span>
          </p>
          <p class="hub-status"><StatusBadge :status="it.card.status" /></p>
          <p class="hub-purpose">{{ it.card.purpose }}</p>
        </li>
      </ul>
    </section>
  </template>
</template>

<style scoped>
.hub-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  margin: 0 0 8px;
  font-size: 14px;
}
.hub-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
  gap: 12px;
  margin: 16px 0 0;
  padding: 0;
  list-style: none;
}
/* the whole card opens the demo: the name's link covers it */
.hub-card {
  position: relative;
  min-width: 0;
  padding: 14px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.2s;
}
.hub-card:hover,
.hub-card:focus-within {
  border-color: var(--vp-c-brand-1);
}
.hub-card p {
  margin: 0;
}
.hub-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.hub-name {
  font-family: var(--vp-font-family-mono);
  font-size: 16px;
  font-weight: 650;
  overflow-wrap: anywhere;
  text-decoration: none;
}
.hub-name::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
}
.hub-name:focus-visible {
  outline: none;
}
.hub-name:focus-visible::after {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
.hub-size {
  flex: none;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-2);
}
.hub-card .hub-status {
  margin-top: 8px;
  line-height: 1;
}
.hub-card .hub-purpose {
  margin-top: 10px;
  font-size: 14px;
  line-height: 1.5;
  color: var(--vp-c-text-2);
}
</style>
