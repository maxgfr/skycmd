<script setup lang="ts">
// One model of the catalogue (lib/catalogue.ts). `compact` is the card of the home page strip.
import { computed } from 'vue'
import { withBase } from 'vitepress'
import StatusBadge from './StatusBadge.vue'
import { GROUPS, isExternal, type CatalogueEntry } from '../lib/catalogue'

const props = defineProps<{ entry: CatalogueEntry; compact?: boolean }>()

const href = (link: string) => (isExternal(link) ? link : withBase(link))
const verb = computed(() => GROUPS[props.entry.group].verb)
// on /models/ the heading's id is the card's anchor (and feeds the page outline)
const headingId = computed(() => (props.compact ? `home-${props.entry.id}` : props.entry.id))
const readLabel = computed(() => (props.entry.group === 'progress' ? 'Pre-registration' : 'Read'))
</script>

<template>
  <article class="mc" :class="{ compact }" :aria-labelledby="headingId">
    <header class="mc-head">
      <h3 :id="headingId" class="mc-name">{{ entry.name }}</h3>
      <span v-if="entry.size" class="mc-size">{{ entry.size }}</span>
    </header>
    <p class="mc-status">
      <StatusBadge :status="entry.status" />
    </p>
    <p class="mc-purpose">{{ entry.purpose }}</p>

    <p v-if="compact" class="mc-key">{{ entry.home }}</p>
    <template v-else>
      <dl class="mc-facts">
        <dt>{{ verb }}</dt>
        <dd>{{ entry.brings }}</dd>
        <template v-if="entry.caveat">
          <dt>Know before you use it</dt>
          <dd>{{ entry.caveat }}</dd>
        </template>
        <template v-if="entry.statusNote">
          <dt>Status</dt>
          <dd>{{ entry.statusNote }}</dd>
        </template>
      </dl>
    </template>

    <div v-if="entry.try || (!compact && (entry.get || entry.read))" class="mc-actions">
      <a v-if="entry.try" class="mc-action primary" :href="href(entry.try.link)">
        Try it<span class="sr-only">: {{ entry.name }} on {{ entry.try.label }}</span>
      </a>
      <template v-if="!compact">
        <a v-if="entry.get" class="mc-action" :href="entry.get">
          Get it<span class="sr-only">: {{ entry.name }} on Hugging Face</span>
        </a>
        <a v-if="entry.read" class="mc-action" :href="href(entry.read.link)">
          {{ readLabel }}<span class="sr-only">: {{ entry.read.label }}</span>
        </a>
      </template>
    </div>
  </article>
</template>

<style scoped>
.mc {
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 20px 22px 18px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  scroll-margin-top: calc(var(--vp-nav-height) + 16px);
}
.mc:target {
  border-color: var(--vp-c-brand-1);
}
.mc-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.vp-doc .mc-name,
.mc-name {
  margin: 0;
  padding: 0;
  border: 0;
  font-family: var(--vp-font-family-mono);
  font-size: 17px;
  font-weight: 650;
  line-height: 1.35;
  letter-spacing: -0.01em;
  overflow-wrap: anywhere;
  scroll-margin-top: calc(var(--vp-nav-height) + 24px);
}
.mc-size {
  flex: none;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-2);
}
.vp-doc .mc-status,
.mc-status {
  margin: 8px 0 0;
  line-height: 1;
}
.vp-doc .mc-purpose,
.mc-purpose {
  margin: 12px 0 0;
  font-size: 15px;
  line-height: 1.55;
  color: var(--vp-c-text-1);
}
.vp-doc .mc-key,
.mc-key {
  margin: 12px 0 0;
  padding-top: 12px;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 15px;
  font-weight: 650;
  line-height: 1.45;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-1);
}
.mc-facts {
  margin: 14px 0 0;
  padding-top: 12px;
  border-top: 1px solid var(--vp-c-divider);
}
.mc-facts dt {
  margin: 10px 0 2px;
  font-size: 13px;
  font-weight: 650;
  color: var(--vp-c-text-1);
}
.mc-facts dt:first-child {
  margin-top: 0;
}
.mc-facts dd {
  margin: 0;
  font-size: 14px;
  line-height: 1.55;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-2);
}
.mc-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: auto;
  padding-top: 16px;
}
.vp-doc .mc-action,
.mc-action {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  text-decoration: none;
  transition: border-color 0.2s, background-color 0.2s, color 0.2s;
}
.vp-doc .mc-action:hover,
.mc-action:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.vp-doc .mc-action.primary,
.mc-action.primary {
  border-color: var(--vp-button-brand-border);
  color: var(--vp-button-brand-text);
  background: var(--vp-button-brand-bg);
}
.vp-doc .mc-action.primary:hover,
.mc-action.primary:hover {
  color: var(--vp-button-brand-hover-text);
  background: var(--vp-button-brand-hover-bg);
}
.mc-action:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
.compact {
  padding: 18px 18px 16px;
}
/* Read by screen readers only: which model and which page a "Try it" link opens. */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}
.compact .mc-purpose {
  font-size: 14px;
  color: var(--vp-c-text-2);
}
</style>
