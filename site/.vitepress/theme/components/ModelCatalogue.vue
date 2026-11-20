<script setup lang="ts">
// Views of the Models catalogue (lib/catalogue.ts):
//   <ModelCatalogue group="works" />   the cards of one group (/models/)
//   <ModelCatalogue legend />          what each status means (/models/)
//   <ModelCatalogue home />            "What works" on the home page
//   <ModelCatalogue lessons />         the lessons of failed gates on the home page
import { computed } from 'vue'
import { withBase } from 'vitepress'
import ModelCard from './ModelCard.vue'
import StatusBadge from './StatusBadge.vue'
import { CATALOGUE, STATUS, entriesOf, type GroupId, type Status } from '../lib/catalogue'

const props = defineProps<{ group?: GroupId; home?: boolean; lessons?: boolean; legend?: boolean }>()

const cards = computed(() =>
  props.home ? CATALOGUE.filter((e) => e.home) : props.group ? entriesOf(props.group) : [],
)
const lessonList = computed(() => CATALOGUE.filter((e) => e.lesson))
const statuses = Object.keys(STATUS) as Status[]
</script>

<template>
  <dl v-if="legend" class="legend">
    <div v-for="s in statuses" :key="s" class="legend-row">
      <dt><StatusBadge :status="s" /></dt>
      <dd>{{ STATUS[s].meaning }}</dd>
    </div>
  </dl>

  <ul v-else-if="lessons" class="lessons">
    <li v-for="e in lessonList" :key="e.id">
      <a :href="withBase(`/models/#${e.id}`)" class="lesson-name">{{ e.lessonName ?? e.name }}</a>
      {{ e.lesson }}
    </li>
  </ul>

  <div v-else class="grid" :class="{ home }">
    <ModelCard v-for="e in cards" :key="e.id" :entry="e" :compact="home" />
  </div>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
  gap: 16px;
  margin: 20px 0 8px;
}
.grid.home {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));
}
.legend {
  display: grid;
  gap: 8px;
  margin: 16px 0;
}
.legend-row {
  display: grid;
  grid-template-columns: minmax(0, 15rem) minmax(0, 1fr);
  gap: 4px 16px;
  align-items: baseline;
}
@media (max-width: 640px) {
  .legend-row {
    grid-template-columns: 1fr;
  }
}
.legend dt,
.legend dd {
  margin: 0;
}
.legend dd {
  font-size: 14px;
  line-height: 1.55;
  color: var(--vp-c-text-2);
}
.lessons {
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}
.vp-doc .lessons li,
.lessons li {
  margin: 0;
  padding: 12px 0;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 15px;
  line-height: 1.55;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-2);
}
.lessons li:last-child {
  border-bottom: 1px solid var(--vp-c-divider);
}
.vp-doc .lesson-name,
.lesson-name {
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  font-weight: 650;
}
</style>
