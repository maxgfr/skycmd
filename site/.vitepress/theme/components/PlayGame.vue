<script setup lang="ts">
// The /play/ page: three games, AI or human control, model selector and a live HUD.
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { GameRunner, newHud, type GameKind, type Hud, type Mode } from '../lib/runner'
import {
  FAMILY_LABEL,
  familyOf,
  formatInt,
  formatKB,
  loadEngine,
  loadManifest,
  loadWasm,
  type Family,
  type ModelEntry,
  type Question,
  type SkyWeb,
  type WasmEngine,
} from '../lib/wasm'

const GAMES: { kind: GameKind; title: string; blurb: string }[] = [
  {
    kind: 'flappy',
    title: 'Flappy Drone',
    blurb: 'A decision every frame: thrust or glide through the gaps. The model also answers "will it crash within half a second if it keeps gliding?".',
  },
  {
    kind: 'rescue',
    title: 'Rescue Run',
    blurb: 'Rescue three hikers on a windy 12×12 grid and get home before the battery runs out. When the sensor drops out, "unknown" is the right answer.',
  },
  {
    kind: 'dodge',
    title: 'Obstacle Dodge',
    blurb: 'Pick a maneuver every 6 frames: over, around, brake or continue, and abort in front of an impassable obstacle. A coded reflex vetoes unsafe maneuvers.',
  },
]

const SECONDARY_LABEL: Record<GameKind, string> = {
  flappy: 'Danger: crash within 0.5 s if gliding',
  rescue: 'Battery risk (0 negligible … 4 critical)',
  dodge: 'Abort: no safe way past the obstacle',
}

const HELP: Record<GameKind, string> = {
  flappy: 'Tap the game, press Space or ↑ to thrust.',
  rescue: 'Arrow keys / WASD or swipe to move, H to hover, R to return home.',
  dodge: '↑ over, ← → around, ↓ brake, Space continue, A abort.',
}

const HUMAN_BUTTONS: Record<GameKind, { key: string; label: string }[]> = {
  flappy: [{ key: 'thrust', label: 'Thrust' }],
  rescue: [
    { key: 'north', label: '↑ N' },
    { key: 'west', label: '← W' },
    { key: 'east', label: 'E →' },
    { key: 'south', label: '↓ S' },
    { key: 'hover', label: 'Hover' },
    { key: 'return_home', label: 'Home' },
  ],
  dodge: [
    { key: 'over', label: 'Over' },
    { key: 'around', label: 'Around' },
    { key: 'brake', label: 'Brake' },
    { key: 'continue', label: 'Continue' },
    { key: 'abort', label: 'Abort' },
  ],
}

const ORACLE = '__oracle__'

/** s2 early-exit threshold choices: the calibrated τ of the file, fixed values, or full depth. */
const TAU_CHOICES: { value: string; label: string }[] = [
  { value: 'auto', label: 'calibrated' },
  { value: '0.7', label: 'τ = 0.7' },
  { value: '0.8', label: 'τ = 0.8' },
  { value: '0.9', label: 'τ = 0.9' },
  { value: '1', label: 'full depth' },
]

const kind = ref<GameKind>('flappy')
const mode = ref<Mode>('ai')
const veto = ref(true)
const models = ref<ModelEntry[]>([])
const selected = ref<string>(ORACLE)
const tau = ref('auto')
const status = ref('Loading the wasm module…')
const error = ref('')
const hud = ref<Hud>(newHud())
const questions = ref<Question[]>([])
const now = ref(0)
const canvas = ref<HTMLCanvasElement | null>(null)
const stage = ref<HTMLElement | null>(null)

let mod: SkyWeb | null = null
const engine = shallowRef<WasmEngine | null>(null)
let runner: GameRunner | null = null

const game = computed(() => GAMES.find((g) => g.kind === kind.value)!)
const selectedEntry = computed(() => models.value.find((m) => m.file === selected.value) ?? null)
const selectedFamily = computed<Family | null>(() => (selectedEntry.value ? familyOf(selectedEntry.value) : null))
const groups = computed(() =>
  (['s1', 's2'] as Family[])
    .map((f) => ({ family: f, label: FAMILY_LABEL[f], items: models.value.filter((m) => familyOf(m) === f) }))
    .filter((g) => g.items.length),
)
const info = computed(() => (mode.value === 'ai' && engine.value ? hud.value.info : null))
const layerCells = computed(() => {
  const i = info.value
  if (!i || i.max_layers == null) return []
  return Array.from({ length: i.max_layers }, (_, k) => k < (i.layers_used ?? 0))
})
const vetoFlash = computed(() => hud.value.vetoAt > 0 && now.value - hud.value.vetoAt < 700)
const primary = computed(() => questions.value[0])
const secondaryPct = computed(() => {
  const s = hud.value.secondary
  return s ? Math.round((100 * s.value) / (s.max || 1)) : 0
})

function bestKey(k: GameKind, m: Mode) {
  return `skycmd:best:${k}:${m}`
}

function makeRunner() {
  if (!mod || !canvas.value) return
  runner?.destroy()
  runner = new GameRunner(mod, canvas.value, {
    kind: kind.value,
    mode: mode.value,
    bestKey: bestKey(kind.value, mode.value),
    autoResetMs: 1000,
    onHud: (h) => {
      now.value = performance.now()
      hud.value = { ...h, probs: h.probs.slice() }
    },
  })
  runner.setVeto(veto.value)
  applyEngine()
  questions.value = runner.questions()
  hud.value = { ...runner.hud }
  runner.start()
}

function applyTau() {
  const e = engine.value
  if (!e || typeof e.setExitThreshold !== 'function') return
  e.setExitThreshold(tau.value === 'auto' ? undefined : Number(tau.value))
}

function applyEngine() {
  if (!runner) return
  const entry = selectedEntry.value
  applyTau()
  runner.setEngine(engine.value, entry ? entry.name : undefined)
}

async function selectModel(file: string) {
  error.value = ''
  if (file === ORACLE) {
    engine.value = null
    applyEngine()
    return
  }
  const entry = models.value.find((m) => m.file === file)
  if (!entry) return
  status.value = `Downloading ${entry.name} (${formatKB(entry.bytes)})…`
  try {
    engine.value = await loadEngine(entry)
    status.value = ''
  } catch (e) {
    engine.value = null
    selected.value = ORACLE
    error.value = `Could not load ${entry.file}: ${(e as Error).message}. Falling back to the oracle.`
    status.value = ''
  }
  applyEngine()
}

function setKind(k: GameKind) {
  if (k === kind.value) return
  kind.value = k
  const url = new URL(window.location.href)
  url.searchParams.set('game', k)
  window.history.replaceState(null, '', url)
}

function press(key: string) {
  if (mode.value !== 'human') mode.value = 'human'
  runner?.input(key)
}

watch(kind, () => makeRunner())
watch(mode, (m) => {
  // Each mode keeps its own high score.
  makeRunner()
  if (m === 'human') stage.value?.focus()
})
watch(veto, (v) => runner?.setVeto(v))
watch(selected, (f) => selectModel(f))
watch(tau, () => applyEngine())

const KEYMAP: Record<GameKind, Record<string, string>> = {
  flappy: { ' ': 'thrust', ArrowUp: 'thrust', w: 'thrust' },
  rescue: {
    ArrowUp: 'north',
    w: 'north',
    ArrowDown: 'south',
    s: 'south',
    ArrowRight: 'east',
    d: 'east',
    ArrowLeft: 'west',
    a: 'west',
    h: 'hover',
    r: 'return_home',
  },
  dodge: {
    ArrowUp: 'over',
    ArrowLeft: 'around',
    ArrowRight: 'around',
    ArrowDown: 'brake',
    ' ': 'continue',
    a: 'abort',
  },
}

function onKey(e: KeyboardEvent) {
  if (mode.value !== 'human') return
  const t = e.target as HTMLElement | null
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
  const key = KEYMAP[kind.value][e.key.length === 1 ? e.key.toLowerCase() : e.key]
  if (!key) return
  e.preventDefault()
  if (!e.repeat || kind.value !== 'flappy') runner?.input(key)
}

// Touch: tap = thrust (flappy), swipe = move (rescue).
let touchStart: { x: number; y: number } | null = null
function onPointerDown(e: PointerEvent) {
  touchStart = { x: e.clientX, y: e.clientY }
  if (mode.value === 'human' && kind.value === 'flappy') runner?.input('thrust')
}
function onPointerUp(e: PointerEvent) {
  if (!touchStart) return
  const dx = e.clientX - touchStart.x
  const dy = e.clientY - touchStart.y
  touchStart = null
  if (mode.value !== 'human' || kind.value === 'flappy') return
  if (Math.hypot(dx, dy) < 24) {
    if (runner && hud.value.done) runner.input('hover')
    return
  }
  if (kind.value === 'rescue') {
    runner?.input(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'east' : 'west') : dy > 0 ? 'south' : 'north')
  } else {
    runner?.input(Math.abs(dx) > Math.abs(dy) ? 'around' : dy < 0 ? 'over' : 'brake')
  }
}

onMounted(async () => {
  const q = new URLSearchParams(window.location.search).get('game')
  if (q === 'flappy' || q === 'rescue' || q === 'dodge') kind.value = q
  window.addEventListener('keydown', onKey)
  try {
    mod = await loadWasm()
  } catch (e) {
    status.value = ''
    error.value = `The wasm module could not be loaded (${(e as Error).message}). Build it with scripts/build-wasm.sh.`
    return
  }
  models.value = await loadManifest()
  status.value = ''
  makeRunner()
  if (models.value.length) selected.value = models.value[0].file
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  runner?.destroy()
  runner = null
})

function pct(p: number) {
  return `${(100 * p).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`
}
</script>

<template>
  <div class="play">
    <div class="tabs" role="tablist" aria-label="Games">
      <button
        v-for="g in GAMES"
        :key="g.kind"
        role="tab"
        type="button"
        class="tab"
        :class="{ active: g.kind === kind }"
        :aria-selected="g.kind === kind"
        @click="setKind(g.kind)"
      >
        {{ g.title }}
      </button>
    </div>
    <p class="blurb">{{ game.blurb }}</p>

    <div class="toolbar">
      <div class="seg" role="group" aria-label="Control">
        <button type="button" :class="{ on: mode === 'ai' }" @click="mode = 'ai'">AI</button>
        <button type="button" :class="{ on: mode === 'human' }" @click="mode = 'human'">Human</button>
      </div>
      <label class="field">
        <span>Model</span>
        <select v-model="selected" :disabled="mode === 'human'">
          <optgroup v-for="g in groups" :key="g.family" :label="g.label">
            <option v-for="m in g.items" :key="m.file" :value="m.file">
              {{ m.name }} · {{ formatInt(m.params) }} params · {{ formatKB(m.bytes) }}{{ m.weight_format ? ` · ${m.weight_format}` : '' }}
            </option>
          </optgroup>
          <option :value="ORACLE">oracle (no model loaded)</option>
        </select>
      </label>
      <label v-if="selectedFamily === 's2' && mode === 'ai'" class="field" title="Early exit: stop at the first exit whose top probability reaches τ">
        <span>Exit</span>
        <select v-model="tau">
          <option v-for="t in TAU_CHOICES" :key="t.value" :value="t.value">{{ t.label }}</option>
        </select>
      </label>
      <label v-if="kind !== 'flappy'" class="check">
        <input v-model="veto" type="checkbox" />
        <span>Safety reflex</span>
      </label>
      <button type="button" class="ghost" @click="runner?.reset()">Restart</button>
    </div>
    <p v-if="status" class="status">{{ status }}</p>
    <p v-if="error" class="error">{{ error }}</p>

    <div class="layout">
      <div
        ref="stage"
        class="stage"
        :class="kind"
        tabindex="0"
        @pointerdown="onPointerDown"
        @pointerup="onPointerUp"
        @pointercancel="touchStart = null"
      >
        <canvas ref="canvas" class="canvas" :aria-label="`${game.title} game`" role="img" />
        <div v-if="vetoFlash" :key="hud.vetoAt" class="veto">VETO</div>
        <div v-if="mode === 'human' && hud.done" class="overlay">Tap or press a key to restart</div>
      </div>

      <aside class="hud" aria-label="Decision HUD">
        <div class="policy">
          <span class="dot" :class="{ model: !!engine && mode === 'ai' }" />
          {{ mode === 'human' ? 'You are flying' : hud.policy }}
        </div>

        <div class="metrics">
          <div>
            <div class="mv">{{ mode === 'ai' && hud.medianUs > 0 ? formatInt(hud.medianUs) : '—' }}</div>
            <div class="ml">µs / decision (median)</div>
          </div>
          <div>
            <div class="mv">{{ mode === 'ai' ? formatInt(hud.decisionsPerS) : '—' }}</div>
            <div class="ml">decisions / s</div>
          </div>
          <div>
            <div class="mv">{{ hud.score }}</div>
            <div class="ml">score · best {{ hud.best }}</div>
          </div>
        </div>
        <div v-if="info" class="depth" aria-label="Model depth and caches">
          <template v-if="info.family === 's2'">
            <div class="drow">
              <span class="dl">layers used</span>
              <span class="cells" :title="`exit ${(info.exit ?? 0) + 1} of ${info.n_exits}`">
                <span v-for="(on, k) in layerCells" :key="k" class="cell" :class="{ on }" />
              </span>
              <span class="dv">{{ info.layers_used }}/{{ info.max_layers }} · avg {{ hud.meanLayers.toFixed(2) }}</span>
            </div>
            <div class="drow">
              <span class="dl">question cache</span>
              <span class="tag" :class="{ ok: info.static_hit }">{{ info.static_hit ? 'hit' : 'miss' }}</span>
              <span class="dv">{{ Math.round(100 * hud.staticHitRate) }}% hits</span>
            </div>
            <div class="drow">
              <span class="dl">state encoding</span>
              <span class="tag" :class="{ ok: info.state_reused }">{{ info.state_reused ? 'shared' : 'single' }}</span>
              <span class="dv">{{ info.state_reused ? '2 questions' : '1 question' }}</span>
            </div>
          </template>
          <template v-else>
            <div class="drow">
              <span class="dl">depth</span>
              <span />
              <span class="dv">all layers, full prompt</span>
            </div>
            <div class="drow">
              <span class="dl">prompt cache</span>
              <span class="tag" :class="{ ok: info.static_hit }">{{ info.static_hit ? 'hit' : 'miss' }}</span>
              <span class="dv">tokens only</span>
            </div>
          </template>
        </div>
        <p v-if="mode === 'ai' && hud.decideEvery > 1" class="throttle">
          This model is slower than the frame budget on this device: it decides every {{ hud.decideEvery }} frames.
        </p>

        <template v-if="mode === 'ai' && primary">
          <div class="qtitle">{{ primary.instructions }}</div>
          <ul class="bars">
            <li v-for="o in hud.probs" :key="o.key" :class="{ chosen: o.key === hud.action }">
              <span class="bk">{{ o.key }}</span>
              <span class="bt"><span class="bf" :style="{ width: `${Math.max(1, o.p * 100)}%` }" /></span>
              <span class="bp">{{ pct(o.p) }}</span>
            </li>
          </ul>
          <div v-if="hud.executed && hud.executed !== hud.action" class="exec">
            executed: <strong>{{ hud.executed }}</strong> (reflex override)
          </div>

          <div v-if="hud.secondary" class="secondary">
            <div class="qtitle">{{ SECONDARY_LABEL[kind] }}</div>
            <div class="gauge" :class="{ warn: secondaryPct > 50 }">
              <span class="gf" :style="{ width: `${secondaryPct}%` }" />
            </div>
            <div class="gv">
              <template v-if="hud.secondary.type === 'noul'">p(true) = {{ pct(hud.secondary.value) }}</template>
              <template v-else>expected level {{ hud.secondary.value.toFixed(2) }} / {{ hud.secondary.max }}</template>
            </div>
          </div>
        </template>

        <div v-if="mode === 'human'" class="controls">
          <p class="help">{{ HELP[kind] }}</p>
          <div class="buttons" :class="kind">
            <button
              v-for="b in HUMAN_BUTTONS[kind]"
              :key="b.key"
              type="button"
              :class="['ctl', b.key]"
              @pointerdown.prevent="press(b.key)"
            >
              {{ b.label }}
            </button>
          </div>
        </div>

        <div class="foot">
          <span>vetoes: {{ hud.vetoCount }}</span>
          <span>episodes: {{ hud.episodes }}</span>
        </div>
        <code v-if="mode === 'ai' && hud.stateText" class="state" title="State text sent to the model">{{ hud.stateText }}</code>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.play {
  margin-top: 12px;
}
.tabs {
  display: flex;
  gap: 6px;
  padding: 4px;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  width: fit-content;
  max-width: 100%;
  overflow-x: auto;
}
.tab {
  padding: 7px 14px;
  border-radius: 9px;
  font-weight: 600;
  font-size: 14px;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}
.tab.active {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
}
.blurb {
  margin: 12px 0 !important;
  color: var(--vp-c-text-2);
  font-size: 14px;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  align-items: center;
  margin-bottom: 12px;
}
.seg {
  display: inline-flex;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  overflow: hidden;
}
.seg button {
  padding: 6px 14px;
  font-weight: 600;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.seg button.on {
  color: #fff;
  background: var(--vp-c-brand-1);
}
.field {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.field select {
  padding: 5px 8px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 13px;
  max-width: 280px;
}
.check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.ghost {
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  font-size: 13px;
  font-weight: 600;
}
.ghost:hover {
  border-color: var(--vp-c-brand-1);
}
.status,
.error {
  margin: 0 0 10px !important;
  font-size: 13px;
}
.error {
  color: var(--vp-c-danger-1);
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 16px;
  align-items: start;
}
@media (max-width: 960px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
.stage {
  position: relative;
  width: 100%;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  touch-action: none;
  user-select: none;
  outline: none;
}
.stage:focus-visible {
  box-shadow: 0 0 0 2px var(--vp-c-brand-1);
}
/* Width is capped by the viewport height so the whole canvas stays visible. */
.stage.flappy {
  aspect-ratio: 15 / 13;
  width: min(100%, calc(72vh * 15 / 13));
}
.stage.rescue {
  aspect-ratio: 1 / 1.06;
  width: min(100%, calc(78vh / 1.06));
}
.stage.dodge {
  aspect-ratio: 16 / 10;
  width: min(100%, calc(72vh * 1.6));
}
.canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.veto {
  position: absolute;
  top: 14px;
  left: 50%;
  transform: translateX(-50%);
  padding: 6px 16px;
  border-radius: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: #fff;
  background: #dc2626;
  box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.3);
  animation: flash 0.7s ease-out forwards;
  pointer-events: none;
}
@keyframes flash {
  0% {
    opacity: 1;
    transform: translateX(-50%) scale(1.15);
  }
  70% {
    opacity: 1;
    transform: translateX(-50%) scale(1);
  }
  100% {
    opacity: 0;
  }
}
@media (prefers-reduced-motion: reduce) {
  .veto {
    animation: none;
  }
}
.overlay {
  position: absolute;
  inset: auto 0 18px 0;
  margin: auto;
  width: fit-content;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  color: #fff;
  background: rgba(15, 23, 42, 0.75);
  pointer-events: none;
}
.hud {
  padding: 16px;
  border-radius: 16px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  font-size: 13px;
}
.policy {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 14px;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f59e0b;
}
.dot.model {
  background: #22c55e;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 12px 0;
}
.mv {
  font-size: 20px;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
.ml {
  font-size: 11px;
  color: var(--vp-c-text-3);
}
.depth {
  margin: 0 0 10px;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  font-size: 12px;
}
.drow {
  display: grid;
  grid-template-columns: 96px auto 1fr;
  align-items: center;
  gap: 8px;
  margin: 2px 0;
}
.dl {
  color: var(--vp-c-text-3);
}
.dv {
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  text-align: right;
  white-space: nowrap;
}
.cells {
  display: inline-flex;
  gap: 3px;
}
.cell {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  background: var(--vp-c-default-soft);
}
.cell.on {
  background: var(--vp-c-brand-1);
}
.tag {
  padding: 0 6px;
  border-radius: 6px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  background: var(--vp-c-default-soft);
}
.tag.ok {
  color: #fff;
  background: #16a34a;
}
.throttle {
  margin: 0 0 10px !important;
  font-size: 12px;
  color: var(--vp-c-warning-1);
}
.qtitle {
  margin: 10px 0 6px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  font-size: 12px;
}
.bars {
  list-style: none;
  margin: 0 !important;
  padding: 0 !important;
}
.bars li {
  display: grid;
  grid-template-columns: 86px 1fr 48px;
  align-items: center;
  gap: 8px;
  margin: 0 0 5px !important;
}
.bk {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bt {
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.bf {
  display: block;
  height: 100%;
  border-radius: 5px;
  background: var(--vp-c-text-3);
  transition: width 0.08s linear;
}
.chosen .bf {
  background: var(--vp-c-brand-1);
}
.chosen .bk {
  font-weight: 700;
  color: var(--vp-c-brand-1);
}
.bp {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}
.exec {
  margin-top: 4px;
  font-size: 12px;
  color: var(--vp-c-danger-1);
}
.gauge {
  height: 12px;
  border-radius: 6px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.gf {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #f59e0b 60%, #ef4444);
  transition: width 0.08s linear;
}
.gv {
  margin-top: 4px;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.controls {
  margin-top: 8px;
}
.help {
  margin: 0 0 8px !important;
  font-size: 12px;
  color: var(--vp-c-text-3);
}
.buttons {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, 1fr);
}
.buttons.flappy {
  grid-template-columns: 1fr;
}
.buttons.rescue {
  grid-template-columns: repeat(3, 1fr);
  grid-template-areas:
    '. n .'
    'w hv e'
    '. s home';
}
.buttons.rescue .north {
  grid-area: n;
}
.buttons.rescue .west {
  grid-area: w;
}
.buttons.rescue .east {
  grid-area: e;
}
.buttons.rescue .south {
  grid-area: s;
}
.buttons.rescue .hover {
  grid-area: hv;
}
.buttons.rescue .return_home {
  grid-area: home;
}
.ctl {
  padding: 12px 8px;
  border-radius: 10px;
  font-weight: 700;
  font-size: 14px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  touch-action: manipulation;
}
.ctl:active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
}
.ctl.thrust {
  padding: 18px;
  font-size: 16px;
}
.ctl.abort {
  color: var(--vp-c-danger-1);
}
.foot {
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  font-size: 12px;
  color: var(--vp-c-text-3);
}
.state {
  display: block;
  margin-top: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 11px;
  line-height: 1.5;
  word-break: break-word;
  white-space: normal;
}
</style>
