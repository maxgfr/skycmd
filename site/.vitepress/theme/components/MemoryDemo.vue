<script setup lang="ts">
// The memory demo (s6-sleep3-nano, the Memory page and its Space): `recall` episodes played in the
// page by the wasm `RecallProbe` (crates/sky-web/src/recall.rs), which runs the evaluation's own
// episode generator and scoring with the published file at its last exit. One episode is shown
// frame by frame (what appears, the model's probabilities, right or wrong); a batch button plays
// 200 episodes with the memory on, with it zeroed and with the control file, and puts the
// accuracy by delay bucket next to the test reading. The model failed its gate: the page says so
// before it shows anything. Pure logic in lib/memory.ts; everything runs in the page.
import { computed, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import {
  clockStepMicros,
  engineLabel,
  formatInt,
  formatMicros,
  loadMemoryEntry,
  loadRecallProbe,
  type MemoryEntry,
  type WasmRecallProbe,
} from '../lib/wasm'
import { formatSize } from '../lib/size'
import {
  ARM_LABEL,
  BATCH_EPISODES,
  BROWSER_FROM,
  BUCKET_LABELS,
  KEYS,
  PASSES,
  TEST_POOLED,
  TEST_SEED0,
  Tally,
  askLine,
  episodeLine,
  formatInterval,
  formatShare,
  frameLine,
  validationSeeds,
  type Arm,
  type Ask,
  type Frame,
  type Passes,
} from '../lib/memory'

/** One frame played by the model (`RecallProbe.step`). */
interface Played extends Frame {
  probs: number[]
  choice: string
  correct?: boolean
  micros: number
  done: boolean
}

const entry = shallowRef<MemoryEntry | null>(null)
const model = shallowRef<WasmRecallProbe | null>(null)
const control = shallowRef<WasmRecallProbe | null>(null)
const status = ref('Loading…')
const missing = ref(false)
const error = ref('')
const clockStep = ref(0)

// The episode viewer.
const arm = ref<Arm>('on')
const passes = ref<Passes>(1)
const episodeIdx = ref(0)
const frames = ref<Frame[]>([])
const played = ref<Played[]>([])
const auto = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

// The batch.
const running = ref(false)
const progress = ref('')
const results = ref<Partial<Record<Arm, Tally>>>({})
const batchPasses = ref<Passes>(1)
let cancel = false

const seed = computed(() => validationSeeds(episodeIdx.value + 1)[episodeIdx.value])
const current = computed<Played | null>(() => played.value[played.value.length - 1] ?? null)
const finished = computed(() => played.value.length > 0 && played.value[played.value.length - 1].done)
const episodeAsks = computed(() => played.value.filter((p) => p.kind === 'ask'))
const episodeRight = computed(() => episodeAsks.value.filter((p) => p.correct).length)
const plannedAsks = computed(() => frames.value.filter((f) => f.kind === 'ask').length)
const probeOf = (a: Arm) => (a === 'control' ? control.value : model.value)

onMounted(async () => {
  clockStep.value = clockStepMicros()
  try {
    const e = await loadMemoryEntry()
    if (!e) {
      missing.value = true
      status.value = ''
      return
    }
    status.value = `Loading ${e.name} (${formatSize(e.bytes)}) and its control…`
    const [m, c] = await Promise.all([loadRecallProbe(e, 'model'), loadRecallProbe(e, 'control')])
    if (!m.hasMemory()) throw new Error(`${e.name}: the model file has no memory`)
    if (c.hasMemory()) throw new Error(`${e.name}: the control file has a memory`)
    const inf = JSON.parse(m.info()) as { n_params: number; file_bytes: number; sleep_every: number }
    const cinf = JSON.parse(c.info()) as { n_params: number; file_bytes: number }
    const line = [
      e.name,
      `${formatInt(inf.n_params)} parameters`,
      formatSize(inf.file_bytes),
      `control ${formatInt(cinf.n_params)} parameters, ${formatSize(cinf.file_bytes)}`,
      'last exit',
    ]
    const eng = engineLabel()
    if (eng) line.push(eng)
    status.value = line.join(' · ')
    entry.value = e
    model.value = m
    control.value = c
    startEpisode()
  } catch (err) {
    error.value = `Could not load the model: ${(err as Error).message ?? err}`
    status.value = ''
  }
})

onBeforeUnmount(() => {
  stopAuto()
  cancel = true
})

/** Applies the viewer's settings to its probe and starts the episode of `seed` again. */
function startEpisode() {
  stopAuto()
  const p = probeOf(arm.value)
  if (!p) return
  error.value = ''
  try {
    if (arm.value !== 'control') {
      p.setMemory(arm.value)
      p.setSleepPasses(passes.value)
    }
    const d = JSON.parse(p.start(seed.value)) as { frames: Frame[] }
    frames.value = d.frames
    played.value = []
  } catch (err) {
    error.value = (err as Error).message ?? String(err)
  }
}

function stepOnce(): boolean {
  const p = probeOf(arm.value)
  if (!p || finished.value) return false
  try {
    played.value = [...played.value, JSON.parse(p.step()) as Played]
    return !finished.value
  } catch (err) {
    error.value = (err as Error).message ?? String(err)
    return false
  }
}

function toggleAuto() {
  if (auto.value) return stopAuto()
  if (finished.value) startEpisode()
  auto.value = true
  timer = setInterval(() => {
    if (!stepOnce()) stopAuto()
  }, 180)
}

function stopAuto() {
  auto.value = false
  if (timer) clearInterval(timer)
  timer = null
}

function toEnd() {
  stopAuto()
  while (stepOnce()) {
    /* play the remaining frames */
  }
}

function nextEpisode() {
  episodeIdx.value += 1
  startEpisode()
}

function pick(a: Arm) {
  arm.value = a
  startEpisode()
}

function pickPasses(n: Passes) {
  passes.value = n
  startEpisode()
}

const yieldToPage = () => new Promise((r) => setTimeout(r, 0))

/** Plays `BATCH_EPISODES` episodes per arm: the memory on at the chosen N, the memory zeroed, the
 * control file. The viewer's episode is started again afterwards. */
async function runBatch() {
  if (running.value || !model.value || !control.value) return
  stopAuto()
  running.value = true
  cancel = false
  error.value = ''
  const n = passes.value
  batchPasses.value = n
  results.value = {}
  const seeds = validationSeeds(BATCH_EPISODES)
  const arms: Arm[] = ['on', 'zeroed', 'control']
  try {
    for (const [i, a] of arms.entries()) {
      const p = probeOf(a)!
      if (a !== 'control') {
        p.setMemory(a)
        p.setSleepPasses(n)
      }
      const t = new Tally()
      results.value = { ...results.value, [a]: t }
      for (const [k, s] of seeds.entries()) {
        if (cancel) throw new Error('stopped')
        const r = JSON.parse(p.runEpisode(s)) as { asks: Ask[]; decisions: number; micros: number }
        t.addEpisode(r.asks, r.decisions, r.micros)
        if (k % 4 === 3 || k === seeds.length - 1) {
          progress.value = `${ARM_LABEL[a]} (${i + 1} of ${arms.length}): episode ${k + 1} of ${seeds.length}`
          results.value = { ...results.value }
          await yieldToPage()
        }
      }
    }
    progress.value = `Done: ${formatInt(3 * seeds.length)} episodes, ${formatInt(
      arms.reduce((x, a) => x + (results.value[a]?.decisions ?? 0), 0),
    )} decisions.`
  } catch (err) {
    const msg = (err as Error).message ?? String(err)
    progress.value = msg === 'stopped' ? 'Stopped: the table holds the episodes played so far.' : ''
    if (msg !== 'stopped') error.value = msg
  } finally {
    running.value = false
    startEpisode()
  }
}

function stopBatch() {
  cancel = true
}

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`
}

function cellClass(f: Frame, i: number) {
  const p = played.value[i]
  return {
    show: f.kind === 'show',
    ask: f.kind === 'ask',
    idle: f.kind === 'idle',
    now: current.value?.frame === f.frame,
    right: p?.kind === 'ask' && p.correct === true,
    wrong: p?.kind === 'ask' && p.correct === false,
    future: i >= played.value.length,
  }
}

function cellLabel(f: Frame, i: number) {
  const p = played.value[i]
  const base = `frame ${f.frame + 1}: ${frameLine(f)}`
  if (p?.kind === 'ask') return `${base}; the model answered ${p.choice}, ${p.correct ? 'right' : 'wrong'}`
  return base
}

const testNote = computed(() => TEST_SEED0.sleep[batchPasses.value])
</script>

<template>
  <div class="mem">
    <p class="mem-failed">
      <strong>s6-sleep3-nano failed its pre-registered gate: a research result, not a part to use.</strong> Its
      memory is real but fades with the delay; sleep passes did not help; and it plays two of the four games much
      worse than the same training without a memory.
    </p>
    <p class="mem-privacy">Everything runs in your browser: the model, its memory and the game.</p>
    <p v-if="status" class="mem-status">{{ status }}</p>
    <p v-if="missing" class="mem-note">s6-sleep3-nano is not published on this site.</p>
    <p v-if="error" class="mem-error" role="alert">{{ error }}</p>

    <template v-if="model && control">
      <section class="mem-episode" aria-labelledby="mem-episode-title">
        <h3 id="mem-episode-title">One episode, frame by frame</h3>
        <div class="mem-row">
          <span class="mem-label" id="mem-arm">Who answers:</span>
          <div class="mem-chips" role="group" aria-labelledby="mem-arm">
            <button
              v-for="a in (['on', 'zeroed', 'control'] as Arm[])"
              :key="a"
              type="button"
              :class="{ on: arm === a }"
              :aria-pressed="arm === a"
              :disabled="running"
              @click="pick(a)"
            >{{ ARM_LABEL[a] }}</button>
          </div>
        </div>
        <div class="mem-row">
          <span class="mem-label" id="mem-passes">Sleep passes N:</span>
          <div class="mem-chips" role="group" aria-labelledby="mem-passes">
            <button
              v-for="n in PASSES"
              :key="n"
              type="button"
              :class="{ on: passes === n }"
              :aria-pressed="passes === n"
              :disabled="running || arm !== 'on'"
              @click="pickPasses(n)"
            >{{ n }}</button>
          </div>
          <span class="mem-hint">
            <template v-if="arm === 'on'">every 8 frames the model replays them N − 1 more times; N = 1 is no sleep</template>
            <template v-else-if="arm === 'zeroed'">the memory is reset before every frame, so there is nothing to replay</template>
            <template v-else>the control has no memory</template>
          </span>
        </div>

        <ol class="mem-strip" aria-label="The 64 frames of the episode">
          <li
            v-for="(f, i) in frames"
            :key="f.frame"
            :class="cellClass(f, i)"
            :aria-label="cellLabel(f, i)"
            :title="cellLabel(f, i)"
          >
            <span v-if="f.kind === 'show'" class="mem-swatch" :data-colour="f.colour">{{ f.door }}</span>
            <span v-else-if="f.kind === 'ask'" class="mem-q">{{ f.door }}?</span>
          </li>
        </ol>
        <p class="mem-legend">
          Episode seed {{ formatInt(seed) }} · a coloured square: a door is shown its colour · a door with “?”: the
          question · grey: a frame with only a distractor number · an asked frame turns green when the model is right
          and red when it is wrong.
        </p>

        <div class="mem-controls">
          <button type="button" :disabled="running || finished" @click="stepOnce">Next frame</button>
          <button type="button" :disabled="running" @click="toggleAuto">{{ auto ? 'Pause' : finished ? 'Replay' : 'Play' }}</button>
          <button type="button" :disabled="running || finished" @click="toEnd">To the end</button>
          <button type="button" :disabled="running" @click="startEpisode">Restart</button>
          <button type="button" :disabled="running" @click="nextEpisode">Next episode</button>
        </div>

        <div class="mem-frame" aria-live="polite">
          <template v-if="current">
            <p class="mem-what">
              <strong>Frame {{ current.frame + 1 }} of {{ frames.length }}:</strong> {{ frameLine(current) }}.
              <template v-if="current.kind === 'ask'"> {{ askLine(current) }}; the right answer is <strong>{{ current.answer }}</strong>.</template>
            </p>
            <p class="mem-sees">The model reads: <code>{{ current.state }}</code></p>
            <ul class="mem-probs" aria-label="The model's probabilities">
              <li v-for="(k, j) in KEYS" :key="k" :class="{ chosen: current.choice === k, answer: current.answer === k }">
                <span class="mem-key">{{ k }}</span>
                <span class="mem-bar"><span class="mem-fill" :data-colour="k" :style="{ width: pct(current.probs[j]) }" /></span>
                <span class="mem-p">{{ current.probs[j].toFixed(3) }}</span>
              </li>
            </ul>
            <p class="mem-verdict">
              It answers <strong>{{ current.choice }}</strong>
              <template v-if="current.kind === 'ask'">
                ·
                <strong :class="current.correct ? 'ok' : 'ko'">{{ current.correct ? 'right' : 'wrong' }}</strong>
              </template>
              <template v-else-if="current.choice !== 'wait'"> · nothing was asked, so this answer scores nothing</template>
              · {{ formatMicros(current.micros) }} for this decision
            </p>
          </template>
          <p v-else class="mem-what">
            The episode has {{ frames.length }} frames and {{ plannedAsks }} questions. Press “Next frame” or “Play”.
          </p>
          <p class="mem-meta">
            {{ episodeLine(episodeRight, episodeAsks.length, plannedAsks, finished) }}; a model without memory is right
            about one time in four.
          </p>
        </div>
      </section>

      <section class="mem-batch" aria-labelledby="mem-batch-title">
        <h3 id="mem-batch-title">{{ BATCH_EPISODES }} episodes: how the memory fades with the delay</h3>
        <p class="mem-meta">
          Plays {{ BATCH_EPISODES }} episodes three times: s6-sleep3-nano with its memory at N = {{ passes }}, the same
          file with its memory zeroed, and the control file. About {{ formatInt(BATCH_EPISODES * 64 * 3) }} decisions;
          it takes some seconds to a minute.
        </p>
        <div class="mem-controls">
          <button type="button" class="primary" :disabled="running" @click="runBatch">Run {{ BATCH_EPISODES }} episodes</button>
          <button v-if="running" type="button" @click="stopBatch">Stop</button>
          <span class="mem-progress" aria-live="polite">{{ progress }}</span>
        </div>

        <div v-if="results.on" class="mem-table-wrap">
          <table class="mem-table">
            <caption>
              Accuracy on the questions, by the delay since the door was shown (frames). Browser columns: the share
              right and its 95% interval. Test columns: the files' test reading, as written in the result files.
            </caption>
            <thead>
              <tr>
                <th scope="col">delay</th>
                <th scope="col">browser: memory on, N = {{ batchPasses }}</th>
                <th scope="col">browser: memory zeroed</th>
                <th scope="col">browser: control file</th>
                <th scope="col">test: this file, N = {{ batchPasses }}</th>
                <th scope="col">test: the control file</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(label, b) in BUCKET_LABELS" :key="label">
                <th scope="row">{{ label }}</th>
                <td v-for="a in (['on', 'zeroed', 'control'] as Arm[])" :key="a">
                  <template v-if="results[a]">
                    {{ formatShare(results[a]!.accuracy(b)) }}
                    <span class="mem-iv">{{ formatInterval(results[a]!.interval(b)) }}</span>
                  </template>
                </td>
                <td>{{ testNote.buckets[b] }}</td>
                <td>{{ TEST_SEED0.control.buckets[b] }}</td>
              </tr>
              <tr class="all">
                <th scope="row">all</th>
                <td v-for="a in (['on', 'zeroed', 'control'] as Arm[])" :key="a">
                  <template v-if="results[a]">
                    {{ formatShare(results[a]!.accuracy()) }}
                    <span class="mem-iv">{{ formatInterval(results[a]!.interval()) }}</span>
                  </template>
                </td>
                <td>{{ testNote.overall }}</td>
                <td>{{ TEST_SEED0.control.overall }}</td>
              </tr>
              <tr class="meta">
                <th scope="row">questions</th>
                <td v-for="a in (['on', 'zeroed', 'control'] as Arm[])" :key="a">
                  <template v-if="results[a]">{{ formatInt(results[a]!.totalAsked) }} in {{ formatInt(results[a]!.episodes) }} episodes</template>
                </td>
                <td colspan="2">1,000 episodes per file</td>
              </tr>
              <tr class="meta">
                <th scope="row">time per decision</th>
                <td v-for="a in (['on', 'zeroed', 'control'] as Arm[])" :key="a">
                  <template v-if="results[a]?.medianMicros != null">{{ formatMicros(results[a]!.medianMicros!) }}</template>
                </td>
                <td colspan="2"></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="mem-meta">
          The browser plays other episodes than the test: seeds of the validation split, from
          {{ formatInt(BROWSER_FROM) }} upward, which no training used, where the test read 1,000 episodes from seed
          5,000,000,000 with the training code. With about 400 questions per delay bucket, the browser's shares differ from
          the test's by sampling noise, about ±0.05 (the intervals). Pooled over the three training seeds, the test reads
          {{ TEST_POOLED.sleep[1].overall }} at N = 1 and {{ TEST_POOLED.sleep[4].overall }} at N = 4, against
          {{ TEST_POOLED.control.overall }} for the controls; the memory zeroed was not read on the test.
          <template v-if="clockStep >= 20"> This browser's clock counts in steps of {{ formatInt(clockStep) }} µs.</template>
        </p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.mem {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 16px 0 24px;
  background: var(--vp-c-bg-soft);
}
.mem h3 {
  margin: 18px 0 8px;
  font-size: 16px;
}
.mem-failed {
  margin: 0 0 10px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  background: var(--vp-c-danger-soft);
}
.mem-privacy {
  margin: 0 0 10px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.mem-status,
.mem-note {
  margin: 0 0 10px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.mem-error {
  margin: 10px 0;
  font-size: 14px;
  color: var(--vp-c-danger-1);
}
.mem-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  align-items: center;
  margin-bottom: 8px;
}
.mem-label,
.mem-hint {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.mem-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.mem button {
  font: inherit;
  font-size: 13px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 3px 12px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
}
.mem button.on,
.mem button.primary {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.mem button:disabled {
  opacity: 0.5;
  cursor: default;
}
.mem button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}
.mem-strip {
  list-style: none;
  padding: 0;
  margin: 10px 0 4px;
  display: grid;
  grid-template-columns: repeat(16, minmax(0, 1fr));
  gap: 3px;
}
.mem-strip li {
  height: 26px;
  border-radius: 4px;
  border: 2px solid transparent;
  background: var(--vp-c-default-soft);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  min-width: 0;
}
.mem-strip li.future {
  opacity: 0.55;
}
.mem-strip li.ask {
  background: var(--vp-c-bg);
  border-color: var(--vp-c-divider);
}
.mem-strip li.now {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}
.mem-strip li.right {
  border-color: var(--vp-c-success-1, #2e7d32);
  background: var(--vp-c-success-soft, rgba(46, 125, 50, 0.16));
}
.mem-strip li.wrong {
  border-color: var(--vp-c-danger-1);
  background: var(--vp-c-danger-soft);
}
.mem-swatch {
  width: 100%;
  height: 100%;
  border-radius: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  text-shadow: 0 0 2px rgba(0, 0, 0, 0.7);
}
[data-colour='red'] {
  background: #d32f2f;
}
[data-colour='green'] {
  background: #2e7d32;
}
[data-colour='blue'] {
  background: #1565c0;
}
[data-colour='yellow'] {
  background: #f9a825;
}
[data-colour='wait'] {
  background: var(--vp-c-text-3);
}
.mem-q {
  color: var(--vp-c-text-1);
}
.mem-legend {
  margin: 4px 0 10px;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.mem-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin: 8px 0;
}
.mem-progress {
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.mem-frame {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--vp-c-bg);
}
.mem-what,
.mem-sees,
.mem-verdict {
  margin: 0 0 6px;
  font-size: 14px;
}
.mem-sees code {
  font-size: 13px;
  word-break: break-word;
}
.mem-probs {
  list-style: none;
  padding: 0;
  margin: 6px 0;
  display: grid;
  gap: 4px;
}
.mem-probs li {
  display: grid;
  grid-template-columns: 4.5em 1fr 3.5em;
  gap: 8px;
  align-items: center;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.mem-probs li.chosen .mem-key {
  font-weight: 700;
}
.mem-probs li.answer .mem-key::after {
  content: ' (right)';
  font-weight: 400;
  color: var(--vp-c-text-2);
}
.mem-bar {
  position: relative;
  height: 10px;
  border-radius: 5px;
  background: var(--vp-c-default-soft);
  overflow: hidden;
}
.mem-fill {
  position: absolute;
  inset: 0 auto 0 0;
  transition: width 0.12s ease-out;
}
.mem-p {
  text-align: right;
}
.ok {
  color: var(--vp-c-success-1, #2e7d32);
}
.ko {
  color: var(--vp-c-danger-1);
}
.mem-meta {
  margin: 8px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
}
.mem-table-wrap {
  overflow-x: auto;
}
.mem-table {
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  margin: 8px 0 0;
}
.mem-table caption {
  caption-side: top;
  text-align: left;
  font-size: 12px;
  color: var(--vp-c-text-2);
  padding-bottom: 4px;
}
.mem-table td,
.mem-table th {
  padding: 4px 8px;
}
.mem-table tr.all {
  font-weight: 700;
}
.mem-table tr.meta {
  color: var(--vp-c-text-2);
}
.mem-iv {
  display: block;
  font-size: 11px;
  font-weight: 400;
  color: var(--vp-c-text-2);
}
</style>
