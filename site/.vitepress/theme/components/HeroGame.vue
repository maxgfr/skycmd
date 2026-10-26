<script setup lang="ts">
// Landing hero: Flappy Drone played live by the smallest model (or the oracle as a fallback).
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { GameRunner } from '../lib/runner'
import { gameModels, loadEngine, loadManifest, loadWasm } from '../lib/wasm'
import { live } from '../lib/store'

const canvas = ref<HTMLCanvasElement | null>(null)
const failed = ref('')
const paused = ref(false)
let runner: GameRunner | null = null
let observer: IntersectionObserver | null = null

function toggle() {
  if (!runner) return
  if (runner.isRunning) {
    runner.stop()
    paused.value = true
  } else {
    runner.start()
    paused.value = false
  }
}

onMounted(async () => {
  try {
    const mod = await loadWasm()
    if (!canvas.value) return
    runner = new GameRunner(mod, canvas.value, {
      kind: 'flappy',
      seed: 7,
      autoResetMs: 700,
      onHud: (h) => {
        live.medianUs = h.medianUs
        live.decisionsPerS = h.decisionsPerS
        live.best = h.best
      },
    })
    const models = gameModels(await loadManifest())
    if (models.length) {
      try {
        const entry = models[0]
        const engine = await loadEngine(entry)
        runner.setEngine(engine, entry.name)
        live.model = entry.name
        live.params = engine.n_params()
        live.bytes = engine.file_bytes()
        live.status = 'model'
      } catch {
        live.status = 'oracle'
      }
    } else {
      live.status = 'oracle'
    }
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    runner.render()
    if (reduced) {
      paused.value = true
    } else {
      runner.start()
    }
    // Pause while the hero is off screen.
    observer = new IntersectionObserver(([e]) => {
      if (!runner || paused.value) return
      if (e.isIntersecting) runner.start()
      else runner.stop()
    })
    observer.observe(canvas.value)
  } catch (e) {
    live.status = 'error'
    failed.value = `The wasm module could not be loaded (${(e as Error).message}). Build it with scripts/build-wasm.sh.`
  }
})

onBeforeUnmount(() => {
  observer?.disconnect()
  runner?.destroy()
  runner = null
})
</script>

<template>
  <div class="hero-game">
    <canvas ref="canvas" class="hero-canvas" aria-label="Flappy Drone played live by the model" role="img" />
    <div class="hero-badge">
      <span class="dot" :class="live.status" />
      <span v-if="live.status === 'model'">{{ live.model }} playing live</span>
      <span v-else-if="live.status === 'oracle'">oracle · coded rules, no model</span>
      <span v-else-if="live.status === 'error'">offline</span>
      <span v-else>loading…</span>
      <button class="hero-pause" type="button" :aria-label="paused ? 'Play' : 'Pause'" @click="toggle">
        {{ paused ? '▶' : '❚❚' }}
      </button>
    </div>
    <p v-if="failed" class="hero-error">{{ failed }}</p>
    <a class="hero-play" :href="withBase('/play/')">Take the controls →</a>
  </div>
</template>

<style scoped>
.hero-game {
  position: relative;
  width: min(100%, 440px);
  aspect-ratio: 15 / 16;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 24px 60px -24px rgba(14, 165, 233, 0.45), 0 0 0 1px var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}
.hero-canvas {
  width: 100%;
  height: 100%;
  display: block;
}
.hero-badge {
  position: absolute;
  left: 12px;
  top: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: #f8fafc;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(6px);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #94a3b8;
}
.dot.model {
  background: #22c55e;
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.25);
}
.dot.oracle {
  background: #f59e0b;
}
.dot.error {
  background: #ef4444;
}
.hero-pause {
  border: 0;
  border-radius: 999px;
  width: 24px;
  height: 22px;
  font-size: 10px;
  color: inherit;
  background: rgba(255, 255, 255, 0.12);
  cursor: pointer;
}
.hero-play {
  position: absolute;
  right: 12px;
  bottom: 12px;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  color: #fff;
  background: rgba(14, 165, 233, 0.9);
  text-decoration: none;
}
.hero-play:hover {
  background: #0284c7;
}
.hero-error {
  position: absolute;
  inset: auto 12px 52px 12px;
  margin: 0;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
  color: #fff;
  background: rgba(185, 28, 28, 0.85);
}
</style>
