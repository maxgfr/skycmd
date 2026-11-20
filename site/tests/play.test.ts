// Tests of the Play page's drawing and counters against the game cores.
//
//   cd site && npm test          (node --test, Node >= 22.18 for TypeScript type stripping)
//
// The pure helpers of the renderer are tested directly. The session tests drive the browser
// module (public/pkg, built by scripts/build-wasm.sh) with the calls the page makes
// (`GameSession`, `step_oracle`, `step_human`, `step_ai`, `snapshot`) and compare, frame by
// frame, what the canvas would draw with the core's state; they are skipped without the module.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  DODGE_BEHIND_M,
  DODGE_MIN_VIEW_M,
  DODGE_TRAIL_FRAMES,
  DRONE_SIDE_EXTENT,
  WIND_DIRS,
  dodgeAboveBody,
  dodgeClearances,
  dodgeClearsAtAltitude,
  dodgeDeckThickness,
  dodgeInGapLane,
  dodgeLayout,
  dodgeManeuverLabel,
  dodgeMeters,
  dodgeNextObstacle,
  dodgeOutcome,
  dodgeOverlapsX,
  dodgeSupportWidth,
  dodgeTilt,
  dodgeZoomTarget,
  droneSizeForSpan,
  nextDodgeTrail,
  rescueHeadwind,
  type DodgeLayout,
  type DodgeTrailPoint,
  type DodgeView,
} from '../.vitepress/theme/lib/render.ts'
import { endedInCrash } from '../.vitepress/theme/lib/events.ts'
import { FirstPress, actionForKey, actionForSwipe } from '../.vitepress/theme/lib/controls.ts'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../public/pkg')
const models = join(here, '../public/models')

type Box = { x0: number; x1: number; y0: number; y1: number }

/** A seeded generator in [0, 1) (mulberry32): scripted pilots press reproducible keys. */
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const overlap = (a: Box, b: Box, tol = 0) =>
  Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) > tol && Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > tol

/** Canvas box of the side-view drone sprite: rotor discs, body and skids, rotated by `tilt`. */
function spriteBox(cx: number, cy: number, size: number, tilt: number): Box {
  const pts: [number, number][] = [
    [-0.42, -0.24],
    [0.42, -0.24],
    [-0.42, 0.12],
    [0.42, 0.12],
    [-0.36, 0.24],
    [0.36, 0.24],
  ]
  for (const sx of [-0.62, 0.62]) {
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2
      pts.push([sx + 0.3 * Math.cos(a), -0.3 + 0.06 * Math.sin(a)])
    }
  }
  const [c, s] = [Math.cos(tilt), Math.sin(tilt)]
  const xs = pts.map(([x, y]) => cx + (x * c - y * s) * size)
  const ys = pts.map(([x, y]) => cy + (x * s + y * c) * size)
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
}

/** What the side view draws as solid: obstacle bodies (not the ghosted ones), decks and their
 * supports. */
function solids(v: DodgeView, lay: DodgeLayout): Box[] {
  const { X, Z, k } = lay
  const out: Box[] = []
  for (const o of v.obstacles) {
    if (!dodgeInGapLane(o, v.drone)) out.push({ x0: X(o.x0), x1: X(o.x1), y0: Z(o.top), y1: Z(0) })
    if (o.deck) {
      const [d0, d1, ceil] = o.deck
      const th = dodgeDeckThickness(k)
      out.push({ x0: X(d0), x1: X(d1), y0: Z(ceil) - th, y1: Z(ceil) })
      // The front and back supports, from the deck up past the top of the view.
      const sw = dodgeSupportWidth(k)
      out.push({ x0: X(d0), x1: X(d0) + sw, y0: -1e6, y1: Z(ceil) - th })
      out.push({ x0: X(d1) - sw, x1: X(d1), y0: -1e6, y1: Z(ceil) - th })
    }
  }
  return out
}

function hitbox(v: DodgeView, lay: DodgeLayout): Box {
  const d = v.drone
  return { x0: lay.X(d.x - d.hl), x1: lay.X(d.x + d.hl), y0: lay.Z(d.z + d.hh), y1: lay.Z(d.z - d.hh) }
}

// ------------------------------------------------------------------------------------------
// Pure helpers.

test('the side-view drone spans its hitbox length, rotor tip to rotor tip', () => {
  for (const k of [8, 14.3, 18, 40]) {
    const size = droneSizeForSpan(2 * 0.5 * k)
    const b = spriteBox(0, 0, size, 0)
    assert.ok(Math.abs(b.x1 - b.x0 - k) < 1e-6, `k ${k}: span ${b.x1 - b.x0}`)
    assert.ok(Math.abs(DRONE_SIDE_EXTENT.right * size - b.x1) < 1e-6)
    // Up and down it stays inside the 0.6 m tall hitbox.
    assert.ok(-b.y0 <= 0.3 * k && b.y1 <= 0.3 * k)
  }
})

test('a drone is in the gap lane only when its whole width is inside the gap', () => {
  const o = { x0: 0, x1: 4, top: 12, deck: null, gap: [1.8, 5] as [number, number], impassable: false, passed: false }
  const d = { x: 0, y: 3.4, z: 2, v: 11, hl: 0.5, hw: 0.5, hh: 0.3 }
  assert.equal(dodgeInGapLane(o, d), true)
  assert.equal(dodgeInGapLane(o, { ...d, y: 2.2 }), false)
  assert.equal(dodgeInGapLane({ ...o, gap: null }, d), false)
})

test('the Dodge banner names every way an episode ends, timeout included', () => {
  const base = {
    drone: { x: 0, y: 0, z: 2, v: 0, hl: 0.5, hw: 0.5, hh: 0.3 },
    obstacles: [{ x0: 30, x1: 34, top: 14, deck: null, gap: null, impassable: true, passed: false }],
    finish_x: 400,
    y_max: 5,
    z_max: 18,
    frame: 7200,
    passed: 0,
    action: 'brake',
    vetoed: true,
    vetoed_request: 'continue',
    crashed: false,
    finished: false,
    aborted: false,
  } satisfies DodgeView
  assert.equal(dodgeOutcome(base, false), null)
  assert.match(dodgeOutcome(base, true)!.text, /Timed out/)
  assert.equal(dodgeOutcome({ ...base, crashed: true }, true)!.text, 'Crashed')
  assert.equal(dodgeOutcome({ ...base, aborted: true }, true)!.ok, true)
  assert.equal(dodgeOutcome({ ...base, finished: true }, true)!.ok, true)
})

/** A Dodge snapshot for the pure tests: the drone at cruise, 2 m up, one obstacle ahead. */
function dodgeView(over: Partial<DodgeView> = {}, drone: Partial<DodgeView['drone']> = {}): DodgeView {
  return {
    drone: { x: 10, y: 0, z: 2, v: 11, hl: 0.5, hw: 0.5, hh: 0.3, ...drone },
    obstacles: [{ x0: 30, x1: 34, top: 4, deck: null, gap: null, impassable: false, passed: false }],
    finish_x: 400,
    y_max: 5,
    z_max: 18,
    frame: 100,
    passed: 0,
    action: 'continue',
    vetoed: false,
    vetoed_request: null,
    crashed: false,
    finished: false,
    aborted: false,
    ...over,
  }
}

test('the Dodge camera keeps 8 m behind the drone and at least 44 m of course in view', () => {
  for (const [w, h] of [[800, 500], [358, 224], [1200, 750]]) {
    for (const zView of [8, 12, 20]) {
      const v = dodgeView({}, { x: 123.4 })
      const lay = dodgeLayout(w, h, v, zView)
      assert.ok(Math.abs(lay.X(v.drone.x) - DODGE_BEHIND_M * lay.k) < 1e-9)
      assert.ok(w / lay.k >= DODGE_MIN_VIEW_M - 1e-9, `${w}x${h}, ${zView} m: ${w / lay.k} m in view`)
      assert.ok(w / lay.k - DODGE_BEHIND_M >= 36 - 1e-9, 'at least 36 m ahead')
    }
  }
})

test('the Dodge trail adds one point per frame, forgets after 4 s and restarts with an episode', () => {
  let trail: DodgeTrailPoint[] = []
  const at = (frame: number, x: number, action = 'continue') => dodgeView({ frame, action }, { x, z: 2 + frame / 100 })
  trail = nextDodgeTrail(trail, at(1, 0.2))
  trail = nextDodgeTrail(trail, at(2, 0.4, 'over'))
  assert.deepEqual(
    trail.map((p) => [p.f, p.a]),
    [
      [1, 'continue'],
      [2, 'over'],
    ],
  )
  // Drawing the same frame again (paused) adds nothing and returns the same trail.
  const again = nextDodgeTrail(trail, at(2, 0.4, 'over'))
  assert.equal(again, trail)
  assert.equal(again.length, 2)
  // The helper never changes the trail it is given.
  const before = trail.slice()
  const later = nextDodgeTrail(trail, at(6, 1.2))
  assert.deepEqual(trail, before)
  assert.equal(later.length, 3)
  // Points older than 4 s (240 frames) are dropped.
  let t2 = later
  for (let f = 7; f <= 400; f += 3) t2 = nextDodgeTrail(t2, at(f, f * 0.2))
  const last = t2[t2.length - 1]
  assert.equal(last.f, 400)
  assert.ok(t2.every((p) => last.f - p.f <= DODGE_TRAIL_FRAMES))
  assert.ok(t2[0].f <= last.f - DODGE_TRAIL_FRAMES + 3)
  // A new episode: the frame goes back, the trail starts again.
  const fresh = nextDodgeTrail(t2, at(2, 0.4))
  assert.deepEqual(
    fresh.map((p) => p.f),
    [2],
  )
  // Same frame number but another position: also a new episode.
  assert.equal(nextDodgeTrail([{ f: 2, x: 9, y: 0, z: 2, a: 'continue' }], at(2, 0.4)).length, 1)
})

test('the Dodge chip names the held maneuver, with the side of the gap for around', () => {
  const gapLeft = { x0: 30, x1: 34, top: 12, deck: null, gap: [2, 5] as [number, number], impassable: false, passed: false }
  const gapRight = { ...gapLeft, gap: [-5, -2] as [number, number] }
  const passed = { ...gapRight, x0: 0, x1: 4, passed: true }
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'continue' })), 'continue · level')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'over' })), 'over')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'brake' })), 'brake')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'abort' })), 'abort')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'around', obstacles: [passed, gapLeft] })), 'around · left gap')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'around', obstacles: [passed, gapRight] })), 'around · right gap')
  assert.equal(dodgeManeuverLabel(dodgeView({ action: 'around' })), 'around · no gap ahead')
  assert.equal(dodgeNextObstacle(dodgeView({ obstacles: [passed, gapLeft] })), gapLeft)
})

test('an obstacle is out of the way at the drone altitude only above its body and under its deck', () => {
  const o = { x0: 30, x1: 34, top: 4, deck: null, gap: null, impassable: false, passed: false }
  const d = dodgeView().drone
  // The core's body test is z − hh < top: at z − hh = top exactly it does not touch.
  assert.equal(dodgeAboveBody(o, { ...d, z: 4.3 }), true)
  assert.equal(dodgeAboveBody(o, { ...d, z: 4.29 }), false)
  assert.equal(dodgeClearsAtAltitude(o, { ...d, z: 6 }), true)
  // The core's deck test is z + hh > ceil: above the underside, the deck is in the way.
  const decked = { ...o, deck: [29, 35, 6] as [number, number, number] }
  assert.equal(dodgeClearsAtAltitude(decked, { ...d, z: 5.7 }), true)
  assert.equal(dodgeClearsAtAltitude(decked, { ...d, z: 5.71 }), false)
  assert.equal(dodgeClearsAtAltitude(decked, { ...d, z: 3 }), false)
})

test('Dodge clearances: above a body, under a deck, nothing beside a body or out of its span', () => {
  const block = { x0: 30, x1: 34, top: 4, deck: [29, 35, 7] as [number, number, number], gap: null, impassable: false, passed: false }
  // Over the body and under the deck: both clearances, from the drone's box.
  const both = dodgeClearances(dodgeView({ obstacles: [block] }, { x: 32, z: 5.5 }))
  assert.deepEqual(
    both.map((c) => [c.kind, Number(c.m.toFixed(6))]),
    [
      ['above', 1.2],
      ['under', 1.2],
    ],
  )
  // On the deck overhang only (x-span past the body's end, inside the deck's): the deck only.
  const overhang = dodgeClearances(dodgeView({ obstacles: [block] }, { x: 34.6, z: 5.5 }))
  assert.deepEqual(
    overhang.map((c) => c.kind),
    ['under'],
  )
  // The x-span uses the core's strict test: touching at x + hl = x0 is not an overlap.
  assert.equal(dodgeOverlapsX({ ...dodgeView().drone, x: 28.5 }, 29, 35), false)
  assert.equal(dodgeClearances(dodgeView({ obstacles: [block] }, { x: 28.5, z: 5.5 })).length, 0)
  // Beside a tall body, through its gap: no vertical clearance.
  const tower = { x0: 30, x1: 34, top: 12, deck: null, gap: [2, 5] as [number, number], impassable: false, passed: false }
  assert.equal(dodgeClearances(dodgeView({ obstacles: [tower] }, { x: 32, y: 3.4 })).length, 0)
  // After a crash nothing is shown.
  assert.equal(dodgeClearances(dodgeView({ obstacles: [block], crashed: true }, { x: 32, z: 5.5 })).length, 0)
  // Rounded down, never overstated.
  assert.equal(dodgeMeters(0.29999), '0.2')
  assert.equal(dodgeMeters(0.3000000119), '0.3')
  assert.equal(dodgeMeters(1.25), '1.2')
  assert.equal(dodgeMeters(0), '0.0')
})

test('wind arrows point the way the core blows: north up, south down, east right, west left', () => {
  // sky_games::rescue: codes 2..=5 are wind toward N, S, E, W; north is row - 1.
  assert.deepEqual(WIND_DIRS[2], [0, -1])
  assert.deepEqual(WIND_DIRS[3], [0, 1])
  assert.deepEqual(WIND_DIRS[4], [1, 0])
  assert.deepEqual(WIND_DIRS[5], [-1, 0])
})

test('crash and death events count as crashes, nothing else does', () => {
  assert.equal(endedInCrash('crashed'), true)
  assert.equal(endedInCrash('died'), true)
  for (const e of ['completed', 'aborted', 'timeout', 'none', 'passed']) assert.equal(endedInCrash(e), false)
})

test('the first press made in AI mode switches to human mode and still acts', () => {
  // The page: a press in AI mode sets mode = human; the mode watcher builds a new runner and
  // replays the kept press on it. A press in human mode goes straight to the runner.
  const first = new FirstPress()
  let mode: 'ai' | 'human' = 'ai'
  const received: string[][] = [[]] // presses seen by each runner
  const act = (key: string) => {
    if (first.press(mode, key)) {
      mode = 'human'
      received.push([]) // the watcher's new runner
      const k = first.take()
      if (k) received[received.length - 1].push(k)
    } else received[received.length - 1].push(key)
  }
  act('east') // an on-screen arrow, an arrow key or a swipe, all go through act()
  act('north')
  assert.deepEqual(received, [[], ['east', 'north']])
  assert.equal(first.take(), null, 'replayed once')
})

test('keys and swipes map to the moves they point to', () => {
  assert.equal(actionForKey('rescue', 'ArrowUp'), 'north')
  assert.equal(actionForKey('rescue', 'ArrowLeft'), 'west')
  assert.equal(actionForKey('rescue', 'D'), 'east')
  assert.equal(actionForKey('dodge', 'ArrowDown'), 'brake')
  assert.equal(actionForKey('flappy', ' '), 'thrust')
  assert.equal(actionForKey('doom', 'ArrowRight'), 'turn_right')
  assert.equal(actionForKey('rescue', 'x'), undefined)
  // Screen y grows downward: a swipe down is south.
  assert.equal(actionForSwipe('rescue', 0, 60), 'south')
  assert.equal(actionForSwipe('rescue', 0, -60), 'north')
  assert.equal(actionForSwipe('rescue', -60, 10), 'west')
  assert.equal(actionForSwipe('dodge', 0, -60), 'over')
  assert.equal(actionForSwipe('rescue', 5, 5), undefined, 'a tap is not a swipe')
  assert.equal(actionForSwipe('flappy', 0, 60), undefined)
})

// ------------------------------------------------------------------------------------------
// Sessions of the browser module.

const hasPkg = existsSync(join(pkg, 'sky_web.js')) && existsSync(join(pkg, 'sky_web_bg.wasm'))
let modP: Promise<any> | null = null
function wasm(): Promise<any> {
  modP ??= (async () => {
    const mod = await import(join(pkg, 'sky_web.js'))
    await mod.default({ module_or_path: readFileSync(join(pkg, 'sky_web_bg.wasm')) })
    return mod
  })()
  return modP
}

interface Tally {
  frames: number
  falseTouches: number
  crashes: number
  crashesShown: number
  beside: number
  /** Frames the drone overlaps a body along x outside its gap lane, without crashing, and how
   * many of them show an `above` clearance; the same for deck spans and `under`. */
  overBody: number
  aboveShown: number
  underDeck: number
  underShown: number
}

const tally = (): Tally => ({ frames: 0, falseTouches: 0, crashes: 0, crashesShown: 0, beside: 0, overBody: 0, aboveShown: 0, underDeck: 0, underShown: 0 })

/** Plays one Dodge episode the way the runner does and checks every frame's drawing. */
function playDodge(session: any, next: () => string, size: [number, number], t: Tally) {
  let vetoes = 0
  let trail: DodgeTrailPoint[] = []
  for (;;) {
    const r = JSON.parse(next())
    vetoes += r.vetoed ? 1 : 0
    const snap = JSON.parse(session.snapshot())
    const v = snap.view as DodgeView
    const lay = dodgeLayout(size[0], size[1], v, dodgeZoomTarget(v))
    const sprite = spriteBox(lay.X(v.drone.x), lay.Z(v.drone.z), lay.droneSize, dodgeTilt(v))
    const drawn = solids(v, lay)
    t.frames++
    t.beside += v.obstacles.some((o) => dodgeInGapLane(o, v.drone) && o.x0 < v.drone.x + 0.5 && o.x1 > v.drone.x - 0.5) ? 1 : 0
    // The trail is fed once per frame, in order, and keeps at most 4 s.
    const before = trail
    trail = nextDodgeTrail(trail, v)
    assert.notEqual(trail, before)
    assert.ok(trail.length <= DODGE_TRAIL_FRAMES + 1 && trail[trail.length - 1].f === v.frame)
    assert.equal(nextDodgeTrail(trail, v), trail, 'a paused redraw adds nothing')
    // Every clearance shown is free space the core agrees with, and one is shown on every frame
    // the drone flies over a body or under a deck.
    const shown = dodgeClearances(v)
    for (const c of shown) assert.ok(c.m >= 0, `${c.kind} ${c.m}`)
    if (!v.crashed) {
      for (const o of v.obstacles) {
        if (dodgeOverlapsX(v.drone, o.x0, o.x1) && !dodgeInGapLane(o, v.drone)) {
          t.overBody++
          t.aboveShown += shown.some((c) => c.kind === 'above' && c.o === o) ? 1 : 0
        }
        if (o.deck && dodgeOverlapsX(v.drone, o.deck[0], o.deck[1])) {
          t.underDeck++
          t.underShown += shown.some((c) => c.kind === 'under' && c.o === o) ? 1 : 0
        }
      }
    }
    // 1 px of tolerance: the tilted sprite reaches 4% past the hitbox (0.02 m).
    if (!v.crashed && drawn.some((b) => overlap(sprite, b, 1))) t.falseTouches++
    if (r.done) {
      assert.equal(snap.done, true)
      assert.equal(endedInCrash(r.event), snap.stats.crashed, `event ${r.event}`)
      assert.equal(vetoes, snap.stats.vetoes, 'the page counts the vetoes the core counts')
      assert.equal(session.score(), v.passed)
      if (v.crashed) {
        t.crashes++
        if (drawn.some((b) => overlap(hitbox(v, lay), b))) t.crashesShown++
      }
      return
    }
  }
}

const SIZES: [number, number][] = [
  [800, 500],
  [360, 225],
]

test('Dodge: the canvas never shows a touch the core does not count, and shows every crash', { skip: !hasPkg }, async () => {
  const mod = await wasm()
  for (const size of SIZES) {
    const t = tally()
    // The oracle with the reflex, as on the page without a model.
    const s = new mod.GameSession('dodge', 11)
    for (let ep = 0; ep < 30; ep++) {
      if (ep) s.reset()
      playDodge(s, () => s.step_oracle(), size, t)
    }
    // A pilot that always asks `continue`, reflex on: held in front of obstacles, never crashes.
    s.set_veto(true)
    for (let ep = 0; ep < 3; ep++) {
      s.reset()
      playDodge(s, () => s.step_human('continue'), size, t)
    }
    // Random keys without the reflex: crashes.
    s.set_veto(false)
    const next = rng(1)
    const keys = ['over', 'around', 'brake', 'continue']
    for (let ep = 0; ep < 30; ep++) {
      s.reset()
      let held = 'continue'
      playDodge(
        s,
        () => {
          // A new key about every 40 frames.
          if (next() < 1 / 40) held = keys[Math.floor(next() * 4)]
          return s.step_human(held)
        },
        size,
        t,
      )
    }
    s.free()
    assert.ok(t.crashes > 0 && t.beside > 0 && t.overBody > 0 && t.underDeck > 0, JSON.stringify(t))
    assert.equal(t.aboveShown, t.overBody, `${size}: ${JSON.stringify(t)}`)
    assert.equal(t.underShown, t.underDeck, `${size}: ${JSON.stringify(t)}`)
    assert.equal(t.falseTouches, 0, `${size}: ${JSON.stringify(t)}`)
    assert.equal(t.crashesShown, t.crashes, `${size}: ${JSON.stringify(t)}`)
  }
})

test('Dodge: a model with the reflex plays without a visible touch', { skip: !hasPkg || !existsSync(join(models, 's3-pico.gguf')) }, async () => {
  const mod = await wasm()
  const engine = new mod.Engine(new Uint8Array(readFileSync(join(models, 's3-pico.gguf'))))
  const t = tally()
  const s = new mod.GameSession('dodge', 5)
  for (let ep = 0; ep < 6; ep++) {
    if (ep) s.reset()
    playDodge(s, () => s.step_ai(engine), SIZES[0], t)
  }
  s.free()
  engine.free()
  assert.equal(t.falseTouches, 0, JSON.stringify(t))
  assert.equal(t.crashesShown, t.crashes)
  assert.equal(t.aboveShown, t.overBody, JSON.stringify(t))
  assert.equal(t.underShown, t.underDeck, JSON.stringify(t))
})

test('Rescue: arrow keys move the drone where they point, the drawn wind sets the cost', { skip: !hasPkg }, async () => {
  const mod = await wasm()
  // The page's key map (controls.ts) and the screen direction of each key.
  const KEYS: Record<string, [string, number, number]> = {
    ArrowUp: [actionForKey('rescue', 'ArrowUp')!, -1, 0],
    ArrowDown: [actionForKey('rescue', 'ArrowDown')!, 1, 0],
    ArrowRight: [actionForKey('rescue', 'ArrowRight')!, 0, 1],
    ArrowLeft: [actionForKey('rescue', 'ArrowLeft')!, 0, -1],
  }
  const names = Object.keys(KEYS)
  let [moves, against, vetoed, reversed, crashes] = [0, 0, 0, 0, 0]
  for (const veto of [true, false]) {
    const s = new mod.GameSession('rescue', 3)
    s.set_veto(veto)
    const next = rng(7)
    for (let ep = 0; ep < 40; ep++) {
      if (ep) s.reset()
      for (;;) {
        const before = JSON.parse(s.snapshot()).view
        const [action, dr, dc] = KEYS[names[Math.floor(next() * 4)]]
        const r = JSON.parse(s.step_human(action))
        const after = JSON.parse(s.snapshot()).view
        const [r0, c0] = before.drone
        const [r1, c1] = after.drone
        if (r.vetoed) {
          vetoed++
          assert.equal(r.executed, after.last_action)
          // `return_home` may fly against the arrow pressed; `hover` stays.
          if (r1 - r0 === -dr && c1 - c0 === -dc && (dr || dc)) reversed++
        } else if (!after.crashed) {
          moves++
          assert.deepEqual([r1 - r0, c1 - c0], [dr, dc], 'the drone lands where the arrow points')
          const code = after.cells[r1 * after.n + c1]
          const arrow = WIND_DIRS[code]
          // The arrow drawn on the entered cell points against the move: 3, else 1.
          const opposed = !!arrow && arrow[0] === -dc && arrow[1] === -dr
          against += opposed ? 1 : 0
          assert.equal(rescueHeadwind(after), opposed, 'the headwind cue')
          const recharged = after.drone[0] === after.base[0] && after.drone[1] === after.base[1]
          if (!recharged) assert.equal(before.battery - after.battery, opposed ? 3 : 1, `wind ${code}`)
        }
        if (r.done) {
          const stats = JSON.parse(s.snapshot()).stats
          assert.equal(endedInCrash(r.event), stats.crashed)
          crashes += stats.crashed ? 1 : 0
          break
        }
      }
    }
    s.free()
  }
  assert.ok(moves > 100 && against > 0 && vetoed > 0 && crashes > 0, JSON.stringify({ moves, against, vetoed, reversed, crashes }))
})

test('Flappy: hitting a pylon or the ground ends the run as a counted crash', { skip: !hasPkg }, async () => {
  const mod = await wasm()
  const s = new mod.GameSession('flappy', 2)
  for (let ep = 0; ep < 5; ep++) {
    if (ep) s.reset()
    for (;;) {
      // Thrust on every frame (up into the sky's top or a pylon) or never (down to the ground).
      const r = JSON.parse(s.step_human(ep % 2 ? '' : 'thrust'))
      if (r.done) {
        const snap = JSON.parse(s.snapshot())
        assert.equal(r.event, 'crashed')
        assert.equal(snap.stats.crashed, true)
        assert.equal(snap.view.alive, false)
        break
      }
    }
  }
  s.free()
})
