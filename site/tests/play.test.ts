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
  DODGE_CAM_BACK,
  DODGE_CAM_LIFT,
  DODGE_DECK_NEAR_M,
  DODGE_FOV_DEG,
  DODGE_MAX_TILT,
  DODGE_TRAIL_FRAMES,
  WIND_DIRS,
  dodgeAboveBody,
  dodgeBoxFaces,
  dodgeCamera,
  dodgeCannotHide,
  dodgeClearances,
  dodgeClipBox,
  dodgeDrawOrder,
  dodgeDrawnBoxes,
  dodgeDroneBox,
  dodgeDroneModelPoints,
  dodgeFocal,
  dodgeInGapLane,
  dodgeManeuverLabel,
  dodgeMeters,
  dodgeNextObstacle,
  dodgeOutcome,
  dodgeOverlapsX,
  dodgeProject,
  dodgeRoll,
  dodgeSceneBoxes,
  dodgeScreenBounds,
  dodgeSplit,
  nextDodgeCamPose,
  nextDodgeTrail,
  rescueHeadwind,
  type DodgeBox,
  type DodgeCamPose,
  type DodgeTrailPoint,
  type DodgeView,
} from '../.vitepress/theme/lib/render.ts'
import { endedInCrash } from '../.vitepress/theme/lib/events.ts'
import { FirstPress, actionForKey, actionForSwipe } from '../.vitepress/theme/lib/controls.ts'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '../public/pkg')
const models = join(here, '../public/models')

type Pt = [number, number]

/** A seeded generator in [0, 1) (mulberry32): scripted pilots press reproducible keys. */
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** How deep two convex polygons overlap on screen (px; 0 or less: they do not), by separating
 * axes: the least overlap of their projections on every edge normal of either. */
function overlapDepth(a: Pt[], b: Pt[]): number {
  let best = Infinity
  for (const poly of [a, b]) {
    for (let i = 0; i < poly.length; i++) {
      const [p, q] = [poly[i], poly[(i + 1) % poly.length]]
      const len = Math.hypot(q[0] - p[0], q[1] - p[1])
      if (len < 1e-9) continue
      const [nx, ny] = [(q[1] - p[1]) / len, (p[0] - q[0]) / len]
      const pa = a.map((r) => r[0] * nx + r[1] * ny)
      const pb = b.map((r) => r[0] * nx + r[1] * ny)
      best = Math.min(best, Math.min(Math.max(...pa), Math.max(...pb)) - Math.max(Math.min(...pa), Math.min(...pb)))
    }
  }
  return best
}

const rect = (r: { x0: number; x1: number; y0: number; y1: number }): Pt[] => [
  [r.x0, r.y0],
  [r.x1, r.y0],
  [r.x1, r.y1],
  [r.x0, r.y1],
]

// ------------------------------------------------------------------------------------------
// Pure helpers.

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

test('the chase projection: the drone ahead of the camera, a 70° field, far points on the horizon', () => {
  const v = dodgeView({ obstacles: [] }, { x: 50, y: 1, z: 3 })
  const pose = nextDodgeCamPose(null, v)
  // A first pose sits rigidly behind and above the drone.
  assert.equal(pose.y, 1)
  assert.ok(Math.abs(pose.z - (3 + DODGE_CAM_LIFT)) < 1e-9)
  for (const [w, h] of [
    [800, 500],
    [358, 224],
  ]) {
    const { mainH } = dodgeSplit(h)
    const c = dodgeCamera(w, mainH, v, pose)
    assert.equal(c.x, 50 - DODGE_CAM_BACK)
    assert.equal(c.near, 50 - 0.5, 'the near plane is the back of the drone box')
    // The drone's center: straight ahead laterally, LIFT below the camera, at its distance.
    const [u, vv] = dodgeProject(c, 50, 1, 3)
    assert.ok(Math.abs(u - w / 2) < 1e-9)
    assert.ok(Math.abs(vv - (c.v0 + (c.f * DODGE_CAM_LIFT) / DODGE_CAM_BACK)) < 1e-9)
    // Its hitbox spans f · 2hw / distance pixels: several dozen on a desktop canvas.
    const width = dodgeProject(c, 50, 0.5, 3)[0] - dodgeProject(c, 50, 1.5, 3)[0]
    assert.ok(Math.abs(width - c.f / DODGE_CAM_BACK) < 1e-9)
    assert.ok(width > (w === 800 ? 60 : 30), `${width} px`)
    // The field of view: a point tan(35°) of its distance to the side is on the canvas's edge.
    assert.ok(Math.abs(dodgeFocal(w) - w / 2 / Math.tan((DODGE_FOV_DEG * Math.PI) / 360)) < 1e-9)
    for (const dx of [3, 20, 90]) {
      const [ue] = dodgeProject(c, c.x + dx, c.y + dx * Math.tan((DODGE_FOV_DEG * Math.PI) / 360), c.z + 2)
      assert.ok(Math.abs(ue) < 1e-6, `${dx} m: ${ue}`)
    }
    // Farther points converge to the horizon's middle (u0, v0), monotonically.
    let last = Infinity
    for (const x of [52, 60, 80, 120, 400, 1e5]) {
      const [a, b] = dodgeProject(c, x, -4, 12)
      const r = Math.hypot(a - c.u0, b - c.v0)
      assert.ok(r < last, `${x}: ${r}`)
      last = r
    }
    assert.ok(last < 0.2)
    // 60 m ahead of the drone the corridor's 10 m are still a readable width.
    const span = dodgeProject(c, 110, -5, 0)[0] - dodgeProject(c, 110, 5, 0)[0]
    assert.ok(span > (w === 800 ? 60 : 28), `${span} px`)
  }
})

test('the near plane: a box behind the drone is not drawn, a box it is in is cut at its back', () => {
  const b: DodgeBox = { x0: 8, x1: 14, y0: -5, y1: 2, z0: 0, z1: 9, kind: 'body', o: 0, solid: true }
  assert.equal(dodgeClipBox({ ...b, x1: 9.5 }, 10), null)
  assert.equal(dodgeClipBox({ ...b, x1: 10 }, 10), null, 'a box ending at the plane is gone')
  assert.deepEqual(dodgeClipBox(b, 10), { ...b, x0: 10, cut: true })
  const ahead = { ...b, x0: 12 }
  assert.equal(dodgeClipBox(ahead, 10), ahead)
  // In a scene: the drone in the gap lane beside a tall body, past its front.
  const tower = { x0: 30, x1: 36, top: 12, deck: null, gap: [2, 5] as [number, number], impassable: false, passed: false }
  const v = dodgeView({ obstacles: [tower] }, { x: 33, y: 3.4 })
  const c = dodgeCamera(800, 370, v, nextDodgeCamPose(null, v))
  const drawn = dodgeDrawnBoxes(v, c)
  assert.ok(drawn.every((d) => d.x0 >= c.near && d.x1 > c.near))
  const body = drawn.find((d) => d.kind === 'body')!
  assert.deepEqual([body.x0, body.cut, body.y0, body.y1], [c.near, true, -5, 2])
  // The body has no box in its gap lane; a deck fills the width from its underside up to the ceiling.
  const decked = dodgeView({ obstacles: [{ ...tower, deck: [29, 37, 12.6] }], finish_x: 60 })
  assert.deepEqual(
    dodgeSceneBoxes(decked)
      .filter((d) => d.solid)
      .map((d) => [d.kind, d.x0, d.x1, d.y0, d.y1, d.z0, d.z1]),
    [
      ['body', 30, 36, -5, 2, 0, 12],
      ['deck', 29, 37, -5, 5, 12.6, 18],
    ],
  )
  // Pillars, fences and gate posts stand outside the corridor; the gate's banner above the airspace.
  const extras = dodgeSceneBoxes(decked).filter((d) => !d.solid)
  assert.deepEqual([...new Set(extras.map((d) => d.kind))], ['pillar', 'fence', 'gate'])
  for (const d of extras) assert.ok(d.y0 >= 5 || d.y1 <= -5 || d.z0 >= 18, JSON.stringify(d))
})

test('the painter order: a box is drawn before every box that could hide it', () => {
  const box = (x0: number, x1: number, y0: number, y1: number, z0: number, z1: number): DodgeBox => ({ x0, x1, y0, y1, z0, z1, kind: 'body', o: 0, solid: true })
  const near = box(20, 24, -5, 5, 0, 6)
  const far = box(50, 54, -5, 5, 0, 6)
  const body = box(70, 74, -5, 5, 0, 4)
  const deck = box(69, 75, -5, 5, 7, 18)
  const cam = (y: number, z: number) => ({ x: 0, y, z })
  // Along x: the farther first.
  assert.deepEqual(dodgeDrawOrder([near, far], cam(0, 3)), [far, near])
  // A deck over its body: from below the body's top the body can hide the deck, from above the
  // deck's underside the deck can hide the body, from between neither hides the other.
  assert.equal(dodgeCannotHide(deck, body, cam(0, 3)), true)
  assert.equal(dodgeCannotHide(body, deck, cam(0, 3)), false)
  assert.deepEqual(dodgeDrawOrder([body, deck], cam(0, 3)), [deck, body])
  assert.deepEqual(dodgeDrawOrder([deck, body], cam(0, 17.9)), [body, deck])
  assert.equal(dodgeCannotHide(deck, body, cam(0, 5)) && dodgeCannotHide(body, deck, cam(0, 5)), true)
  // Two pieces of a body around a camera in the middle gap: neither hides the other.
  const right = box(30, 34, -5, -1, 0, 9)
  const left = box(30, 34, 1, 5, 0, 9)
  assert.equal(dodgeCannotHide(right, left, cam(0, 3)) && dodgeCannotHide(left, right, cam(0, 3)), true)
  // A larger scene, in any order: no box is drawn before one it could hide.
  const scene = dodgeSceneBoxes(
    dodgeView(
      {
        finish_x: 120,
        obstacles: [
          { x0: 20, x1: 24, top: 6, deck: [19, 25, 9], gap: [2, 5], impassable: false, passed: false },
          { x0: 50, x1: 55, top: 12, deck: [49, 56, 12.5], gap: [-5, -1.5], impassable: false, passed: false },
          { x0: 80, x1: 84, top: 14, deck: [79, 85, 14.4], gap: null, impassable: true, passed: false },
        ],
      },
      { x: 10 },
    ),
  )
  const next = rng(3)
  for (const c of [cam(0, 3.4), cam(3.5, 2), cam(-3, 8), cam(4.9, 17.9), cam(-4.9, 0.3)]) {
    const at = { ...c, x: 3 }
    const order = dodgeDrawOrder(
      scene.slice().sort(() => next() - 0.5),
      at,
    )
    assert.equal(order.length, scene.length)
    for (let i = 0; i < order.length; i++) {
      for (let j = i + 1; j < order.length; j++) assert.ok(dodgeCannotHide(order[i], order[j], at), `${JSON.stringify(c)}: ${i} before ${j}`)
    }
  }
  // Faces: the front always; a side, the top or the bottom only when the camera is beyond it.
  const c = dodgeCamera(800, 370, dodgeView({ obstacles: [] }), { y: 0, z: 3 })
  const sides = (b: DodgeBox) => dodgeBoxFaces(b, c).map((f) => f.side)
  assert.deepEqual(sides(box(20, 24, 1, 5, 0, 2)), ['front', 'right', 'top'])
  assert.deepEqual(sides(box(20, 24, -5, -1, 4, 9)), ['front', 'left', 'bottom'])
  assert.deepEqual(sides(box(20, 24, -5, 5, 0, 3)), ['front'])
})

test('the camera stays where nothing can come between it and the drone', () => {
  const lane = { x0: 30, x1: 36, top: 12, deck: [29, 37, 12.6] as [number, number, number], gap: [2, 5] as [number, number], impassable: false, passed: false }
  // Beside the body, in its gap lane: the camera is brought into the lane, wherever it was.
  const v = dodgeView({ obstacles: [lane], frame: 101 }, { x: 33, y: 2.6, z: 2 })
  const pose = nextDodgeCamPose({ f: 100, x: 32.8, y: -1, z: 3.4 }, v)
  assert.ok(pose.y >= 2 && pose.y <= 5, `${pose.y}`)
  // Under a deck: the camera under its underside, still above the drone's center.
  const bridge = { x0: 30, x1: 34, top: 2, deck: [29, 35, 4.5] as [number, number, number], gap: null, impassable: false, passed: false }
  const under = dodgeView({ obstacles: [bridge], frame: 101 }, { x: 31, z: 3.2 })
  const pu = nextDodgeCamPose({ f: 100, x: 30.8, y: 0, z: 4.6 }, under)
  assert.ok(pu.z <= 4.5 && pu.z > under.drone.z, `${pu.z}`)
  // Above a body: the camera above its top.
  const block = { x0: 30, x1: 34, top: 6, deck: null, gap: null, impassable: false, passed: false }
  const over = dodgeView({ obstacles: [block], frame: 101 }, { x: 31, z: 6.8 })
  const po = nextDodgeCamPose({ f: 100, x: 30.8, y: 0, z: 5.5 }, over)
  assert.ok(po.z >= 6, `${po.z}`)
  // Free of obstacles it lags behind the drone's moves, so they show on screen.
  const free = dodgeView({ obstacles: [], frame: 101 }, { x: 31, y: 1, z: 2 })
  const pf = nextDodgeCamPose({ f: 100, x: 30.8, y: 0, z: 3.4 }, free)
  assert.ok(pf.y > 0 && pf.y < 0.2, `${pf.y}`)
  // A paused redraw keeps the pose; a new episode starts it on the drone.
  assert.equal(nextDodgeCamPose(pf, free), pf)
  const restart = nextDodgeCamPose(pf, dodgeView({ obstacles: [], frame: 1 }, { x: 0.2, y: 0, z: 2 }))
  assert.deepEqual([restart.y, restart.z], [0, 2 + DODGE_CAM_LIFT])
})

test('the drone model stays inside its hitbox and spans its width', () => {
  const d = dodgeView().drone
  for (const pitch of [-DODGE_MAX_TILT, 0, DODGE_MAX_TILT]) {
    for (const roll of [-DODGE_MAX_TILT, 0, DODGE_MAX_TILT]) {
      for (const [x, y, z] of dodgeDroneModelPoints(d, pitch, roll)) {
        assert.ok(Math.abs(x) <= d.hl && Math.abs(y) <= d.hw && Math.abs(z) <= d.hh, `${pitch} ${roll}: ${[x, y, z]}`)
      }
    }
  }
  const ys = dodgeDroneModelPoints(d, 0, 0).map((p) => p[1])
  assert.ok(Math.max(...ys) - Math.min(...ys) >= 0.97 * 2 * d.hw)
  // It banks into a sideways move, within the tilt the model allows.
  const trail = (dy: number): DodgeTrailPoint[] => [0, 1, 2, 3].map((f) => ({ f, x: f * 0.2, y: f * dy, z: 2, a: 'around' }))
  assert.ok(dodgeRoll(trail(0.066)) < 0 && dodgeRoll(trail(0.066)) >= -DODGE_MAX_TILT)
  assert.ok(dodgeRoll(trail(-0.066)) > 0)
  assert.equal(dodgeRoll(trail(0)), 0)
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

test('a body is out of the way only at or above its top, as the core tests it', () => {
  const o = { x0: 30, x1: 34, top: 4, deck: null, gap: null, impassable: false, passed: false }
  const d = dodgeView().drone
  // The core's body test is z − hh < top: at z − hh = top exactly it does not touch.
  assert.equal(dodgeAboveBody(o, { ...d, z: 4.3 }), true)
  assert.equal(dodgeAboveBody(o, { ...d, z: 4.29 }), false)
})

test('Dodge clearances: above a body, under a deck, beside a body in its gap lane, none out of its span', () => {
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
  // A deck more than 3 m above is not one the drone squeezes under: no read-out.
  assert.deepEqual(
    dodgeClearances(dodgeView({ obstacles: [{ ...block, deck: [29, 35, 9.2] }] }, { x: 32, z: 5.5 })).map((c) => c.kind),
    ['above'],
  )
  assert.equal(DODGE_DECK_NEAR_M, 3)
  // On the deck overhang only (x-span past the body's end, inside the deck's): the deck only.
  const overhang = dodgeClearances(dodgeView({ obstacles: [block] }, { x: 34.6, z: 5.5 }))
  assert.deepEqual(
    overhang.map((c) => c.kind),
    ['under'],
  )
  // The x-span uses the core's strict test: touching at x + hl = x0 is not an overlap.
  assert.equal(dodgeOverlapsX({ ...dodgeView().drone, x: 28.5 }, 29, 35), false)
  assert.equal(dodgeClearances(dodgeView({ obstacles: [block] }, { x: 28.5, z: 5.5 })).length, 0)
  // Beside a tall body, through its gap: no vertical clearance, the distance to its side.
  const tower = { x0: 30, x1: 34, top: 12, deck: null, gap: [2, 5] as [number, number], impassable: false, passed: false }
  const kinds = (v: DodgeView) => dodgeClearances(v).map((c) => [c.kind, Number(c.m.toFixed(6)), c.edge])
  assert.deepEqual(kinds(dodgeView({ obstacles: [tower] }, { x: 32, y: 3.4 })), [['beside', 0.9, 2]])
  assert.deepEqual(kinds(dodgeView({ obstacles: [{ ...tower, gap: [-5, -1.5] }] }, { x: 32, y: -3 })), [['beside', 1, -1.5]])
  // Over the body, in its lane or not: the height above it.
  assert.deepEqual(kinds(dodgeView({ obstacles: [{ ...tower, top: 3 }] }, { x: 32, y: 3.4, z: 4 })), [['above', 0.7, undefined]])
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
  crashes: number
  /** Crash frames on which the drone, at its depth, touches the drawn front of a box it hit. */
  crashesShown: number
  /** Non-crash frames on which the drone touches, at its depth, a drawn solid face. */
  falseTouches: number
  /** Non-crash frames on which a drawn face that could be in front of the drone overlaps it. */
  covered: number
  /** Pairs drawn in an order that paints a box over one that could hide it. */
  orderErrors: number
  /** Frames on which the drone's projected box leaves the chase view. */
  outOfView: number
  /** Frames beside a body in its gap lane (below its top), over a body, under a deck, and how
   * many of them show the matching clearance (a deck within 3 m above). */
  beside: number
  besideShown: number
  overBody: number
  aboveShown: number
  underDeck: number
  underShown: number
}

const tally = (): Tally => ({
  frames: 0,
  crashes: 0,
  crashesShown: 0,
  falseTouches: 0,
  covered: 0,
  orderErrors: 0,
  outOfView: 0,
  beside: 0,
  besideShown: 0,
  overBody: 0,
  aboveShown: 0,
  underDeck: 0,
  underShown: 0,
})

/** Less than this overlap on screen (px) is a shared edge, not a touch. */
const TOL = 0.5

/**
 * Plays one Dodge episode the way the runner does and checks every frame's chase view:
 * - the painter order never paints a box over one that could hide it;
 * - on a frame that is not a crash, every drawn face that overlaps the drone's projected box
 *   belongs to a box that cannot hide the drone (a plane separates them with the camera on the
 *   drone's side), so drawing the drone last shows it as it is: nothing is in front of it;
 * - at the drone's depth, the drawn front (or near-plane section) of every body or deck its
 *   x-span overlaps does not meet the drone's cross-section, nor, for a section, its whole
 *   projected box, unless the core counts a crash; and on a crash frame one of them does;
 * - the drone's projected box stays in the chase view.
 */
function playDodge(session: any, next: () => string, size: [number, number], t: Tally) {
  let vetoes = 0
  let trail: DodgeTrailPoint[] = []
  let pose: DodgeCamPose | null = null
  const [w, h] = size
  const { mainH } = dodgeSplit(h)
  for (;;) {
    const r = JSON.parse(next())
    vetoes += r.vetoed ? 1 : 0
    const snap = JSON.parse(session.snapshot())
    const v = snap.view as DodgeView
    const d = v.drone
    t.frames++
    // The trail and the camera are fed once per frame, in order; a paused redraw changes nothing.
    const before = trail
    trail = nextDodgeTrail(trail, v)
    assert.notEqual(trail, before)
    assert.ok(trail.length <= DODGE_TRAIL_FRAMES + 1 && trail[trail.length - 1].f === v.frame)
    assert.equal(nextDodgeTrail(trail, v), trail, 'a paused redraw adds nothing')
    pose = nextDodgeCamPose(pose, v)
    assert.equal(nextDodgeCamPose(pose, v), pose, 'a paused redraw keeps the camera')
    const c = dodgeCamera(w, mainH, v, pose)
    const db = dodgeDroneBox(v)
    const P = dodgeScreenBounds(c, db)
    if (P.x0 < 0 || P.x1 > w || P.y0 < 0 || P.y1 > mainH) t.outOfView++
    const drawn = dodgeDrawnBoxes(v, c)
    for (let i = 0; i < drawn.length; i++) {
      for (let j = i + 1; j < drawn.length; j++) if (!dodgeCannotHide(drawn[i], drawn[j], c)) t.orderErrors++
    }
    let touched = false
    for (const b of drawn) {
      const faces = dodgeBoxFaces(b, c)
      if (!v.crashed && !dodgeCannotHide(b, db, c) && faces.some((f) => overlapDepth(f.pts, rect(P)) > TOL)) t.covered++
      if (!b.solid || !(b.x0 < db.x1 && b.x1 > db.x0)) continue
      // The box's drawn front is at the drone's depth (its section at the near plane, or its
      // facade inside the drone's x-span): the drone's cross-section there, projected.
      const front = faces[0].pts
      const section = rect(dodgeScreenBounds(c, { ...db, x0: b.x0, x1: b.x0 }))
      const depth = overlapDepth(front, section)
      if (v.crashed) touched ||= depth > 0
      else if (depth > TOL || (b.cut && overlapDepth(front, rect(P)) > TOL)) t.falseTouches++
    }
    // Every clearance shown is free space the core agrees with, and one is shown on every frame
    // the drone flies over a body, under a deck or beside a body in its gap lane.
    const shown = dodgeClearances(v)
    for (const cl of shown) assert.ok(cl.m >= 0, `${cl.kind} ${cl.m}`)
    if (!v.crashed) {
      for (const o of v.obstacles) {
        if (dodgeOverlapsX(d, o.x0, o.x1)) {
          if (dodgeAboveBody(o, d)) {
            t.overBody++
            t.aboveShown += shown.some((cl) => cl.kind === 'above' && cl.o === o) ? 1 : 0
          } else if (dodgeInGapLane(o, d)) {
            t.beside++
            t.besideShown += shown.some((cl) => cl.kind === 'beside' && cl.o === o) ? 1 : 0
          }
        }
        if (o.deck && dodgeOverlapsX(d, o.deck[0], o.deck[1]) && o.deck[2] - (d.z + d.hh) <= DODGE_DECK_NEAR_M) {
          t.underDeck++
          t.underShown += shown.some((cl) => cl.kind === 'under' && cl.o === o) ? 1 : 0
        }
      }
    }
    if (r.done) {
      assert.equal(snap.done, true)
      assert.equal(endedInCrash(r.event), snap.stats.crashed, `event ${r.event}`)
      assert.equal(vetoes, snap.stats.vetoes, 'the page counts the vetoes the core counts')
      assert.equal(session.score(), v.passed)
      if (v.crashed) {
        t.crashes++
        if (touched) t.crashesShown++
      }
      return
    }
  }
}

const SIZES: [number, number][] = [
  [800, 500],
  [358, 224],
]

function assertClean(t: Tally, what: string) {
  const s = `${what}: ${JSON.stringify(t)}`
  assert.equal(t.covered, 0, s)
  assert.equal(t.falseTouches, 0, s)
  assert.equal(t.orderErrors, 0, s)
  assert.equal(t.outOfView, 0, s)
  assert.equal(t.crashesShown, t.crashes, s)
  assert.equal(t.aboveShown, t.overBody, s)
  assert.equal(t.underShown, t.underDeck, s)
  assert.equal(t.besideShown, t.beside, s)
}

test('Dodge: nothing drawn hides the drone or shows it inside a building, and every crash shows', { skip: !hasPkg }, async () => {
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
    assertClean(t, `${size}`)
    console.log(`# dodge ${size}: ${JSON.stringify(t)}`)
  }
})

test('Dodge: a model with the reflex plays with the drone in clear view', { skip: !hasPkg || !existsSync(join(models, 's3-pico.gguf')) }, async () => {
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
  assertClean(t, 's3-pico')
  console.log(`# dodge s3-pico: ${JSON.stringify(t)}`)
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
