// Canvas renderers for the three games. They draw the JSON snapshots produced by the wasm
// module (`GameSession.snapshot().view`) and keep no game state of their own: only cosmetic
// state (particles, tweening) lives in `FxState`.

export type Theme = 'dark' | 'light'

export interface FxState {
  /** Seconds since start (for rotors, blinking and wind animation). */
  t: number
  /** Decays after each thrust, drives the air-flow effect. */
  thrustGlow: number
  /** Tween progress between the previous and the current rescue cell (0..1). */
  tween: number
  /** Time of the last crash (for the burst effect). */
  crashAt: number
  /** The dodge chase camera's pose, smoothed frame by frame (see [`nextDodgeCamPose`]). */
  dodgeCam: DodgeCamPose | null
  /** The dodge drone's recent positions, oldest first (see [`nextDodgeTrail`]). */
  dodgeTrail: DodgeTrailPoint[]
}

export function newFx(): FxState {
  return { t: 0, thrustGlow: 0, tween: 1, crashAt: -10, dodgeCam: null, dodgeTrail: [] }
}

export function currentTheme(): Theme {
  if (typeof document === 'undefined') return 'dark'
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

/** Resizes the backing store to the CSS size times devicePixelRatio; returns CSS size. */
export function fitCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } {
  const dpr = Math.min(window.devicePixelRatio || 1, 3)
  const w = Math.max(1, canvas.clientWidth)
  const h = Math.max(1, canvas.clientHeight)
  const bw = Math.round(w * dpr)
  const bh = Math.round(h * dpr)
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw
    canvas.height = bh
  }
  const ctx = canvas.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { ctx, w, h }
}

// Deterministic hash noise in [0, 1).
function hash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453
  return s - Math.floor(s)
}

function smoothNoise(x: number): number {
  const i = Math.floor(x)
  const f = x - i
  const u = f * f * (3 - 2 * f)
  return hash(i) * (1 - u) + hash(i + 1) * u
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v))

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

const PALETTE = {
  dark: {
    skyTop: '#0b1026',
    skyBottom: '#1e2a5a',
    far: '#1b2550',
    mid: '#151d3d',
    ground: '#0f172a',
    grass: '#22c55e',
    pipe: ['#334155', '#475569', '#1e293b'],
    cap: '#64748b',
    stripe: '#f59e0b',
    drone: '#e2e8f0',
    droneDark: '#94a3b8',
    text: '#f8fafc',
    tile: ['#14233a', '#16263f'],
    rock: ['#64748b', '#334155'],
    wind: 'rgba(56, 189, 248, 0.16)',
    windArrow: 'rgba(125, 211, 252, 0.85)',
    grid: 'rgba(148, 163, 184, 0.08)',
  },
  light: {
    skyTop: '#7dd3fc',
    skyBottom: '#e0f2fe',
    far: '#a5c8e4',
    mid: '#86b3d6',
    ground: '#65a30d',
    grass: '#4d7c0f',
    pipe: ['#64748b', '#94a3b8', '#475569'],
    cap: '#334155',
    stripe: '#f59e0b',
    drone: '#1e293b',
    droneDark: '#475569',
    text: '#0f172a',
    tile: ['#dcfce7', '#d1fae5'],
    rock: ['#94a3b8', '#64748b'],
    wind: 'rgba(14, 165, 233, 0.16)',
    windArrow: 'rgba(2, 132, 199, 0.85)',
    grid: 'rgba(15, 23, 42, 0.06)',
  },
}

/**
 * A quadcopter seen from the side: body, two visible arms with spinning rotors, status LEDs.
 * `size` is the body length in pixels (the rotors make it 1.84 `size` long); `tilt` in radians
 * (positive = nose down).
 */
export function drawDroneSide(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  tilt: number,
  t: number,
  theme: Theme,
  thrust = 0,
) {
  const p = PALETTE[theme]
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(tilt)
  const L = size
  // Air flow under the rotors.
  if (thrust > 0.02) {
    ctx.strokeStyle = theme === 'dark' ? `rgba(125,211,252,${0.55 * thrust})` : `rgba(14,116,144,${0.45 * thrust})`
    ctx.lineWidth = Math.max(1, L * 0.05)
    for (const sx of [-0.62, 0.62]) {
      for (let k = 0; k < 3; k++) {
        const ox = sx * L + (k - 1) * L * 0.12
        const len = L * (0.35 + 0.35 * thrust) * (1 + 0.3 * Math.sin(t * 40 + k))
        ctx.beginPath()
        ctx.moveTo(ox, L * 0.12)
        ctx.lineTo(ox, L * 0.12 + len)
        ctx.stroke()
      }
    }
  }
  // Arms.
  ctx.strokeStyle = p.droneDark
  ctx.lineWidth = Math.max(1.5, L * 0.08)
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(-0.62 * L, -0.12 * L)
  ctx.lineTo(0.62 * L, -0.12 * L)
  ctx.stroke()
  // Rotor masts and discs.
  for (const sx of [-0.62, 0.62]) {
    ctx.beginPath()
    ctx.moveTo(sx * L, -0.12 * L)
    ctx.lineTo(sx * L, -0.28 * L)
    ctx.stroke()
    ctx.fillStyle = theme === 'dark' ? 'rgba(148,163,184,0.28)' : 'rgba(51,65,85,0.22)'
    ctx.beginPath()
    ctx.ellipse(sx * L, -0.3 * L, 0.3 * L, 0.06 * L, 0, 0, Math.PI * 2)
    ctx.fill()
    // Blade: its apparent length oscillates as it spins.
    const phase = Math.cos(t * 55 + sx * 3)
    ctx.strokeStyle = p.drone
    ctx.lineWidth = Math.max(1, L * 0.045)
    ctx.beginPath()
    ctx.moveTo(sx * L - 0.3 * L * phase, -0.3 * L)
    ctx.lineTo(sx * L + 0.3 * L * phase, -0.3 * L)
    ctx.stroke()
    ctx.strokeStyle = p.droneDark
    ctx.lineWidth = Math.max(1.5, L * 0.08)
  }
  // Body.
  const grad = ctx.createLinearGradient(0, -0.25 * L, 0, 0.2 * L)
  grad.addColorStop(0, p.drone)
  grad.addColorStop(1, p.droneDark)
  ctx.fillStyle = grad
  roundRect(ctx, -0.42 * L, -0.24 * L, 0.84 * L, 0.36 * L, 0.14 * L)
  ctx.fill()
  // Camera dome and skids.
  ctx.fillStyle = '#0ea5e9'
  ctx.beginPath()
  ctx.arc(0.3 * L, 0.06 * L, 0.09 * L, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = p.droneDark
  ctx.lineWidth = Math.max(1, L * 0.04)
  ctx.beginPath()
  ctx.moveTo(-0.3 * L, 0.12 * L)
  ctx.lineTo(-0.36 * L, 0.24 * L)
  ctx.lineTo(0.36 * L, 0.24 * L)
  ctx.lineTo(0.3 * L, 0.12 * L)
  ctx.stroke()
  // LEDs: green at the front, blinking red at the back.
  ctx.fillStyle = '#22c55e'
  ctx.beginPath()
  ctx.arc(0.4 * L, -0.1 * L, 0.045 * L, 0, Math.PI * 2)
  ctx.fill()
  if (Math.sin(t * 9) > 0) {
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(-0.4 * L, -0.1 * L, 0.045 * L, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

/** A quadcopter seen from above (rescue grid). */
function drawDroneTop(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, t: number, theme: Theme) {
  const p = PALETTE[theme]
  ctx.save()
  ctx.translate(x, y)
  const r = size * 0.36
  ctx.strokeStyle = p.droneDark
  ctx.lineWidth = Math.max(2, size * 0.07)
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(-r, -r)
  ctx.lineTo(r, r)
  ctx.moveTo(r, -r)
  ctx.lineTo(-r, r)
  ctx.stroke()
  for (const [sx, sy] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ]) {
    ctx.fillStyle = theme === 'dark' ? 'rgba(148,163,184,0.25)' : 'rgba(51,65,85,0.2)'
    ctx.beginPath()
    ctx.arc(sx * r, sy * r, size * 0.2, 0, Math.PI * 2)
    ctx.fill()
    const a = t * 30 * (sx * sy)
    ctx.strokeStyle = p.drone
    ctx.lineWidth = Math.max(1, size * 0.04)
    ctx.beginPath()
    ctx.moveTo(sx * r + Math.cos(a) * size * 0.18, sy * r + Math.sin(a) * size * 0.18)
    ctx.lineTo(sx * r - Math.cos(a) * size * 0.18, sy * r - Math.sin(a) * size * 0.18)
    ctx.stroke()
  }
  ctx.fillStyle = p.drone
  roundRect(ctx, -size * 0.16, -size * 0.16, size * 0.32, size * 0.32, size * 0.08)
  ctx.fill()
  ctx.fillStyle = '#0ea5e9'
  ctx.beginPath()
  ctx.arc(0, -size * 0.06, size * 0.05, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function drawSky(ctx: CanvasRenderingContext2D, w: number, h: number, theme: Theme, t: number) {
  const p = PALETTE[theme]
  const g = ctx.createLinearGradient(0, 0, 0, h)
  g.addColorStop(0, p.skyTop)
  g.addColorStop(1, p.skyBottom)
  ctx.fillStyle = g
  ctx.fillRect(0, 0, w, h)
  if (theme === 'dark') {
    for (let i = 0; i < 70; i++) {
      const x = hash(i) * w
      const y = hash(i + 100) * h * 0.6
      const a = 0.35 + 0.35 * Math.sin(t * 1.5 + i)
      ctx.fillStyle = `rgba(255,255,255,${a})`
      ctx.fillRect(x, y, 1.5, 1.5)
    }
  } else {
    ctx.fillStyle = 'rgba(255,255,255,0.75)'
    for (let i = 0; i < 6; i++) {
      const x = ((hash(i) * w * 1.5 - t * (6 + 4 * hash(i + 9))) % (w * 1.5) + w * 1.5) % (w * 1.5) - w * 0.25
      const y = h * (0.08 + 0.3 * hash(i + 20))
      const s = 18 + 30 * hash(i + 40)
      ctx.beginPath()
      ctx.ellipse(x, y, s * 1.8, s * 0.55, 0, 0, Math.PI * 2)
      ctx.ellipse(x + s * 0.8, y - s * 0.3, s, s * 0.5, 0, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

/** Parallax ridge line: `base` is the ground y, `amp` the height in px, `offset` the scroll in px. */
function drawRidge(
  ctx: CanvasRenderingContext2D,
  w: number,
  base: number,
  amp: number,
  offset: number,
  scale: number,
  color: string,
) {
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(0, base)
  for (let x = 0; x <= w + 8; x += 8) {
    const n = smoothNoise((x + offset) / scale) * 0.7 + smoothNoise((x + offset) / (scale * 0.37)) * 0.3
    ctx.lineTo(x, base - amp * n)
  }
  ctx.lineTo(w, base)
  ctx.closePath()
  ctx.fill()
}

function drawSkyline(ctx: CanvasRenderingContext2D, w: number, base: number, unit: number, offset: number, color: string) {
  ctx.fillStyle = color
  const bw = unit * 1.6
  const first = Math.floor(offset / bw)
  for (let i = first; i * bw - offset < w + bw; i++) {
    const x = i * bw - offset
    const hh = unit * (1.2 + 3.4 * hash(i * 7.3))
    ctx.fillRect(x, base - hh, bw * (0.7 + 0.25 * hash(i)), hh)
  }
}

/** A faint outline of the box a game core tests for collisions ("show hitbox"). */
function strokeHitbox(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number) {
  ctx.save()
  ctx.strokeStyle = 'rgba(239,68,68,0.75)'
  ctx.lineWidth = 1
  ctx.setLineDash([3, 2])
  ctx.strokeRect(x, y, w, h)
  ctx.restore()
}

// ---------------------------------------------------------------------------------------------
// Flappy Drone

export interface FlappyView {
  width: number
  height: number
  drone: { x: number; y: number; w: number; h: number; vy: number; thrusting: boolean }
  pipes: { x: number; w: number; gap_top: number; gap_bottom: number }[]
  pipes_passed: number
  frame: number
  alive: boolean
}

export function drawFlappy(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  v: FlappyView,
  fx: FxState,
  theme: Theme,
  opts: { showScore?: boolean; hitbox?: boolean } = {},
) {
  const p = PALETTE[theme]
  const s = Math.min(h / v.height, w / 400)
  const oy = (h - v.height * s) / 2
  const sy = (alt: number) => oy + (v.height - alt) * s
  const sx = (x: number) => x * s
  const scroll = v.frame * (130 / 60) // px scrolled in world units (SCROLL * DT per frame)

  drawSky(ctx, w, h, theme, fx.t)
  const groundY = sy(0)
  drawRidge(ctx, w, groundY, 170 * s, scroll * s * 0.12, 260 * s, p.far)
  drawSkyline(ctx, w, groundY, 22 * s, scroll * s * 0.35, p.mid)

  // Pipes (pylons with caps and hazard stripes).
  for (const pipe of v.pipes) {
    const x0 = sx(pipe.x)
    const pw = pipe.w * s
    if (x0 > w || x0 + pw < 0) continue
    const grad = ctx.createLinearGradient(x0, 0, x0 + pw, 0)
    grad.addColorStop(0, p.pipe[2])
    grad.addColorStop(0.35, p.pipe[1])
    grad.addColorStop(1, p.pipe[0])
    ctx.fillStyle = grad
    const top = sy(pipe.gap_top)
    const bottom = sy(pipe.gap_bottom)
    ctx.fillRect(x0, Math.min(0, oy), pw, top - Math.min(0, oy))
    ctx.fillRect(x0, bottom, pw, Math.max(h, groundY) - bottom)
    const capH = 14 * s
    for (const [cy, dir] of [
      [top - capH, 1],
      [bottom, -1],
    ] as const) {
      ctx.fillStyle = p.cap
      roundRect(ctx, x0 - 4 * s, cy, pw + 8 * s, capH, 3 * s)
      ctx.fill()
      ctx.save()
      ctx.beginPath()
      ctx.rect(x0 - 4 * s, cy, pw + 8 * s, capH)
      ctx.clip()
      ctx.strokeStyle = p.stripe
      ctx.lineWidth = 4 * s
      for (let k = -2; k < 12; k++) {
        const bx = x0 + k * 10 * s
        ctx.beginPath()
        ctx.moveTo(bx, cy)
        ctx.lineTo(bx + 8 * s, cy + capH)
        ctx.stroke()
      }
      ctx.restore()
      // Obstruction light.
      const blink = Math.sin(fx.t * 4 + pipe.x * 0.05) > 0
      ctx.fillStyle = blink ? '#ef4444' : 'rgba(239,68,68,0.35)'
      ctx.beginPath()
      ctx.arc(x0 + pw / 2, dir > 0 ? cy - 3 * s : cy + capH + 3 * s, 2.5 * s, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // Ground band with moving dashes.
  ctx.fillStyle = p.ground
  ctx.fillRect(0, groundY, w, h - groundY)
  ctx.fillStyle = p.grass
  ctx.fillRect(0, groundY - 2 * s, w, 3 * s)
  ctx.fillStyle = theme === 'dark' ? 'rgba(148,163,184,0.25)' : 'rgba(255,255,255,0.4)'
  const dash = 24 * s
  for (let x = -((scroll * s) % (dash * 2)); x < w; x += dash * 2) ctx.fillRect(x, groundY + 6 * s, dash, 2 * s)

  // Drone.
  if (v.drone.thrusting) fx.thrustGlow = 1
  const tilt = clamp(-v.drone.vy / 900, -0.45, 0.45)
  const dx = sx(v.drone.x)
  const dy = sy(v.drone.y)
  drawDroneSide(ctx, dx, dy, v.drone.w * s * 1.15, tilt, fx.t, theme, fx.thrustGlow)
  // The box the core tests (the sprite's rotors reach past it).
  if (opts.hitbox) strokeHitbox(ctx, sx(v.drone.x - v.drone.w / 2), sy(v.drone.y + v.drone.h / 2), v.drone.w * s, v.drone.h * s)

  // Crash burst.
  const since = fx.t - fx.crashAt
  if (!v.alive && since < 1.2) {
    for (let i = 0; i < 18; i++) {
      const a = (i / 18) * Math.PI * 2
      const r = since * 120 * s * (0.6 + 0.4 * hash(i))
      ctx.fillStyle = `rgba(251,146,60,${1 - since / 1.2})`
      ctx.beginPath()
      ctx.arc(dx + Math.cos(a) * r, dy + Math.sin(a) * r, 3 * s, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  if (opts.showScore !== false) {
    ctx.fillStyle = p.text
    ctx.font = `700 ${Math.round(clamp(40 * s, 22, 56))}px ui-sans-serif, system-ui, sans-serif`
    ctx.textAlign = 'center'
    ctx.globalAlpha = 0.9
    ctx.fillText(String(v.pipes_passed), w / 2, oy + 54 * s)
    ctx.globalAlpha = 1
  }
}

// ---------------------------------------------------------------------------------------------
// Rescue Run

export interface RescueView {
  n: number
  cells: number[]
  drone: [number, number]
  prev: [number, number]
  base: [number, number]
  hikers: [number, number, boolean][]
  battery: number
  battery_max: number
  sensor_ok: boolean
  rescued: number
  steps: number
  last_action: string
  vetoed: boolean
  crashed: boolean
  completed: boolean
}

/** Wind cell code -> arrow direction on the canvas `[dx, dy]` (x = column, y = row, down). */
export const WIND_DIRS: Record<number, [number, number]> = {
  2: [0, -1], // toward north (up)
  3: [0, 1], // south
  4: [1, 0], // east
  5: [-1, 0], // west
}

/**
 * True when the last step entered the drone's cell against its wind arrow: the core charged 3
 * battery instead of 1 (`rescue::enter_cost`). The wind never moves the drone.
 */
export function rescueHeadwind(v: RescueView): boolean {
  const dr = v.drone[0] - v.prev[0]
  const dc = v.drone[1] - v.prev[1]
  if (!dr && !dc) return false
  const arrow = WIND_DIRS[v.cells[v.drone[0] * v.n + v.drone[1]]]
  return !!arrow && arrow[0] === -dc && arrow[1] === -dr
}

export function drawRescue(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  v: RescueView,
  fx: FxState,
  theme: Theme,
) {
  const p = PALETTE[theme]
  ctx.fillStyle = theme === 'dark' ? '#0b1220' : '#f0fdf4'
  ctx.fillRect(0, 0, w, h)
  const bar = 34
  const size = Math.min(w - 16, h - bar - 16)
  const cs = size / v.n
  const ox = (w - size) / 2
  const oy = bar + (h - bar - size) / 2
  const cx = (c: number) => ox + c * cs + cs / 2
  const cy = (r: number) => oy + r * cs + cs / 2

  // Battery bar and sensor pill.
  const frac = clamp(v.battery / v.battery_max, 0, 1)
  ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif'
  const label = `battery ${v.battery}/${v.battery_max}`
  const sensor = v.sensor_ok ? '● sensor ok' : '● sensor dropout'
  const bx = ox
  const bw = clamp(size - ctx.measureText(label).width - ctx.measureText(sensor).width - 28, 40, 240)
  ctx.fillStyle = theme === 'dark' ? '#1e293b' : '#e2e8f0'
  roundRect(ctx, bx, 8, bw, 18, 5)
  ctx.fill()
  ctx.fillStyle = frac > 0.5 ? '#22c55e' : frac > 0.25 ? '#f59e0b' : '#ef4444'
  roundRect(ctx, bx + 2, 10, Math.max(0, (bw - 4) * frac), 14, 4)
  ctx.fill()
  ctx.fillStyle = p.text
  ctx.textAlign = 'left'
  ctx.fillText(label, bx + bw + 8, 22)
  ctx.textAlign = 'right'
  ctx.fillStyle = v.sensor_ok ? '#22c55e' : '#f59e0b'
  ctx.fillText(sensor, ox + size, 22)

  // Tiles.
  for (let r = 0; r < v.n; r++) {
    for (let c = 0; c < v.n; c++) {
      const code = v.cells[r * v.n + c]
      const x = ox + c * cs
      const y = oy + r * cs
      ctx.fillStyle = p.tile[(r + c) % 2]
      ctx.fillRect(x, y, cs, cs)
      if (code >= 2 && code <= 5) {
        ctx.fillStyle = p.wind
        ctx.fillRect(x, y, cs, cs)
        const [dx, dy] = WIND_DIRS[code]
        const phase = (fx.t * 0.9 + (r + c) * 0.13) % 1
        ctx.strokeStyle = p.windArrow
        ctx.lineWidth = Math.max(1.2, cs * 0.06)
        ctx.lineCap = 'round'
        for (const k of [0, 0.5]) {
          const ph = ((phase + k) % 1) - 0.5
          const mx = x + cs / 2 + dx * ph * cs * 0.7
          const my = y + cs / 2 + dy * ph * cs * 0.7
          const a = cs * 0.13
          ctx.globalAlpha = 1 - Math.abs(ph) * 1.6
          ctx.beginPath()
          ctx.moveTo(mx - dx * a - dy * a, my - dy * a - dx * a)
          ctx.lineTo(mx, my)
          ctx.lineTo(mx - dx * a + dy * a, my - dy * a + dx * a)
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }
    }
  }
  ctx.strokeStyle = p.grid
  ctx.lineWidth = 1
  for (let i = 0; i <= v.n; i++) {
    ctx.beginPath()
    ctx.moveTo(ox + i * cs, oy)
    ctx.lineTo(ox + i * cs, oy + size)
    ctx.moveTo(ox, oy + i * cs)
    ctx.lineTo(ox + size, oy + i * cs)
    ctx.stroke()
  }

  // Rocks: jittered polygons.
  for (let r = 0; r < v.n; r++) {
    for (let c = 0; c < v.n; c++) {
      if (v.cells[r * v.n + c] !== 1) continue
      const x = cx(c)
      const y = cy(r)
      const g = ctx.createRadialGradient(x - cs * 0.15, y - cs * 0.15, cs * 0.05, x, y, cs * 0.5)
      g.addColorStop(0, p.rock[0])
      g.addColorStop(1, p.rock[1])
      ctx.fillStyle = g
      ctx.beginPath()
      const k = 7
      for (let i = 0; i < k; i++) {
        const a = (i / k) * Math.PI * 2
        const rr = cs * (0.32 + 0.12 * hash(r * 31 + c * 7 + i))
        const px = x + Math.cos(a) * rr
        const py = y + Math.sin(a) * rr
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.closePath()
      ctx.fill()
    }
  }

  // Base pad.
  const [br, bc] = v.base
  ctx.fillStyle = theme === 'dark' ? '#0c4a6e' : '#bae6fd'
  ctx.beginPath()
  ctx.arc(cx(bc), cy(br), cs * 0.42, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = '#0ea5e9'
  ctx.lineWidth = Math.max(1.5, cs * 0.06)
  ctx.stroke()
  ctx.fillStyle = theme === 'dark' ? '#e0f2fe' : '#0c4a6e'
  ctx.font = `800 ${Math.round(cs * 0.45)}px ui-sans-serif, system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('H', cx(bc), cy(br) + 1)
  ctx.textBaseline = 'alphabetic'

  // Hikers.
  v.hikers.forEach(([r, c, rescued], i) => {
    const x = cx(c)
    const y = cy(r)
    if (rescued) {
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = Math.max(2, cs * 0.08)
      ctx.beginPath()
      ctx.moveTo(x - cs * 0.18, y)
      ctx.lineTo(x - cs * 0.04, y + cs * 0.14)
      ctx.lineTo(x + cs * 0.2, y - cs * 0.14)
      ctx.stroke()
      return
    }
    const hidden = !v.sensor_ok
    ctx.globalAlpha = hidden ? 0.35 : 1
    // Pulse ring.
    const ring = (fx.t * 0.8 + i * 0.3) % 1
    ctx.strokeStyle = `rgba(249,115,22,${0.6 * (1 - ring)})`
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(x, y, cs * (0.2 + 0.3 * ring), 0, Math.PI * 2)
    ctx.stroke()
    ctx.strokeStyle = '#f97316'
    ctx.fillStyle = '#f97316'
    ctx.lineWidth = Math.max(1.5, cs * 0.07)
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.arc(x, y - cs * 0.2, cs * 0.08, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.moveTo(x, y - cs * 0.1)
    ctx.lineTo(x, y + cs * 0.1)
    ctx.moveTo(x, y + cs * 0.1)
    ctx.lineTo(x - cs * 0.1, y + cs * 0.25)
    ctx.moveTo(x, y + cs * 0.1)
    ctx.lineTo(x + cs * 0.1, y + cs * 0.25)
    const wave = Math.sin(fx.t * 6 + i) * 0.5
    ctx.moveTo(x, y - cs * 0.04)
    ctx.lineTo(x + Math.cos(-1.1 + wave) * cs * 0.2, y - cs * 0.04 + Math.sin(-1.1 + wave) * cs * 0.2)
    ctx.moveTo(x, y - cs * 0.04)
    ctx.lineTo(x - cs * 0.14, y + cs * 0.06)
    ctx.stroke()
    ctx.globalAlpha = 1
  })

  // Drone, tweened from the previous cell.
  const e = 1 - Math.pow(1 - clamp(fx.tween, 0, 1), 3)
  const dr = v.prev[0] + (v.drone[0] - v.prev[0]) * e
  const dc = v.prev[1] + (v.drone[1] - v.prev[1]) * e
  const x = ox + dc * cs + cs / 2
  const y = oy + dr * cs + cs / 2
  ctx.fillStyle = 'rgba(0,0,0,0.18)'
  ctx.beginPath()
  ctx.ellipse(x + cs * 0.08, y + cs * 0.14, cs * 0.36, cs * 0.2, 0, 0, Math.PI * 2)
  ctx.fill()
  drawDroneTop(ctx, x, y - cs * 0.04 * Math.sin(fx.t * 3), cs * 0.9, fx.t, theme)

  // Cues for what the arrows and the reflex did on the last step.
  const cue = v.vetoed
    ? `reflex: ${v.last_action.replace('_', ' ')}`
    : rescueHeadwind(v)
      ? 'headwind: -3 battery'
      : ''
  if (cue && !v.crashed && !v.completed) {
    ctx.font = '700 11px ui-sans-serif, system-ui, sans-serif'
    ctx.textAlign = 'center'
    const tw = ctx.measureText(cue).width + 10
    const ly = Math.max(oy + 10, y - cs * 0.62)
    ctx.fillStyle = v.vetoed ? 'rgba(220,38,38,0.92)' : 'rgba(2,132,199,0.92)'
    roundRect(ctx, x - tw / 2, ly - 9, tw, 16, 5)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.fillText(cue, x, ly + 3)
    ctx.textAlign = 'left'
  }

  if (v.crashed || v.completed) {
    ctx.fillStyle = v.completed ? 'rgba(34,197,94,0.92)' : 'rgba(239,68,68,0.92)'
    roundRect(ctx, w / 2 - 110, oy + size / 2 - 24, 220, 48, 12)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.font = '700 18px ui-sans-serif, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(v.completed ? 'Mission complete' : 'Drone lost', w / 2, oy + size / 2 + 6)
  }
}

// ---------------------------------------------------------------------------------------------
// Obstacle Dodge
//
// World axes as in `sky_games::dodge`: x along the course, y to the left, z up, in meters. The
// main view is a chase camera in perspective, behind and a little above the drone, looking
// forward along x; the strip below it is a top view (a map) of the corridor. Everything solid is
// drawn opaque, as the core tests it: an obstacle's body fills the corridor's width from the
// ground to its top, except its gap lane; a deck fills the width from its underside up to the
// airspace ceiling, along its span (overhang included).

export interface DodgeView {
  drone: { x: number; y: number; z: number; v: number; hl: number; hw: number; hh: number }
  obstacles: {
    x0: number
    x1: number
    top: number
    deck: [number, number, number] | null
    gap: [number, number] | null
    impassable: boolean
    passed: boolean
  }[]
  finish_x: number
  y_max: number
  z_max: number
  frame: number
  passed: number
  action: string
  vetoed: boolean
  vetoed_request: string | null
  crashed: boolean
  finished: boolean
  aborted: boolean
}

export type DodgeObstacle = DodgeView['obstacles'][number]

/** Seconds before a Dodge episode times out (`dodge::MAX_FRAMES` = 7,200 frames at 60 Hz). */
export const DODGE_MAX_S = 120

/** Frames of flight the trail keeps: 4 s at 60 Hz. */
export const DODGE_TRAIL_FRAMES = 240

/** One sampled position of the dodge drone, with the maneuver it was holding there. */
export interface DodgeTrailPoint {
  f: number
  x: number
  y: number
  z: number
  a: string
}

/**
 * The flight trail after the snapshot `v`: its position is added once per frame number, so
 * drawing the same frame again (paused) adds nothing; points older than `keep` frames are
 * dropped; a frame number or an x that goes back (a new episode) starts a new trail. Returns
 * `trail` itself when nothing changes.
 */
export function nextDodgeTrail(trail: DodgeTrailPoint[], v: DodgeView, keep = DODGE_TRAIL_FRAMES): DodgeTrailPoint[] {
  const d = v.drone
  const last = trail[trail.length - 1]
  if (last && last.f === v.frame && last.x === d.x) return trail
  const fresh = !last || v.frame <= last.f || d.x < last.x
  const kept = fresh ? [] : trail.filter((p) => v.frame - p.f <= keep)
  kept.push({ f: v.frame, x: d.x, y: d.y, z: d.z, a: v.action })
  return kept
}

/** The obstacle the current decision is about: the first one not passed (the core's `next`). */
export function dodgeNextObstacle(v: DodgeView): DodgeObstacle | undefined {
  return v.obstacles.find((o) => !o.passed)
}

/**
 * The held maneuver, as the chip next to the drone shows it. `continue` holds the altitude and
 * lateral offset reached (`tz = z`, `ty = y` in the core), hence "level"; `around` heads for the
 * gap of the next obstacle, on its side.
 */
export function dodgeManeuverLabel(v: DodgeView): string {
  if (v.action === 'continue') return 'continue · level'
  if (v.action === 'around') {
    const gap = dodgeNextObstacle(v)?.gap
    return gap ? `around · ${gap[0] > 0 ? 'left' : 'right'} gap` : 'around · no gap ahead'
  }
  return v.action
}

/** True when the drone's x-span overlaps `[x0, x1]`, with the core's strict inequalities. */
export function dodgeOverlapsX(d: DodgeView['drone'], x0: number, x1: number): boolean {
  return d.x + d.hl > x0 && d.x - d.hl < x1
}

/**
 * True when the drone's underside is at or above the top of `o`'s body (`z − hh >= top`): at
 * this altitude the body is not in its way (the core's body test needs `z − hh < top`).
 */
export function dodgeAboveBody(o: DodgeObstacle, d: DodgeView['drone']): boolean {
  return d.z - d.hh >= o.top
}

/**
 * True when the drone is laterally inside the gap lane of `o`, as the core tests it (its whole
 * width inside the gap): the body is then beside it, not in its way.
 */
export function dodgeInGapLane(o: DodgeObstacle, d: DodgeView['drone']): boolean {
  return !!o.gap && d.y - d.hw >= o.gap[0] && d.y + d.hw <= o.gap[1]
}

/**
 * A clearance to show while the drone overlaps an obstacle along x: `m` meters of free space
 * between its box and the obstacle's top (`above`), the deck's underside (`under`), or the side
 * of the body beside its gap lane (`beside`; `edge` is the y of that side).
 */
export interface DodgeClearance {
  kind: 'above' | 'under' | 'beside'
  m: number
  o: DodgeObstacle
  edge?: number
}

/**
 * The clearances of the obstacles the drone's x-span overlaps (the deck overhang included): over
 * a body (in its gap lane or not), under a deck it squeezes under (within
 * [`DODGE_DECK_NEAR_M`]), and, in a gap lane below the body's top, to the body's side.
 */
export function dodgeClearances(v: DodgeView): DodgeClearance[] {
  const d = v.drone
  if (v.crashed) return []
  const out: DodgeClearance[] = []
  for (const o of v.obstacles) {
    if (dodgeOverlapsX(d, o.x0, o.x1)) {
      if (dodgeAboveBody(o, d)) out.push({ kind: 'above', m: d.z - d.hh - o.top, o })
      else if (dodgeInGapLane(o, d)) {
        const [g0, g1] = o.gap!
        if (g0 > -v.y_max) out.push({ kind: 'beside', m: d.y - d.hw - g0, o, edge: g0 })
        if (g1 < v.y_max) out.push({ kind: 'beside', m: g1 - (d.y + d.hw), o, edge: g1 })
      }
    }
    if (o.deck && dodgeOverlapsX(d, o.deck[0], o.deck[1]) && d.z + d.hh <= o.deck[2] && o.deck[2] - (d.z + d.hh) <= DODGE_DECK_NEAR_M) {
      out.push({ kind: 'under', m: o.deck[2] - (d.z + d.hh), o })
    }
  }
  return out
}

/** A deck farther above the drone than this (m) is not one it squeezes under: no read-out. */
export const DODGE_DECK_NEAR_M = 3

/** Meters with one decimal, rounded down so that a clearance is never overstated. */
export function dodgeMeters(m: number): string {
  return (Math.floor(m * 10 + 1e-6) / 10).toFixed(1)
}

/** Pitch of the drone model, in radians: nose up when braking, slightly down at speed. */
export function dodgeTilt(v: DodgeView): number {
  return v.action === 'brake' || v.action === 'abort' ? -DODGE_MAX_TILT : clamp(v.drone.v / 110, 0, DODGE_MAX_TILT)
}

/** How an episode ended, for the banner (`done` is the session's `done`). */
export function dodgeOutcome(v: DodgeView, done: boolean): { text: string; ok: boolean } | null {
  if (v.crashed) return { text: 'Crashed', ok: false }
  if (v.finished) return { text: 'Finish line reached', ok: true }
  if (v.aborted) {
    const at = v.obstacles.find((o) => !o.passed)
    return at?.impassable
      ? { text: 'Aborted in front of an impassable obstacle', ok: true }
      : { text: 'Aborted, but a way past existed', ok: false }
  }
  if (done) return { text: `Timed out after ${DODGE_MAX_S} s`, ok: false }
  return null
}

/** Colors of the held maneuvers: the chip next to the drone and the trail it leaves. */
export const DODGE_MANEUVER_COLOR: Record<Theme, Record<string, string>> = {
  dark: { over: '#38bdf8', around: '#4ade80', brake: '#fbbf24', continue: '#cbd5e1', abort: '#f87171' },
  light: { over: '#0369a1', around: '#15803d', brake: '#b45309', continue: '#334155', abort: '#b91c1c' },
}

// --- The chase camera ------------------------------------------------------------------------

/** Horizontal field of view of the chase camera, in degrees. */
export const DODGE_FOV_DEG = 70
/** The camera sits this many meters behind the drone's center... */
export const DODGE_CAM_BACK = 7
/**
 * ...and aims to sit this many meters above it: none, level with it. Everything converges to
 * the point the camera looks at; from above the drone, a roof it clears by half a meter would
 * be drawn above its skids until it is a few meters away, and from below, a deck over it would
 * be drawn across its top. Level, both stay off the drone until much farther away.
 */
export const DODGE_CAM_LIFT = 0
/** Time constants of the camera's follow of the drone's y and of its z, in seconds: the drone's
 * sideways moves show more than its climbs, which must not take it out of the view. */
export const DODGE_CAM_TAU = 0.22
export const DODGE_CAM_TAU_Z = 0.08
/** Faster time constant used to bring the camera where it must be before an obstacle. */
export const DODGE_CAM_TAU_FAST = 0.06
/** Least distance kept between the camera and a surface it must stay on one side of (m). */
const CAM_EPS = 0.05
/** Meters of course drawn ahead of the drone. */
export const DODGE_FAR_M = 95
/** Share of the canvas height taken by the chase view; the map gets the rest. */
export const DODGE_MAIN_SHARE = 0.74
/** Height of the horizon in the chase view, as a share of its height from its top. */
export const DODGE_HORIZON = 0.4
/** Largest pitch and roll of the drone model (rad): within them it stays inside its hitbox. */
export const DODGE_MAX_TILT = 0.1

/** Heights of the chase view and of the map below it, in CSS pixels. */
export function dodgeSplit(h: number): { mainH: number; mapH: number } {
  const mainH = Math.round(h * DODGE_MAIN_SHARE)
  return { mainH, mapH: h - mainH }
}

/** The camera's lateral position and height, smoothed frame by frame (see [`nextDodgeCamPose`]). */
export interface DodgeCamPose {
  f: number
  x: number
  y: number
  z: number
}

/**
 * Moves the camera point `(y, z)` to where the drone is in clear view through free space: for
 * every obstacle whose span reaches the drone's x-span (or reaches it within `ahead` meters), the
 * camera is put on the drone's side of what the drone is clear of. Above a body, the camera is
 * above its top; in its gap lane, inside the lane; under a deck, under its underside. A straight
 * line from the camera to any point of the drone's box then stays in the free space the drone is
 * in, so nothing drawn can come between them. Then inside the corridor and the airspace.
 */
export function dodgeCamConstrain(v: DodgeView, y: number, z: number, ahead = 0): [number, number] {
  const d = v.drone
  const reach = d.x + d.hl + ahead
  for (const o of v.obstacles) {
    if (o.deck && reach > o.deck[0] && d.x - d.hl < o.deck[1] && d.z + d.hh <= o.deck[2]) z = Math.min(z, o.deck[2] - CAM_EPS)
    if (!(reach > o.x0 && d.x - d.hl < o.x1)) continue
    const above = dodgeAboveBody(o, d)
    const lane = dodgeInGapLane(o, d)
    if ((above && z >= o.top) || (lane && y >= o.gap![0] && y <= o.gap![1])) continue
    // Neither: the drone is in the body's way (it crashes, or the body is still ahead).
    if (!above && !lane) continue
    // The least move that puts the camera on the drone's side.
    const up = above ? o.top + CAM_EPS - z : Infinity
    const side = lane ? clamp(y, o.gap![0] + CAM_EPS, o.gap![1] - CAM_EPS) : y
    if (up <= (lane ? Math.abs(side - y) : Infinity)) z = o.top + CAM_EPS
    else y = side
  }
  return [clamp(y, -(v.y_max - CAM_EPS), v.y_max - CAM_EPS), clamp(z, 0.2, v.z_max - CAM_EPS)]
}

/**
 * The camera pose after the snapshot `v`. It follows the drone's y and z (plus the lift) with the
 * time constants [`DODGE_CAM_TAU`] and [`DODGE_CAM_TAU_Z`], so the drone's moves show on screen; an obstacle about to
 * be reached pulls it, faster, to where [`dodgeCamConstrain`] wants it; and it is always moved
 * there once the drone overlaps the obstacle. Updated once per frame number, like the trail; a
 * new episode (the frame or x goes back) starts it rigidly on the drone. Returns `prev` itself
 * when nothing changes (a paused redraw).
 */
export function nextDodgeCamPose(prev: DodgeCamPose | null, v: DodgeView): DodgeCamPose {
  const d = v.drone
  const ty = d.y
  const tz = d.z + DODGE_CAM_LIFT
  if (prev && prev.f === v.frame && prev.x === d.x) return prev
  if (!prev || v.frame <= prev.f || d.x < prev.x) {
    const [y, z] = dodgeCamConstrain(v, ty, tz)
    return { f: v.frame, x: d.x, y, z }
  }
  const n = Math.min(60, v.frame - prev.f)
  const ay = 1 - Math.exp(-n / (60 * DODGE_CAM_TAU))
  const az = 1 - Math.exp(-n / (60 * DODGE_CAM_TAU_Z))
  let y = prev.y + (ty - prev.y) * ay
  let z = prev.z + (tz - prev.z) * az
  const [sy, sz] = dodgeCamConstrain(v, y, z, Math.max(2, d.v * 0.5))
  const b = 1 - Math.exp(-n / (60 * DODGE_CAM_TAU_FAST))
  y += (sy - y) * b
  z += (sz - z) * b
  ;[y, z] = dodgeCamConstrain(v, y, z)
  return { f: v.frame, x: d.x, y, z }
}

/** The chase camera: position (m), focal length (px), principal point (px) and near plane (x). */
export interface DodgeCamera {
  x: number
  y: number
  z: number
  f: number
  u0: number
  v0: number
  /** Nothing at a smaller x is drawn: the plane through the back of the drone's box. */
  near: number
}

/** Focal length in pixels of a view `w` pixels wide with the field of view [`DODGE_FOV_DEG`]. */
export function dodgeFocal(w: number): number {
  return w / 2 / Math.tan((DODGE_FOV_DEG * Math.PI) / 360)
}

/** The camera for a chase view `w` by `mainH` pixels, at the pose `pose`. */
export function dodgeCamera(w: number, mainH: number, v: DodgeView, pose: { y: number; z: number }): DodgeCamera {
  return {
    x: v.drone.x - DODGE_CAM_BACK,
    y: pose.y,
    z: pose.z,
    f: dodgeFocal(w),
    u0: w / 2,
    v0: mainH * DODGE_HORIZON,
    near: v.drone.x - v.drone.hl,
  }
}

/**
 * The one perspective projection of the chase view: a world point to canvas pixels. y to the
 * left is screen left, z up is screen up; points converge to `(u0, v0)` (the horizon's middle)
 * as they recede. Only for points ahead of the camera (`x > c.x`).
 */
export function dodgeProject(c: DodgeCamera, x: number, y: number, z: number): [number, number] {
  const dx = x - c.x
  return [c.u0 - (c.f * (y - c.y)) / dx, c.v0 - (c.f * (z - c.z)) / dx]
}

// --- The scene as boxes ----------------------------------------------------------------------

export type DodgeBoxKind = 'body' | 'deck' | 'pillar' | 'fence' | 'gate'

/** An axis-aligned box of the scene (m). */
export interface DodgeBox {
  x0: number
  x1: number
  y0: number
  y1: number
  z0: number
  z1: number
  kind: DodgeBoxKind
  /** Index of the obstacle in the snapshot; -1 for the fences and the finish gate. */
  o: number
  /** True for what the core tests: a body (outside its gap lane) or a deck. */
  solid: boolean
  /** True when the near plane cut the box: its front face is a section at the near plane. */
  cut?: boolean
}

const PILLAR_M = 0.4
const FENCE_H = 0.6

/**
 * The scene's boxes near the drone, before clipping: each obstacle's body (one box on each side
 * of a gap lane, none in it), its deck from the underside up to the airspace ceiling, with two
 * pillars at each end standing outside the corridor; the low fences along the corridor's sides
 * (outside it); the finish gate's posts (outside) and banner (above the airspace).
 */
export function dodgeSceneBoxes(v: DodgeView, far = DODGE_FAR_M): DodgeBox[] {
  const d = v.drone
  const Y = v.y_max
  const Z = v.z_max
  const lo = d.x - d.hl - 1
  const hi = d.x + far
  const out: DodgeBox[] = []
  v.obstacles.forEach((o, i) => {
    const s0 = o.deck ? Math.min(o.x0, o.deck[0]) : o.x0
    const s1 = o.deck ? Math.max(o.x1, o.deck[1]) : o.x1
    if (s1 < lo || s0 > hi) return
    const body = (y0: number, y1: number) => {
      if (y1 > y0 && o.top > 0) out.push({ x0: o.x0, x1: o.x1, y0, y1, z0: 0, z1: o.top, kind: 'body', o: i, solid: true })
    }
    if (o.gap) {
      body(-Y, o.gap[0])
      body(o.gap[1], Y)
    } else body(-Y, Y)
    if (o.deck) {
      const [d0, d1, ceil] = o.deck
      out.push({ x0: d0, x1: d1, y0: -Y, y1: Y, z0: ceil, z1: Z, kind: 'deck', o: i, solid: true })
      for (const [a, b] of [
        [d0, d0 + PILLAR_M],
        [d1 - PILLAR_M, d1],
      ]) {
        out.push({ x0: a, x1: b, y0: Y + 0.3, y1: Y + 0.3 + PILLAR_M, z0: 0, z1: Z, kind: 'pillar', o: i, solid: false })
        out.push({ x0: a, x1: b, y0: -Y - 0.3 - PILLAR_M, y1: -Y - 0.3, z0: 0, z1: Z, kind: 'pillar', o: i, solid: false })
      }
    }
  })
  out.push({ x0: lo, x1: hi, y0: Y, y1: Y + 0.08, z0: 0, z1: FENCE_H, kind: 'fence', o: -1, solid: false })
  out.push({ x0: lo, x1: hi, y0: -Y - 0.08, y1: -Y, z0: 0, z1: FENCE_H, kind: 'fence', o: -1, solid: false })
  const fx = v.finish_x
  if (fx + 0.5 > lo && fx < hi) {
    out.push({ x0: fx, x1: fx + 0.5, y0: Y + 0.15, y1: Y + 0.85, z0: 0, z1: Z, kind: 'gate', o: -1, solid: false })
    out.push({ x0: fx, x1: fx + 0.5, y0: -Y - 0.85, y1: -Y - 0.15, z0: 0, z1: Z, kind: 'gate', o: -1, solid: false })
    out.push({ x0: fx, x1: fx + 0.5, y0: -Y - 0.85, y1: Y + 0.85, z0: Z, z1: Z + 1.2, kind: 'gate', o: -1, solid: false })
  }
  return out
}

/** The drone's hitbox, the box the core tests, as a scene box. */
export function dodgeDroneBox(v: DodgeView): DodgeBox {
  const d = v.drone
  return { x0: d.x - d.hl, x1: d.x + d.hl, y0: d.y - d.hw, y1: d.y + d.hw, z0: d.z - d.hh, z1: d.z + d.hh, kind: 'body', o: -1, solid: false }
}

/** `b` clipped at the near plane `near`: none when wholly behind it, cut when it crosses it. */
export function dodgeClipBox(b: DodgeBox, near: number): DodgeBox | null {
  if (b.x1 <= near) return null
  return b.x0 >= near ? b : { ...b, x0: near, cut: true }
}

type Point3 = { x: number; y: number; z: number }
type Extent3 = Pick<DodgeBox, 'x0' | 'x1' | 'y0' | 'y1' | 'z0' | 'z1'>

/**
 * True when `a` cannot hide any part of `b` from the camera at `c`: a plane across x, y or z
 * separates them with the camera on `b`'s side, so no line from the camera to `b` meets `a`.
 */
export function dodgeCannotHide(a: Extent3, b: Extent3, c: Point3): boolean {
  return (
    (b.x1 <= a.x0 && c.x <= a.x0) ||
    (a.x1 <= b.x0 && c.x >= a.x1) ||
    (b.y1 <= a.y0 && c.y <= a.y0) ||
    (a.y1 <= b.y0 && c.y >= a.y1) ||
    (b.z1 <= a.z0 && c.z <= a.z0) ||
    (a.z1 <= b.z0 && c.z >= a.z1)
  )
}

/**
 * Painter's order for `boxes` seen from `c`: a box comes before every box that could hide part
 * of it (see [`dodgeCannotHide`]), the farthest first among those free to go. Disjoint boxes
 * never hide each other both ways; a cycle, which this scene does not make, is broken by distance.
 */
export function dodgeDrawOrder<B extends Extent3>(boxes: B[], c: Point3): B[] {
  const n = boxes.length
  const after: number[][] = boxes.map(() => [])
  const indeg = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const jHidesI = !dodgeCannotHide(boxes[j], boxes[i], c)
      const iHidesJ = !dodgeCannotHide(boxes[i], boxes[j], c)
      if (jHidesI && !iHidesJ) {
        after[i].push(j)
        indeg[j]++
      } else if (iHidesJ && !jHidesI) {
        after[j].push(i)
        indeg[i]++
      }
    }
  }
  const dist = (b: B) => b.x0 - c.x
  const done = new Array(n).fill(false)
  const out: B[] = []
  for (let k = 0; k < n; k++) {
    let pick = -1
    let free = false
    for (let i = 0; i < n; i++) {
      if (done[i]) continue
      const ok = indeg[i] === 0
      if (pick < 0 || (ok && !free) || (ok === free && dist(boxes[i]) > dist(boxes[pick]))) {
        pick = i
        free = ok
      }
    }
    done[pick] = true
    out.push(boxes[pick])
    for (const j of after[pick]) indeg[j]--
  }
  return out
}

/** What the chase view draws, in order: the scene's boxes clipped at the near plane. */
export function dodgeDrawnBoxes(v: DodgeView, c: DodgeCamera): DodgeBox[] {
  const clipped = dodgeSceneBoxes(v)
    .map((b) => dodgeClipBox(b, c.near))
    .filter((b): b is DodgeBox => !!b)
  return dodgeDrawOrder(clipped, c)
}

export type DodgeFaceSide = 'front' | 'left' | 'right' | 'top' | 'bottom'

/** A face of a box as drawn: its side and its projected corners. */
export interface DodgeFace {
  side: DodgeFaceSide
  pts: [number, number][]
}

/**
 * The faces of `b` that face the camera, projected: the front (x0) always (the camera is behind
 * the near plane), a side, the top or the bottom when the camera is beyond that plane. The faces
 * of one box never overlap each other on screen.
 */
export function dodgeBoxFaces(b: Extent3, c: DodgeCamera): DodgeFace[] {
  const P = (x: number, y: number, z: number) => dodgeProject(c, x, y, z)
  const { x0, x1, y0, y1, z0, z1 } = b
  const out: DodgeFace[] = [{ side: 'front', pts: [P(x0, y0, z0), P(x0, y1, z0), P(x0, y1, z1), P(x0, y0, z1)] }]
  if (c.y > y1) out.push({ side: 'left', pts: [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)] })
  if (c.y < y0) out.push({ side: 'right', pts: [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)] })
  if (c.z > z1) out.push({ side: 'top', pts: [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)] })
  if (c.z < z0) out.push({ side: 'bottom', pts: [P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0)] })
  return out
}

/** Screen bounds of a box's projection (all eight corners; the box must be ahead of the camera). */
export function dodgeScreenBounds(c: DodgeCamera, b: Extent3): { x0: number; x1: number; y0: number; y1: number } {
  let [u0, u1, w0, w1] = [Infinity, -Infinity, Infinity, -Infinity]
  for (const x of [b.x0, b.x1]) {
    for (const y of [b.y0, b.y1]) {
      for (const z of [b.z0, b.z1]) {
        const [u, w] = dodgeProject(c, x, y, z)
        u0 = Math.min(u0, u)
        u1 = Math.max(u1, u)
        w0 = Math.min(w0, w)
        w1 = Math.max(w1, w)
      }
    }
  }
  return { x0: u0, x1: u1, y0: w0, y1: w1 }
}

// --- The drone model -------------------------------------------------------------------------

type V3 = [number, number, number]

/** Rotates a point of the drone model (m, around its center) by `roll` about x, then `pitch`
 * about y (positive pitch: nose down; positive roll: right side down). */
function tiltPoint([x, y, z]: V3, pitch: number, roll: number): V3 {
  const [cr, sr] = [Math.cos(roll), Math.sin(roll)]
  const y1 = y * cr - z * sr
  const z1 = y * sr + z * cr
  const [cp, sp] = [Math.cos(pitch), Math.sin(pitch)]
  return [x * cp + z1 * sp, y1, -x * sp + z1 * cp]
}

/** The drone model, in meters around the hitbox center: four rotor discs, the arms, the body
 * box and the skids, sized from the hitbox (rotor tip to rotor tip it spans 98% of the width). */
function droneParts(d: DodgeView['drone']) {
  const { hl, hw, hh } = d
  const rr = 0.34 * Math.min(hl, hw)
  const rotors: { c: V3; r: number }[] = []
  for (const sx of [1, -1]) for (const sy of [1, -1]) rotors.push({ c: [sx * (0.98 * hl - rr), sy * (0.98 * hw - rr), 0.27 * hh], r: rr })
  const body: V3[] = []
  for (const x of [-0.42 * hl, 0.42 * hl]) for (const y of [-0.3 * hw, 0.3 * hw]) for (const z of [-0.35 * hh, 0.25 * hh]) body.push([x, y, z])
  const skids: [V3, V3][] = []
  for (const sy of [1, -1]) {
    skids.push([[-0.36 * hl, sy * 0.3 * hw, -0.8 * hh], [0.36 * hl, sy * 0.3 * hw, -0.8 * hh]])
    skids.push([[-0.2 * hl, sy * 0.22 * hw, -0.35 * hh], [-0.24 * hl, sy * 0.3 * hw, -0.8 * hh]])
    skids.push([[0.2 * hl, sy * 0.22 * hw, -0.35 * hh], [0.24 * hl, sy * 0.3 * hw, -0.8 * hh]])
  }
  return { rotors, body, skids }
}

/** Points of the rotor disc around `c` (in the model's horizontal plane). */
function discPoints(c: V3, r: number, n = 14): V3[] {
  const out: V3[] = []
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2
    out.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]])
  }
  return out
}

/** Every vertex of the drone model, tilted, relative to the hitbox center (for tests). */
export function dodgeDroneModelPoints(d: DodgeView['drone'], pitch: number, roll: number): V3[] {
  const { rotors, body, skids } = droneParts(d)
  const pts: V3[] = [...body, ...skids.flat()]
  for (const r of rotors) pts.push(...discPoints(r.c, r.r, 32))
  return pts.map((p) => tiltPoint(p, pitch, roll))
}

/** Roll of the drone model from its lateral speed over the last frames of the trail. */
export function dodgeRoll(trail: DodgeTrailPoint[]): number {
  const n = trail.length
  if (n < 2) return 0
  const a = trail[Math.max(0, n - 4)]
  const b = trail[n - 1]
  if (b.f <= a.f) return 0
  const vy = ((b.y - a.y) * 60) / (b.f - a.f)
  // Moving left (+y) banks left: the left side goes down.
  return clamp(-vy * 0.03, -DODGE_MAX_TILT, DODGE_MAX_TILT) || 0
}

function convexHull(pts: [number, number][]): [number, number][] {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (p.length < 3) return p
  const cross = (o: [number, number], a: [number, number], b: [number, number]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: [number, number][] = []
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop()
    lower.push(q)
  }
  const upper: [number, number][] = []
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop()
    upper.push(q)
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1))
}

function polyPath(ctx: CanvasRenderingContext2D, pts: [number, number][]) {
  ctx.beginPath()
  ctx.moveTo(pts[0][0], pts[0][1])
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
  ctx.closePath()
}

/** The drone seen by the chase camera: the model above, tilted, projected with the scene's own
 * projection, so it spans its hitbox at its distance. */
function drawDrone3D(ctx: CanvasRenderingContext2D, c: DodgeCamera, d: DodgeView['drone'], pitch: number, roll: number, t: number, theme: Theme) {
  const p = PALETTE[theme]
  const P = (q: V3): [number, number] => {
    const [x, y, z] = tiltPoint(q, pitch, roll)
    return dodgeProject(c, d.x + x, d.y + y, d.z + z)
  }
  const { rotors, body, skids } = droneParts(d)
  const lw = Math.max(1, (c.f * 0.035) / DODGE_CAM_BACK)
  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const disc = theme === 'dark' ? 'rgba(148,163,184,0.32)' : 'rgba(51,65,85,0.26)'
  const rotor = (r: { c: V3; r: number }, i: number) => {
    ctx.fillStyle = disc
    polyPath(ctx, discPoints(r.c, r.r).map(P))
    ctx.fill()
    // A blade: its two tips go round the disc.
    const a = t * 50 + i * 1.3
    const tip = (s: number): V3 => [r.c[0] + s * r.r * Math.cos(a), r.c[1] + s * r.r * Math.sin(a), r.c[2]]
    ctx.strokeStyle = p.drone
    ctx.lineWidth = lw
    ctx.beginPath()
    ctx.moveTo(...P(tip(-0.95)))
    ctx.lineTo(...P(tip(0.95)))
    ctx.stroke()
  }
  const arm = (r: { c: V3 }) => {
    ctx.strokeStyle = p.droneDark
    ctx.lineWidth = lw * 1.8
    ctx.beginPath()
    ctx.moveTo(...P([0, 0, 0.1 * d.hh]))
    ctx.lineTo(...P(r.c))
    ctx.stroke()
  }
  // Far parts first: the front rotors and arms, the skids, the body, then the rear ones.
  const front = rotors.filter((r) => r.c[0] > 0)
  const rear = rotors.filter((r) => r.c[0] < 0)
  front.forEach((r, i) => {
    arm(r)
    rotor(r, i)
  })
  ctx.strokeStyle = p.droneDark
  ctx.lineWidth = lw
  for (const [a, b] of skids) {
    ctx.beginPath()
    ctx.moveTo(...P(a))
    ctx.lineTo(...P(b))
    ctx.stroke()
  }
  rear.forEach(arm)
  const hull = convexHull(body.map(P))
  ctx.fillStyle = p.droneDark
  polyPath(ctx, hull)
  ctx.fill()
  // The top and the back of the body, lighter.
  const top = [body[1], body[3], body[7], body[5]].map(P)
  ctx.fillStyle = p.drone
  polyPath(ctx, top)
  ctx.fill()
  // Status LEDs at the back: red, blinking.
  if (Math.sin(t * 9) > 0) {
    ctx.fillStyle = '#ef4444'
    for (const sy of [-0.2, 0.2]) {
      const [u, w] = P([-0.42 * d.hl, sy * d.hw, 0])
      ctx.beginPath()
      ctx.arc(u, w, Math.max(1.2, lw * 1.1), 0, Math.PI * 2)
      ctx.fill()
    }
  }
  rear.forEach((r, i) => rotor(r, i + 2))
  ctx.restore()
}

// --- Drawing ---------------------------------------------------------------------------------

type RGB = [number, number, number]

function rgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** `a` mixed toward `b` by `t` (0..1), as an opaque CSS color. */
function mixColor(a: RGB, b: RGB, t: number): string {
  const m = (i: number) => Math.round(a[i] + (b[i] - a[i]) * t)
  return `rgb(${m(0)},${m(1)},${m(2)})`
}

interface DodgeFacePaint {
  front: string
  side: string
  top: string
  bottom: string
  cut: string
}

const DODGE_COLORS = {
  dark: {
    fog: '#1e2a5a',
    grass: '#0d1a1f',
    lane: '#18223a',
    tile: '#1c2742',
    tileLine: '#2a3858',
    gapFloor: '#166534',
    gapEdge: '#22c55e',
    body: { front: '#4a5a75', side: '#38455e', top: '#6a7b98', bottom: '#2a3447', cut: '#323e55' },
    wall: { front: '#a11d1d', side: '#7f1717', top: '#c53030', bottom: '#5f1010', cut: '#7a1515' },
    hatch: '#fca5a5',
    deck: { front: '#5b6472', side: '#4a525e', top: '#737c8a', bottom: '#3a4049', cut: '#454c57' },
    pillar: { front: '#4f5764', side: '#3e4550', top: '#636b78', bottom: '#30353e', cut: '#3e4550' },
    fence: { front: '#64748b', side: '#526074', top: '#f59e0b', bottom: '#475569', cut: '#64748b' },
    gate: { front: '#e2e8f0', side: '#cbd5e1', top: '#f8fafc', bottom: '#94a3b8', cut: '#e2e8f0' },
    windowLit: '#f2cc66',
    windowDark: '#28324a',
    stripe: '#f59e0b',
    edge: '#0b1222',
  },
  light: {
    fog: '#e0f2fe',
    grass: '#8fbf5f',
    lane: '#cfd8e3',
    tile: '#dbe3ec',
    tileLine: '#9aa8ba',
    gapFloor: '#4ade80',
    gapEdge: '#15803d',
    body: { front: '#8597b2', side: '#6b7d99', top: '#b4c1d4', bottom: '#56657d', cut: '#61718b' },
    wall: { front: '#c02626', side: '#991b1b', top: '#dc2626', bottom: '#7f1d1d', cut: '#8f1c1c' },
    hatch: '#fecaca',
    deck: { front: '#6b7280', side: '#5b616b', top: '#9ca3af', bottom: '#4b5058', cut: '#565c66' },
    pillar: { front: '#646b77', side: '#535964', top: '#8a919c', bottom: '#454a53', cut: '#535964' },
    fence: { front: '#475569', side: '#3b4757', top: '#d97706', bottom: '#334155', cut: '#475569' },
    gate: { front: '#f8fafc', side: '#e2e8f0', top: '#ffffff', bottom: '#cbd5e1', cut: '#f8fafc' },
    windowLit: '#f1f7ff',
    windowDark: '#4f5f78',
    stripe: '#f59e0b',
    edge: '#334155',
  },
}

/** Share of the fog color mixed into something `dx` meters from the camera. */
function fogAt(dx: number): number {
  return clamp((dx - 32) / 75, 0, 0.82)
}

const LABEL_FONT = '600 11px ui-sans-serif, system-ui, sans-serif'
const REFLEX_FONT = '700 11px ui-sans-serif, system-ui, sans-serif'

interface PillStyle {
  font: string
  bg: string
  fg: string
  border?: string
  dot?: string
}

function pillWidth(ctx: CanvasRenderingContext2D, text: string, s: PillStyle): number {
  ctx.font = s.font
  return Math.ceil(ctx.measureText(text).width) + 12 + (s.dot ? 10 : 0)
}

/** A rounded label, `wd` px wide and `ht` tall, from its top-left corner. */
function drawPill(ctx: CanvasRenderingContext2D, text: string, left: number, top: number, wd: number, ht: number, s: PillStyle) {
  ctx.fillStyle = s.bg
  roundRect(ctx, left, top, wd, ht, ht / 2)
  ctx.fill()
  if (s.border) {
    ctx.strokeStyle = s.border
    ctx.lineWidth = 1
    ctx.stroke()
  }
  let tx = left + 6
  if (s.dot) {
    ctx.fillStyle = s.dot
    ctx.beginPath()
    ctx.arc(left + 9, top + ht / 2, 3, 0, Math.PI * 2)
    ctx.fill()
    tx += 10
  }
  ctx.fillStyle = s.fg
  ctx.font = s.font
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, tx, top + ht / 2 + 0.5)
  ctx.textBaseline = 'alphabetic'
}

/** The trail as runs of a few points fading with age (in frames), each in the color of the
 * maneuver held along it. Butt caps, so that the runs meet without overlapping. */
function drawTrail(
  ctx: CanvasRenderingContext2D,
  trail: DodgeTrailPoint[],
  frame: number,
  at: (p: DodgeTrailPoint) => [number, number],
  theme: Theme,
  width: number,
) {
  ctx.save()
  ctx.lineWidth = width
  ctx.lineCap = 'butt'
  ctx.lineJoin = 'round'
  let i = 1
  while (i < trail.length) {
    const a = trail[i].a
    let j = i
    while (j + 1 < trail.length && j - i < 8 && trail[j + 1].a === a) j++
    ctx.globalAlpha = 0.85 * Math.max(0, 1 - (frame - trail[j].f) / DODGE_TRAIL_FRAMES)
    ctx.strokeStyle = DODGE_MANEUVER_COLOR[theme][a] ?? DODGE_MANEUVER_COLOR[theme].continue
    ctx.beginPath()
    ctx.moveTo(...at(trail[i - 1]))
    for (let q = i; q <= j; q++) ctx.lineTo(...at(trail[q]))
    ctx.stroke()
    i = j + 1
  }
  ctx.restore()
}

/** A dimension mark from `a` to `b` (canvas px), with end ticks across it. */
function dimensionMark(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number], color: string) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1
  const [nx, ny] = [(-(b[1] - a[1]) / len) * 3.5, ((b[0] - a[0]) / len) * 3.5]
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(a[0], a[1])
  ctx.lineTo(b[0], b[1])
  ctx.moveTo(a[0] - nx, a[1] - ny)
  ctx.lineTo(a[0] + nx, a[1] + ny)
  ctx.moveTo(b[0] - nx, b[1] - ny)
  ctx.lineTo(b[0] + nx, b[1] + ny)
  ctx.stroke()
}

/** Paints one face of a box: its fill (with the distance fog), the windows of a facade, the red
 * hatch of an impassable obstacle, the hazard band under a deck's front, and a thin edge. */
function paintFace(
  ctx: CanvasRenderingContext2D,
  c: DodgeCamera,
  b: DodgeBox,
  face: DodgeFace,
  v: DodgeView,
  theme: Theme,
) {
  const C = DODGE_COLORS[theme]
  const o = b.o >= 0 ? v.obstacles[b.o] : null
  const paint: DodgeFacePaint =
    b.kind === 'body' ? (o?.impassable ? C.wall : C.body) : b.kind === 'deck' ? (o?.impassable ? C.wall : C.deck) : C[b.kind]
  const base = face.side === 'front' && b.cut ? paint.cut : paint[face.side === 'left' || face.side === 'right' ? 'side' : face.side]
  const fog = rgb(C.fog)
  const fogT = fogAt(b.x0 - c.x)
  ctx.fillStyle = mixColor(rgb(base), fog, fogT)
  polyPath(ctx, face.pts)
  ctx.fill()
  const P = (x: number, y: number, z: number) => dodgeProject(c, x, y, z)
  const quad = (pts: [number, number][], color: string) => {
    ctx.fillStyle = color
    polyPath(ctx, pts)
    ctx.fill()
  }
  const near = b.x0 - c.x < 70
  // Windows on the facades of an ordinary body (not on a section cut by the near plane).
  if (b.kind === 'body' && o && !o.impassable && near && face.side !== 'top' && face.side !== 'bottom' && !(face.side === 'front' && b.cut)) {
    const lit = mixColor(rgb(C.windowLit), fog, fogT)
    const dark = mixColor(rgb(C.windowDark), fog, fogT)
    for (let z = 0.9; z + 0.7 <= b.z1 - 0.4; z += 1.5) {
      if (face.side === 'front') {
        for (let y = b.y0 + 0.5; y + 0.6 <= b.y1 - 0.4; y += 1.3) {
          const on = hash(b.o * 31.7 + y * 3.1 + z * 7.7) > 0.45
          quad([P(b.x0, y, z), P(b.x0, y + 0.6, z), P(b.x0, y + 0.6, z + 0.7), P(b.x0, y, z + 0.7)], on ? lit : dark)
        }
      } else {
        const yf = face.side === 'left' ? b.y1 : b.y0
        for (let x = o.x0 + 0.4; x + 0.6 <= o.x1 - 0.3; x += 1.2) {
          if (x < b.x0) continue
          const on = hash(b.o * 17.3 + x * 5.3 + z * 2.9 + yf) > 0.45
          quad([P(x, yf, z), P(x + 0.6, yf, z), P(x + 0.6, yf, z + 0.7), P(x, yf, z + 0.7)], on ? lit : dark)
        }
      }
    }
  }
  // The impassable obstacle's red hatch, in world space on the face's plane.
  if (o?.impassable && face.side !== 'bottom') {
    ctx.save()
    polyPath(ctx, face.pts)
    ctx.clip()
    ctx.strokeStyle = mixColor(rgb(C.hatch), fog, fogT)
    ctx.lineWidth = Math.max(1, Math.min(2.5, (c.f * 0.12) / (b.x0 - c.x)))
    ctx.beginPath()
    if (face.side === 'front') {
      for (let s = b.y0 - (b.z1 - b.z0); s < b.y1; s += 1.2) {
        ctx.moveTo(...P(b.x0, s, b.z0))
        ctx.lineTo(...P(b.x0, s + (b.z1 - b.z0), b.z1))
      }
    } else if (face.side === 'top') {
      for (let s = b.y0 - (b.x1 - b.x0); s < b.y1; s += 1.2) {
        ctx.moveTo(...P(b.x0, s, b.z1))
        ctx.lineTo(...P(b.x1, s + (b.x1 - b.x0), b.z1))
      }
    } else {
      const yf = face.side === 'left' ? b.y1 : b.y0
      for (let s = b.x0 - (b.z1 - b.z0); s < b.x1; s += 1.2) {
        ctx.moveTo(...P(Math.max(s, b.x0), yf, b.z0 + Math.max(0, b.x0 - s)))
        ctx.lineTo(...P(s + (b.z1 - b.z0), yf, b.z1))
      }
    }
    ctx.stroke()
    ctx.restore()
  }
  // A deck's hazard band along its front underside.
  if (b.kind === 'deck' && face.side === 'front' && !b.cut && !o?.impassable) {
    const z = b.z0
    ctx.fillStyle = mixColor(rgb(C.stripe), fog, fogT)
    polyPath(ctx, [P(b.x0, b.y0, z), P(b.x0, b.y1, z), P(b.x0, b.y1, z + 0.35), P(b.x0, b.y0, z + 0.35)])
    ctx.fill()
  }
  // The gate's chequer.
  if (b.kind === 'gate' && face.side !== 'bottom') {
    ctx.fillStyle = mixColor(rgb('#0f172a'), fog, fogT)
    const sq = 0.5
    if (face.side === 'front') {
      for (let z = b.z0, i = 0; z < b.z1 - 1e-6; z += sq, i++) {
        for (let y = b.y0, j = 0; y < b.y1 - 1e-6; y += sq, j++) {
          if ((i + j) % 2) continue
          polyPath(ctx, [P(b.x0, y, z), P(b.x0, Math.min(y + sq, b.y1), z), P(b.x0, Math.min(y + sq, b.y1), Math.min(z + sq, b.z1)), P(b.x0, y, Math.min(z + sq, b.z1))])
          ctx.fill()
        }
      }
    } else if (face.side === 'left' || face.side === 'right') {
      const yf = face.side === 'left' ? b.y1 : b.y0
      for (let z = b.z0, i = 0; z < b.z1 - 1e-6; z += sq, i++) {
        if (i % 2) continue
        polyPath(ctx, [P(b.x0, yf, z), P(b.x1, yf, z), P(b.x1, yf, Math.min(z + sq, b.z1)), P(b.x0, yf, Math.min(z + sq, b.z1))])
        ctx.fill()
      }
    }
  }
  // Fence posts every 2 m, so the sides scroll by.
  if (b.kind === 'fence' && face.side !== 'top' && face.side !== 'bottom' && face.side !== 'front') {
    const yf = face.side === 'left' ? b.y1 : b.y0
    ctx.strokeStyle = mixColor(rgb(C.edge), fog, 0)
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let x = Math.ceil(b.x0 / 2) * 2; x < Math.min(b.x1, c.x + 70); x += 2) {
      ctx.moveTo(...P(x, yf, 0))
      ctx.lineTo(...P(x, yf, b.z1))
    }
    ctx.stroke()
  }
  ctx.strokeStyle = mixColor(rgb(C.edge), fog, fogT)
  ctx.lineWidth = 1
  polyPath(ctx, face.pts)
  ctx.stroke()
}

/** The ground: grass, the corridor's lane in tiles 2 m long that scroll by, the gap lanes in
 * green under their obstacles, and the chequered finish line. */
function drawGround(ctx: CanvasRenderingContext2D, c: DodgeCamera, v: DodgeView, w: number, mainH: number, theme: Theme) {
  const C = DODGE_COLORS[theme]
  const fog = rgb(C.fog)
  const Y = v.y_max
  const x0 = c.x + 0.6
  const x1 = v.drone.x + DODGE_FAR_M
  const P = (x: number, y: number) => dodgeProject(c, x, y, 0)
  const g = ctx.createLinearGradient(0, c.v0, 0, mainH)
  g.addColorStop(0, mixColor(rgb(C.grass), fog, 0.82))
  g.addColorStop(0.25, mixColor(rgb(C.grass), fog, 0.3))
  g.addColorStop(1, C.grass)
  ctx.fillStyle = g
  ctx.fillRect(0, c.v0, w, mainH - c.v0)
  // Tiles, from far to near, each mixed toward the fog with its distance.
  const step = 2
  const tileFar = Math.min(x1, c.x + 90)
  const first = Math.floor(x0 / step) * step
  for (let x = Math.floor(tileFar / step) * step; x >= first; x -= step) {
    const a = Math.max(x, x0)
    const b = Math.min(x + step, tileFar)
    if (b <= a) continue
    const base = (Math.round(x / step) & 1) === 0 ? C.lane : C.tile
    ctx.fillStyle = mixColor(rgb(base), fog, fogAt(a - c.x))
    polyPath(ctx, [P(a, -Y), P(b, -Y), P(b, Y), P(a, Y)])
    ctx.fill()
  }
  // The far lane, beyond the tiles, as one piece in the fog.
  if (x1 > tileFar) {
    ctx.fillStyle = mixColor(rgb(C.lane), fog, fogAt(tileFar - c.x))
    polyPath(ctx, [P(tileFar, -Y), P(x1, -Y), P(x1, Y), P(tileFar, Y)])
    ctx.fill()
  }
  // Lane lines along the corridor's middle.
  ctx.strokeStyle = mixColor(rgb(C.tileLine), fog, 0.2)
  ctx.lineWidth = 1
  ctx.setLineDash([])
  ctx.beginPath()
  for (let x = Math.ceil(x0 / 4) * 4; x < Math.min(x1, c.x + 70); x += 4) {
    ctx.moveTo(...P(x, 0))
    ctx.lineTo(...P(x + 1.6, 0))
  }
  ctx.stroke()
  // Gap lanes: the green opening on the ground, under each obstacle that has one.
  for (const o of v.obstacles) {
    if (!o.gap || o.x1 <= x0 || o.x0 > x1) continue
    const a = Math.max(o.x0, x0)
    const f = fogAt(a - c.x)
    const pts = [P(a, o.gap[0]), P(o.x1, o.gap[0]), P(o.x1, o.gap[1]), P(a, o.gap[1])]
    ctx.fillStyle = mixColor(rgb(C.gapFloor), fog, f)
    polyPath(ctx, pts)
    ctx.fill()
    ctx.strokeStyle = mixColor(rgb(C.gapEdge), fog, f)
    ctx.lineWidth = 2
    ctx.stroke()
  }
  // The finish line, chequered across the corridor.
  const fx = v.finish_x
  if (fx > x0 && fx < x1) {
    const sq = 0.5
    for (let y = -Y, i = 0; y < Y - 1e-6; y += sq, i++) {
      for (let r = 0; r < 2; r++) {
        ctx.fillStyle = mixColor(rgb((i + r) % 2 ? '#f8fafc' : '#0f172a'), fog, fogAt(fx - c.x))
        polyPath(ctx, [P(fx + r * sq, y), P(fx + (r + 1) * sq, y), P(fx + (r + 1) * sq, y + sq), P(fx + r * sq, y + sq)])
        ctx.fill()
      }
    }
  }
}

/** The sky above the horizon, with stars (dark) or clouds (light), and low hills far away. */
function drawDodgeSky(ctx: CanvasRenderingContext2D, c: DodgeCamera, w: number, theme: Theme, t: number) {
  const p = PALETTE[theme]
  const C = DODGE_COLORS[theme]
  const g = ctx.createLinearGradient(0, 0, 0, c.v0)
  g.addColorStop(0, p.skyTop)
  g.addColorStop(1, C.fog)
  ctx.fillStyle = g
  ctx.fillRect(0, 0, w, c.v0 + 1)
  if (theme === 'dark') {
    for (let i = 0; i < 60; i++) {
      const x = hash(i) * w
      const y = hash(i + 100) * c.v0 * 0.85
      ctx.fillStyle = `rgba(255,255,255,${0.3 + 0.3 * Math.sin(t * 1.5 + i)})`
      ctx.fillRect(x, y, 1.5, 1.5)
    }
  } else {
    ctx.fillStyle = 'rgba(255,255,255,0.8)'
    for (let i = 0; i < 5; i++) {
      const x = ((hash(i) * w * 1.5 - t * (4 + 3 * hash(i + 9))) % (w * 1.5) + w * 1.5) % (w * 1.5) - w * 0.25
      const y = c.v0 * (0.12 + 0.5 * hash(i + 20))
      const s = 14 + 22 * hash(i + 40)
      ctx.beginPath()
      ctx.ellipse(x, y, s * 1.8, s * 0.5, 0, 0, Math.PI * 2)
      ctx.ellipse(x + s * 0.8, y - s * 0.28, s, s * 0.45, 0, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  // Hills on the horizon, shifted a little with the camera's side position.
  drawRidge(ctx, w, c.v0 + 1, Math.max(8, c.v0 * 0.12), c.y * 6, 140, mixColor(rgb(p.far), rgb(C.fog), 0.35))
}

export function drawDodge(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  v: DodgeView,
  fx: FxState,
  theme: Theme,
  opts: { done?: boolean; hitbox?: boolean } = {},
) {
  const p = PALETTE[theme]
  const C = DODGE_COLORS[theme]
  const mc = DODGE_MANEUVER_COLOR[theme]
  const okColor = theme === 'dark' ? '#86efac' : '#15803d'
  const labelBg = theme === 'dark' ? 'rgba(15,23,42,0.85)' : 'rgba(255,255,255,0.92)'
  fx.dodgeTrail = nextDodgeTrail(fx.dodgeTrail, v)
  fx.dodgeCam = nextDodgeCamPose(fx.dodgeCam, v)
  const trail = fx.dodgeTrail
  const { mainH, mapH } = dodgeSplit(h)
  const d = v.drone
  const c = dodgeCamera(w, mainH, v, fx.dodgeCam)

  // Chase view.
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, 0, w, mainH)
  ctx.clip()
  drawDodgeSky(ctx, c, w, theme, fx.t)
  drawGround(ctx, c, v, w, mainH, theme)
  for (const b of dodgeDrawnBoxes(v, c)) {
    for (const face of dodgeBoxFaces(b, c)) paintFace(ctx, c, b, face, v, theme)
  }

  // Between the camera and the near plane nothing is drawn. Then the drone's shadow on what is
  // under it (the trail is drawn on the map only: seen from behind it would hang from the drone).
  if (!v.crashed) {
    const under = v.obstacles.find((o) => dodgeOverlapsX(d, o.x0, o.x1) && !dodgeInGapLane(o, d) && dodgeAboveBody(o, d))
    const zs = under ? under.top : 0
    const shade: [number, number][] = []
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2
      shade.push(dodgeProject(c, d.x + d.hl * 0.9 * Math.cos(a), d.y + d.hw * 0.9 * Math.sin(a), zs))
    }
    ctx.fillStyle = theme === 'dark' ? 'rgba(0,0,0,0.45)' : 'rgba(15,23,42,0.28)'
    polyPath(ctx, shade)
    ctx.fill()
  }
  const pitch = dodgeTilt(v)
  const roll = dodgeRoll(trail)
  drawDrone3D(ctx, c, d, pitch, roll, fx.t, theme)
  const db = dodgeDroneBox(v)
  const box = dodgeScreenBounds(c, db)
  const edges = (color: string, dash: number[], width: number) => {
    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.setLineDash(dash)
    ctx.beginPath()
    const P = (x: number, y: number, z: number) => dodgeProject(c, x, y, z)
    for (const [y, z] of [
      [db.y0, db.z0],
      [db.y1, db.z0],
      [db.y1, db.z1],
      [db.y0, db.z1],
    ]) {
      ctx.moveTo(...P(db.x0, y, z))
      ctx.lineTo(...P(db.x1, y, z))
    }
    for (const x of [db.x0, db.x1]) {
      const q = [P(x, db.y0, db.z0), P(x, db.y1, db.z0), P(x, db.y1, db.z1), P(x, db.y0, db.z1)]
      ctx.moveTo(...q[0])
      for (const r of q.slice(1)) ctx.lineTo(...r)
      ctx.closePath()
    }
    ctx.stroke()
    ctx.restore()
  }
  if (opts.hitbox && !v.crashed) edges('rgba(239,68,68,0.85)', [3, 2], 1)
  const outcome = dodgeOutcome(v, opts.done ?? (v.crashed || v.finished || v.aborted))

  // The held maneuver, and above it (it takes priority) the reflex's replacement, if any: above
  // the drone when there is room under the top line, below it otherwise. Placed first, drawn last.
  const stack: { text: string; style: PillStyle }[] = [
    {
      text: dodgeManeuverLabel(v),
      style: { font: LABEL_FONT, bg: labelBg, fg: p.text, border: mc[v.action] ?? mc.continue, dot: mc[v.action] ?? mc.continue },
    },
  ]
  if (!outcome && v.vetoed && v.vetoed_request) {
    // The reflex replaced the request (it can hold the drone in front of an obstacle until the
    // model aborts or the episode times out).
    const held = d.v < 0.5 ? ' · holding' : ''
    stack.push({ text: `reflex: ${v.vetoed_request} → ${v.action}${held}`, style: { font: REFLEX_FONT, bg: '#dc2626', fg: '#fff' } })
  }
  const CH = 17
  const stackH = stack.length * CH + (stack.length - 1) * 3
  const above = box.y0 - 8 - stackH >= (outcome ? 72 : 26)
  const mid = (box.x0 + box.x1) / 2
  type Rect = { l: number; t: number; w: number; h: number }
  const taken: Rect[] = [{ l: box.x0, t: box.y0, w: box.x1 - box.x0, h: box.y1 - box.y0 }]
  const chips = stack.map((it, i) => {
    const pw = pillWidth(ctx, it.text, it.style)
    const r = { l: clamp(mid - pw / 2, 4, w - 4 - pw), t: above ? box.y0 - 8 - (i + 1) * CH - i * 3 : box.y1 + 8 + i * (CH + 3), w: pw, h: CH }
    taken.push(r)
    return { ...it, r }
  })
  const hits = (r: Rect) => taken.some((q) => r.l < q.l + q.w + 3 && q.l < r.l + r.w + 3 && r.t < q.t + q.h + 2 && q.t < r.t + r.h + 2)

  // Clearances while the drone overlaps an obstacle: a dimension mark from its box to the
  // obstacle's top, the deck's underside or the body's side, and the free space in meters, in a
  // label put where it covers neither the drone nor another label.
  const PH = 15
  const cxp = dodgeProject(c, d.x, d.y, d.z)
  const style: PillStyle = { font: LABEL_FONT, bg: labelBg, fg: okColor }
  for (const cl of dodgeClearances(v)) {
    const P = (y: number, z: number) => dodgeProject(c, d.x, y, z)
    let a: [number, number]
    let b: [number, number]
    let text: string
    if (cl.kind === 'beside') {
      const toward = cl.edge! < d.y ? d.y - d.hw : d.y + d.hw
      a = P(toward, d.z)
      b = P(cl.edge!, d.z)
      text = `${dodgeMeters(cl.m)} m to the wall`
    } else {
      const ym = d.y - d.hw
      a = cl.kind === 'above' ? P(ym, d.z - d.hh) : P(ym, d.z + d.hh)
      b = cl.kind === 'above' ? P(ym, cl.o.top) : P(ym, cl.o.deck![2])
      text = cl.kind === 'above' ? `${dodgeMeters(cl.m)} m above` : `${dodgeMeters(cl.m)} m under the deck`
    }
    dimensionMark(ctx, a, b, okColor)
    const pw = pillWidth(ctx, text, style)
    const yc =
      cl.kind === 'beside'
        ? box.y1 + 6 + PH / 2
        : cl.kind === 'above'
          ? Math.max(cxp[1] + 9, (a[1] + b[1]) / 2)
          : Math.min(cxp[1] - 9, (a[1] + b[1]) / 2)
    const xs = cl.kind === 'beside' ? [(a[0] + b[0]) / 2 - pw / 2, box.x1 + 8, box.x0 - 8 - pw] : [box.x1 + 8, box.x0 - 8 - pw]
    const ys = [yc - PH / 2, box.y1 + 6, box.y1 + 10 + PH, box.y0 - 6 - PH]
    let pick: Rect | null = null
    search: for (const t of ys) {
      for (const l of xs) {
        const r = { l: clamp(l, 4, w - 4 - pw), t: clamp(t, 28, mainH - PH - 4), w: pw, h: PH }
        pick ??= r
        if (!hits(r)) {
          pick = r
          break search
        }
      }
    }
    taken.push(pick!)
    drawPill(ctx, text, pick!.l, pick!.t, pw, PH, style)
  }
  for (const it of chips) drawPill(ctx, it.text, it.r.l, it.r.t, it.r.w, it.r.h, it.style)

  if (v.crashed) {
    // The box the core tested, where it touched.
    edges('#ef4444', [], 2)
    const since = fx.t - fx.crashAt
    const r0 = (box.x1 - box.x0) / 2
    for (let i = 0; i < 16 && since < 1.2; i++) {
      const a = (i / 16) * Math.PI * 2
      const r = r0 * 0.4 + since * 110 * (0.6 + 0.4 * hash(i))
      ctx.fillStyle = `rgba(251,146,60,${1 - since / 1.2})`
      ctx.beginPath()
      ctx.arc(cxp[0] + Math.cos(a) * r, cxp[1] + Math.sin(a) * r, 3, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  const secs = Math.floor(v.frame / 60)
  const status = `${d.v.toFixed(1)} m/s · alt ${d.z.toFixed(1)} m · ${Math.max(0, v.finish_x - d.x).toFixed(0)} m to go · ${secs} / ${DODGE_MAX_S} s`
  const head: PillStyle = { font: '600 12px ui-sans-serif, system-ui, sans-serif', bg: labelBg, fg: p.text }
  drawPill(ctx, status, 4, 5, pillWidth(ctx, status, head), 19, head)
  if (outcome) {
    ctx.font = '700 16px ui-sans-serif, system-ui, sans-serif'
    const tw = ctx.measureText(outcome.text).width + 32
    ctx.fillStyle = outcome.ok ? 'rgba(34,197,94,0.92)' : 'rgba(239,68,68,0.92)'
    roundRect(ctx, (w - tw) / 2, 30, tw, 36, 10)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.fillText(outcome.text, w / 2, 54)
    ctx.textAlign = 'left'
  }
  ctx.restore()

  drawDodgeMap(ctx, w, mainH, mapH, v, trail, fx, theme)
}

/** Meters of course the map shows behind the drone, and in all. */
export const DODGE_MAP_BEHIND_M = 10
export const DODGE_MAP_SPAN_M = 70

/** The map below the chase view: the corridor from above, x to the right and y (left) up. Every
 * building is opaque, its gap lane the green opening; a deck is outlined across the corridor,
 * its pillars outside it; the drone flies above it all, its shadow offset by its altitude. */
function drawDodgeMap(
  ctx: CanvasRenderingContext2D,
  w: number,
  ty0: number,
  topH: number,
  v: DodgeView,
  trail: DodgeTrailPoint[],
  fx: FxState,
  theme: Theme,
) {
  const p = PALETTE[theme]
  const d = v.drone
  const kx = w / DODGE_MAP_SPAN_M
  const x0m = d.x - DODGE_MAP_BEHIND_M
  const X = (x: number) => (x - x0m) * kx
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, ty0, w, topH)
  ctx.clip()
  const bg = theme === 'dark' ? '#0f172a' : '#e2e8f0'
  ctx.fillStyle = bg
  ctx.fillRect(0, ty0, w, topH)
  const ky = (topH - 14) / (2 * v.y_max)
  const Y = (y: number) => ty0 + 7 + (v.y_max - y) * ky
  ctx.fillStyle = theme === 'dark' ? '#1e293b' : '#f8fafc'
  ctx.fillRect(0, Y(v.y_max), w, Y(-v.y_max) - Y(v.y_max))
  ctx.strokeStyle = theme === 'dark' ? 'rgba(148,163,184,0.25)' : 'rgba(15,23,42,0.15)'
  ctx.lineWidth = 1
  ctx.setLineDash([8, 8])
  ctx.beginPath()
  ctx.moveTo(0, Y(0))
  ctx.lineTo(w, Y(0))
  ctx.stroke()
  ctx.setLineDash([])
  const next = dodgeNextObstacle(v)
  const bodyColor = theme === 'dark' ? '#64748b' : '#94a3b8'
  const wallColor = theme === 'dark' ? '#b91c1c' : '#dc2626'
  const gapColor = theme === 'dark' ? '#166534' : '#86efac'
  const deckColor = theme === 'dark' ? '#cbd5e1' : '#334155'
  const fx0 = X(v.finish_x)
  if (fx0 > -10 && fx0 < w + 10) {
    const sq = Math.max(3, (Y(-v.y_max) - Y(v.y_max)) / 8)
    for (let i = 0; Y(v.y_max) + i * sq < Y(-v.y_max) - 0.5; i++) {
      for (let r = 0; r < 2; r++) {
        ctx.fillStyle = (i + r) % 2 ? '#f8fafc' : '#0f172a'
        ctx.fillRect(fx0 + r * sq, Y(v.y_max) + i * sq, sq, Math.min(sq, Y(-v.y_max) - Y(v.y_max) - i * sq))
      }
    }
  }
  for (const o of v.obstacles) {
    const a = X(o.deck ? o.deck[0] : o.x0)
    const b = X(o.deck ? o.deck[1] : o.x1)
    if (b < -10 || a > w + 10) continue
    const x0 = X(o.x0)
    const x1 = X(o.x1)
    ctx.fillStyle = o.impassable ? wallColor : bodyColor
    if (o.gap) {
      const [lo, hi] = o.gap
      // The body covers the corridor except the gap lane.
      if (hi < v.y_max) ctx.fillRect(x0, Y(v.y_max), x1 - x0, Y(hi) - Y(v.y_max))
      if (lo > -v.y_max) ctx.fillRect(x0, Y(lo), x1 - x0, Y(-v.y_max) - Y(lo))
      ctx.fillStyle = gapColor
      ctx.fillRect(x0, Y(hi), x1 - x0, Y(lo) - Y(hi))
    } else {
      ctx.fillRect(x0, Y(v.y_max), x1 - x0, Y(-v.y_max) - Y(v.y_max))
    }
    if (o.deck) {
      // The deck spans the whole width overhead: its outline, and its pillars outside the corridor.
      const [d0, d1] = o.deck
      ctx.strokeStyle = deckColor
      ctx.lineWidth = 1.5
      ctx.strokeRect(X(d0), Y(v.y_max), X(d1) - X(d0), Y(-v.y_max) - Y(v.y_max))
      ctx.fillStyle = deckColor
      const pw = Math.max(2, PILLAR_M * kx)
      for (const xx of [X(d0), X(d1) - pw]) {
        ctx.fillRect(xx, Y(v.y_max) - 4, pw, 3)
        ctx.fillRect(xx, Y(-v.y_max) + 1, pw, 3)
      }
    }
    if (o === next) {
      // The obstacle the current decision is about.
      ctx.strokeStyle = theme === 'dark' ? 'rgba(248,250,252,0.6)' : 'rgba(15,23,42,0.55)'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.strokeRect(a - 2, Y(v.y_max) - 2, b - a + 4, Y(-v.y_max) - Y(v.y_max) + 4)
      ctx.setLineDash([])
    }
  }
  drawTrail(ctx, trail, v.frame, (q) => [X(q.x), Y(q.y)], theme, 2)
  // The drone above everything, with its shadow on what is under it (a roof or the ground),
  // offset by its height above it.
  const under = v.obstacles.find((o) => dodgeOverlapsX(d, o.x0, o.x1) && !dodgeInGapLane(o, d) && dodgeAboveBody(o, d))
  const alt = Math.max(0, d.z - d.hh - (under ? under.top : 0))
  const off = Math.min(9, 1 + alt * 0.5)
  const size = Math.min(2 * d.hl * kx, 2 * d.hw * ky) / 1.12
  ctx.fillStyle = theme === 'dark' ? 'rgba(0,0,0,0.55)' : 'rgba(15,23,42,0.3)'
  ctx.beginPath()
  ctx.ellipse(X(d.x) + off, Y(d.y) + off * 0.7, Math.max(2, d.hl * kx * 0.9), Math.max(2, d.hw * ky * 0.9), 0, 0, Math.PI * 2)
  ctx.fill()
  if (Math.max(2 * d.hl * kx, 2 * d.hw * ky) >= 10) drawDroneTop(ctx, X(d.x), Y(d.y), size, fx.t, theme)
  else {
    ctx.fillStyle = '#0ea5e9'
    ctx.beginPath()
    ctx.ellipse(X(d.x), Y(d.y), Math.max(2, d.hl * kx), Math.max(2, d.hw * ky), 0, 0, Math.PI * 2)
    ctx.fill()
  }
  const tag: PillStyle = { font: LABEL_FONT, bg: theme === 'dark' ? 'rgba(15,23,42,0.85)' : 'rgba(255,255,255,0.9)', fg: p.text }
  drawPill(ctx, 'map', 4, ty0 + 3, pillWidth(ctx, 'map', tag), 14, tag)
  ctx.restore()
}
