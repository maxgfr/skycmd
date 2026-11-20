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
  /** Smoothed vertical extent of the dodge side view, in meters (0 = not set). */
  dodgeZ: number
  /** The dodge drone's recent positions, oldest first (see [`nextDodgeTrail`]). */
  dodgeTrail: DodgeTrailPoint[]
}

export function newFx(): FxState {
  return { t: 0, thrustGlow: 0, tween: 1, crashAt: -10, dodgeZ: 0, dodgeTrail: [] }
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
 * Extent of the side-view quadcopter of [`drawDroneSide`] around its anchor, in units of its
 * `size` (y down): the rotor discs reach ±0.92 sideways and 0.36 up, the skids 0.24 down. The
 * air-flow streaks under the rotors are not counted.
 */
export const DRONE_SIDE_EXTENT = { left: -0.92, right: 0.92, top: -0.36, bottom: 0.24 } as const

/** The `size` that makes the side-view quadcopter, rotor tip to rotor tip, `span` pixels long. */
export function droneSizeForSpan(span: number): number {
  return span / (DRONE_SIDE_EXTENT.right - DRONE_SIDE_EXTENT.left)
}

/**
 * A quadcopter seen from the side: body, two visible arms with spinning rotors, status LEDs.
 * `size` is the body length in pixels (the rotors make it 1.84 `size` long, see
 * [`DRONE_SIDE_EXTENT`]); `tilt` in radians (positive = nose down).
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

/** Meters of course shown behind the drone (its trail), and the least shown in all. */
export const DODGE_BEHIND_M = 8
export const DODGE_MIN_VIEW_M = 44

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
 * True when the obstacle is not in the drone's way at its altitude: above the body and, when
 * there is a deck, not above its underside (the core's deck test is `z + hh > ceil`). The top
 * view ghosts such an obstacle.
 */
export function dodgeClearsAtAltitude(o: DodgeObstacle, d: DodgeView['drone']): boolean {
  return dodgeAboveBody(o, d) && (!o.deck || d.z + d.hh <= o.deck[2])
}

/** A clearance to show while the drone overlaps an obstacle along x: `m` meters of free space
 * between its box and the obstacle's top (`above`) or the deck's underside (`under`). */
export interface DodgeClearance {
  kind: 'above' | 'under'
  m: number
  o: DodgeObstacle
}

/** The clearances of the obstacles the drone's x-span overlaps (the deck overhang included). A
 * body the drone passes beside, in its gap lane, has no vertical clearance to show. */
export function dodgeClearances(v: DodgeView): DodgeClearance[] {
  const d = v.drone
  if (v.crashed) return []
  const out: DodgeClearance[] = []
  for (const o of v.obstacles) {
    if (dodgeOverlapsX(d, o.x0, o.x1) && !dodgeInGapLane(o, d) && dodgeAboveBody(o, d)) {
      out.push({ kind: 'above', m: d.z - d.hh - o.top, o })
    }
    if (o.deck && dodgeOverlapsX(d, o.deck[0], o.deck[1]) && d.z + d.hh <= o.deck[2]) {
      out.push({ kind: 'under', m: o.deck[2] - (d.z + d.hh), o })
    }
  }
  return out
}

/** Meters with one decimal, rounded down so that a clearance is never overstated. */
export function dodgeMeters(m: number): string {
  return (Math.floor(m * 10 + 1e-6) / 10).toFixed(1)
}

/** Altitude the side view zooms to: the drone and the obstacles ahead, at least 8 m. */
export function dodgeZoomTarget(v: DodgeView): number {
  const ahead = v.obstacles.filter((o) => o.x1 > v.drone.x - DODGE_BEHIND_M && o.x0 < v.drone.x + 50)
  const zTop = Math.max(v.drone.z + 2, ...ahead.map((o) => (o.deck ? o.deck[2] + 1.5 : o.top + 2)))
  return clamp(zTop + 1, 8, v.z_max + 2)
}

/** World-to-canvas mapping of the Dodge views, for a side view `zView` meters tall. */
export interface DodgeLayout {
  sideH: number
  topH: number
  /** Pixels per meter of the side view (both axes) and of the top view's x axis. */
  k: number
  camX: number
  X: (x: number) => number
  Z: (z: number) => number
  /** `size` of the side-view drone: rotor tip to rotor tip it spans the hitbox length. */
  droneSize: number
}

export function dodgeLayout(w: number, h: number, v: DodgeView, zView: number): DodgeLayout {
  const sideH = Math.round(h * 0.7)
  let k = sideH / zView // px per meter
  if (w / k < DODGE_MIN_VIEW_M) k = w / DODGE_MIN_VIEW_M // at least 44 m of course in view
  // The drone 8 m from the left edge: its trail shows behind it, 36 m or more of course ahead.
  const camX = v.drone.x - DODGE_BEHIND_M
  return {
    sideH,
    topH: h - sideH,
    k,
    camX,
    X: (x: number) => (x - camX) * k,
    Z: (z: number) => sideH - 6 - z * k,
    droneSize: droneSizeForSpan(2 * v.drone.hl * k),
  }
}

/**
 * True when the drone is laterally inside the side gap of `o` (as the core tests it): it flies
 * beside the obstacle's body, so the side view, which cannot show depth, draws it crossing the
 * body. The renderer ghosts such an obstacle.
 */
export function dodgeInGapLane(o: DodgeObstacle, d: DodgeView['drone']): boolean {
  return !!o.gap && d.y - d.hw >= o.gap[0] && d.y + d.hw <= o.gap[1]
}

/** Drawn thickness of a deck, in px, above its underside (the surface the core tests). */
export function dodgeDeckThickness(k: number): number {
  return Math.max(5, 0.8 * k)
}

/** Drawn width of a deck's supports, in px. */
export function dodgeSupportWidth(k: number): number {
  return Math.max(3, 0.3 * k)
}

/** Pitch of the side-view drone, in radians: nose up when braking, slightly down at speed. */
export function dodgeTilt(v: DodgeView): number {
  return v.action === 'brake' || v.action === 'abort' ? -0.2 : clamp(v.drone.v / 60, 0, 0.18)
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

/** A vertical dimension mark from `y0` to `y1` at `x`, with end ticks. */
function dimensionMark(ctx: CanvasRenderingContext2D, x: number, y0: number, y1: number, color: string) {
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(x, y0)
  ctx.lineTo(x, y1)
  ctx.moveTo(x - 3, y0)
  ctx.lineTo(x + 3, y0)
  ctx.moveTo(x - 3, y1)
  ctx.lineTo(x + 3, y1)
  ctx.stroke()
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
  const mc = DODGE_MANEUVER_COLOR[theme]
  const okColor = theme === 'dark' ? '#86efac' : '#15803d'
  const labelBg = theme === 'dark' ? 'rgba(15,23,42,0.82)' : 'rgba(255,255,255,0.9)'
  const nextOutline = theme === 'dark' ? 'rgba(248,250,252,0.55)' : 'rgba(15,23,42,0.5)'
  // Zoom on what matters: the drone and the obstacles ahead, with at least 36 m of lookahead.
  const target = dodgeZoomTarget(v)
  fx.dodgeZ = fx.dodgeZ ? fx.dodgeZ + (target - fx.dodgeZ) * 0.04 : target
  fx.dodgeTrail = nextDodgeTrail(fx.dodgeTrail, v)
  const trail = fx.dodgeTrail
  const { sideH, topH, k, camX, X, Z, droneSize } = dodgeLayout(w, h, v, fx.dodgeZ)
  const d = v.drone
  const next = dodgeNextObstacle(v)

  // Side view.
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, 0, w, sideH)
  ctx.clip()
  drawSky(ctx, w, sideH, theme, fx.t)
  drawRidge(ctx, w, Z(0), 7 * k, camX * k * 0.2, 30 * k, p.far)
  drawSkyline(ctx, w, Z(0), 1.1 * k, camX * k * 0.45, p.mid)
  ctx.fillStyle = p.ground
  ctx.fillRect(0, Z(0), w, sideH - Z(0))

  // Finish line.
  const fxp = X(v.finish_x)
  if (fxp > -20 && fxp < w + 20) {
    const sq = Math.max(4, k * 0.5)
    for (let i = 0; i * sq < Z(0); i++) {
      ctx.fillStyle = i % 2 ? '#f8fafc' : '#0f172a'
      ctx.fillRect(fxp, i * sq, sq, sq)
      ctx.fillStyle = i % 2 ? '#0f172a' : '#f8fafc'
      ctx.fillRect(fxp + sq, i * sq, sq, sq)
    }
  }

  for (const o of v.obstacles) {
    const x0 = X(o.x0)
    const x1 = X(o.x1)
    if (x1 < -40 || x0 > w + 40) continue
    // The drone beside the body, in the side gap: the body is drawn as a ghost so the side view
    // does not show the drone inside it.
    const beside = dodgeInGapLane(o, d)
    const alpha = beside ? 0.28 : o.gap ? 0.75 : 1
    ctx.globalAlpha = o.passed ? Math.min(0.55, alpha) : alpha
    const g = ctx.createLinearGradient(x0, 0, x1, 0)
    g.addColorStop(0, o.impassable ? '#7f1d1d' : p.pipe[2])
    g.addColorStop(1, o.impassable ? '#b91c1c' : p.pipe[1])
    ctx.fillStyle = g
    ctx.fillRect(x0, Z(o.top), x1 - x0, Z(0) - Z(o.top))
    // Windows.
    ctx.fillStyle = theme === 'dark' ? 'rgba(253,224,71,0.55)' : 'rgba(255,255,255,0.55)'
    const ww = Math.max(2, k * 0.35)
    for (let zz = 0.8; zz < o.top - 0.6; zz += 1.4) {
      for (let xx = o.x0 + 0.5; xx < o.x1 - 0.5; xx += 1.2) {
        if (hash(xx * 3.1 + zz * 7.7) > 0.45) ctx.fillRect(X(xx), Z(zz) - ww, ww, ww)
      }
    }
    if (o.impassable) {
      ctx.strokeStyle = 'rgba(254,202,202,0.6)'
      ctx.lineWidth = 2
      ctx.save()
      ctx.beginPath()
      ctx.rect(x0, Z(o.top), x1 - x0, Z(0) - Z(o.top))
      ctx.clip()
      for (let s = -sideH; s < x1 - x0 + sideH; s += 12) {
        ctx.beginPath()
        ctx.moveTo(x0 + s, Z(o.top))
        ctx.lineTo(x0 + s - sideH, Z(o.top) + sideH)
        ctx.stroke()
      }
      ctx.restore()
    }
    const th = dodgeDeckThickness(k)
    if (o.deck) {
      const [d0, d1, ceil] = o.deck
      ctx.fillStyle = theme === 'dark' ? '#475569' : '#334155'
      ctx.fillRect(X(d0), Z(ceil) - th, X(d1) - X(d0), th)
      ctx.strokeStyle = theme === 'dark' ? '#94a3b8' : '#64748b'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let xx = d0; xx < d1; xx += 1) {
        ctx.moveTo(X(xx), Z(ceil) - th)
        ctx.lineTo(X(xx + 0.5), Z(ceil))
        ctx.lineTo(X(xx + 1), Z(ceil) - th)
      }
      ctx.stroke()
      // Supports up to the top of the view: the core's deck is solid from its underside up to
      // the airspace ceiling, so a drone above the deck level hits the front support.
      const sw = dodgeSupportWidth(k)
      ctx.fillStyle = theme === 'dark' ? 'rgba(71,85,105,0.7)' : 'rgba(51,65,85,0.6)'
      ctx.fillRect(X(d0), 0, sw, Z(ceil) - th)
      ctx.fillRect(X(d1) - sw, 0, sw, Z(ceil) - th)
    }
    ctx.globalAlpha = 1
    if (beside && !o.passed) {
      ctx.strokeStyle = 'rgba(34,197,94,0.9)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([5, 4])
      ctx.strokeRect(x0, Z(o.top), x1 - x0, Z(0) - Z(o.top))
      ctx.setLineDash([])
      // Above the body, below the top line of text, inside the canvas.
      // Lined up with the gap ahead of the body, passing beside it once along it.
      const along = dodgeOverlapsX(d, o.x0, o.x1)
      const text = `side gap (${o.gap![0] > 0 ? 'left' : 'right'}): ${along ? 'passing beside' : 'lined up'}`
      const style: PillStyle = { font: LABEL_FONT, bg: labelBg, fg: okColor }
      const pw = pillWidth(ctx, text, style)
      const left = clamp((x0 + x1) / 2 - pw / 2, 4, w - 4 - pw)
      // Above the body, unless that is where the drone's own labels go (a drone climbing beside
      // a tall body): then at its foot.
      const cx = X(d.x)
      const cy = Z(d.z)
      const clash = (top: number) => left < cx + 200 && left + pw > cx - 130 && top < cy + 32 && top + 15 > cy - 60
      let top = Math.max(26, Z(o.top) - 19)
      if (clash(top) && !clash(Z(0) - 21)) top = Z(0) - 21
      drawPill(ctx, text, left, top, pw, 15, style)
    } else if (o === next) {
      // The obstacle the current decision is about.
      ctx.strokeStyle = nextOutline
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.strokeRect(x0 - 1.5, Z(o.top) - 1.5, x1 - x0 + 3, Z(0) - Z(o.top) + 1.5)
      if (o.deck) ctx.strokeRect(X(o.deck[0]) - 1.5, Z(o.deck[2]) - th - 1.5, X(o.deck[1]) - X(o.deck[0]) + 3, th + 3)
      ctx.setLineDash([])
    }
  }

  // The last seconds of flight, in the colors of the maneuvers held.
  drawTrail(ctx, trail, v.frame, (q) => [X(q.x), Z(q.z)], theme, 2.5)

  // The sprite spans the hitbox length (2 hl), so what touches on screen is what collides.
  const cx = X(d.x)
  const cy = Z(d.z)
  const reach = droneSize * DRONE_SIDE_EXTENT.right
  drawDroneSide(ctx, cx, cy, droneSize, dodgeTilt(v), fx.t, theme, 0.4)
  if (opts.hitbox && !v.crashed) strokeHitbox(ctx, X(d.x - d.hl), Z(d.z + d.hh), 2 * d.hl * k, 2 * d.hh * k)
  const outcome = dodgeOutcome(v, opts.done ?? (v.crashed || v.finished || v.aborted))

  // Clearances while the drone overlaps an obstacle: a dimension mark just ahead of the drone,
  // from its box to the obstacle's top or the deck's underside, and the free space in meters.
  const PH = 15
  const markX = cx + reach + 3
  for (const c of dodgeClearances(v)) {
    const y0 = c.kind === 'above' ? Z(d.z - d.hh) : Z(d.z + d.hh)
    const y1 = c.kind === 'above' ? Z(c.o.top) : Z(c.o.deck![2])
    // A high deck is not what the drone is squeezing under: its distance only, no long mark.
    const mark = c.kind === 'above' || c.m <= 3
    if (mark) dimensionMark(ctx, markX, y0, y1, okColor)
    const text = c.kind === 'above' ? `${dodgeMeters(c.m)} m above` : `${dodgeMeters(c.m)} m under the deck`
    const style: PillStyle = { font: LABEL_FONT, bg: labelBg, fg: okColor }
    const pw = pillWidth(ctx, text, style)
    const mid = mark ? (y0 + y1) / 2 : cy
    const pyc = c.kind === 'above' ? Math.max(cy + 9, mid) : Math.min(cy - 9, mid)
    const left = markX + 6 + pw <= w - 4 ? markX + 6 : cx - reach - 6 - pw
    drawPill(ctx, text, left, pyc - PH / 2, pw, PH, style)
  }

  // The held maneuver, and above it (it takes priority) the reflex's replacement, if any: above
  // the drone when there is room under the top line, below it otherwise.
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
  const off = 20
  const stackH = stack.length * CH + (stack.length - 1) * 3
  const above = cy - off - stackH >= 26
  stack.forEach((it, i) => {
    const pw = pillWidth(ctx, it.text, it.style)
    const top = above ? cy - off - (i + 1) * CH - i * 3 : cy + off + i * (CH + 3)
    drawPill(ctx, it.text, clamp(cx - pw / 2, 4, w - 4 - pw), top, pw, CH, it.style)
  })

  if (v.crashed) {
    // The box the core tested, where it touched.
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2
    ctx.strokeRect(X(d.x - d.hl), Z(d.z + d.hh), 2 * d.hl * k, 2 * d.hh * k)
    const since = fx.t - fx.crashAt
    for (let i = 0; i < 16 && since < 1.2; i++) {
      const a = (i / 16) * Math.PI * 2
      const r = since * 90 * (0.6 + 0.4 * hash(i))
      ctx.fillStyle = `rgba(251,146,60,${1 - since / 1.2})`
      ctx.beginPath()
      ctx.arc(X(d.x) + Math.cos(a) * r, Z(d.z) + Math.sin(a) * r, 3, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  ctx.fillStyle = p.text
  ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif'
  ctx.textAlign = 'left'
  const secs = Math.floor(v.frame / 60)
  ctx.fillText(
    `${d.v.toFixed(1)} m/s · alt ${d.z.toFixed(1)} m · ${Math.max(0, v.finish_x - d.x).toFixed(0)} m to go · ${secs} / ${DODGE_MAX_S} s`,
    10,
    18,
  )
  if (outcome) {
    ctx.font = '700 16px ui-sans-serif, system-ui, sans-serif'
    const tw = ctx.measureText(outcome.text).width + 32
    ctx.fillStyle = outcome.ok ? 'rgba(34,197,94,0.92)' : 'rgba(239,68,68,0.92)'
    roundRect(ctx, (w - tw) / 2, sideH / 2 - 20, tw, 40, 10)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.fillText(outcome.text, w / 2, sideH / 2 + 6)
    ctx.textAlign = 'left'
  }
  ctx.restore()

  // Top view strip (lateral position y).
  const ty0 = sideH
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, ty0, w, topH)
  ctx.clip()
  ctx.fillStyle = theme === 'dark' ? '#0f172a' : '#e2e8f0'
  ctx.fillRect(0, ty0, w, topH)
  const ky = (topH - 16) / (2 * v.y_max)
  const Y = (y: number) => ty0 + 8 + (v.y_max - y) * ky
  ctx.fillStyle = theme === 'dark' ? '#1e293b' : '#f8fafc'
  ctx.fillRect(0, Y(v.y_max), w, Y(-v.y_max) - Y(v.y_max))
  ctx.strokeStyle = theme === 'dark' ? 'rgba(148,163,184,0.25)' : 'rgba(15,23,42,0.15)'
  ctx.setLineDash([8, 8])
  ctx.beginPath()
  ctx.moveTo(0, Y(0))
  ctx.lineTo(w, Y(0))
  ctx.stroke()
  ctx.setLineDash([])
  let overLabel: number | null = null
  for (const o of v.obstacles) {
    const x0 = X(o.x0)
    const x1 = X(o.x1)
    if (x1 < -40 || x0 > w + 40) continue
    // At the drone's altitude, an obstacle whose body is below it (and whose deck, if any, is
    // above it) is not in its way: a ghost, as the side view draws a body passed beside.
    const clear = !v.crashed && dodgeClearsAtAltitude(o, d)
    const alpha = clear ? 0.28 : 1
    ctx.fillStyle = o.impassable ? '#b91c1c' : p.pipe[1]
    ctx.globalAlpha = o.passed ? Math.min(0.5, alpha) : alpha
    if (o.gap) {
      const [lo, hi] = o.gap
      // Obstacle covers the corridor except the gap.
      if (hi < v.y_max) ctx.fillRect(x0, Y(v.y_max), x1 - x0, Y(hi) - Y(v.y_max))
      if (lo > -v.y_max) ctx.fillRect(x0, Y(lo), x1 - x0, Y(-v.y_max) - Y(lo))
      ctx.fillStyle = 'rgba(34,197,94,0.3)'
      ctx.fillRect(x0, Y(hi), x1 - x0, Y(lo) - Y(hi))
    } else {
      ctx.fillRect(x0, Y(v.y_max), x1 - x0, Y(-v.y_max) - Y(v.y_max))
    }
    ctx.globalAlpha = 1
    if (clear && !o.passed) {
      ctx.strokeStyle = 'rgba(34,197,94,0.9)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([5, 4])
      ctx.strokeRect(x0, Y(v.y_max), x1 - x0, Y(-v.y_max) - Y(v.y_max))
      ctx.setLineDash([])
      if (dodgeOverlapsX(d, o.x0, o.x1)) overLabel = (x0 + x1) / 2
    } else if (o === next) {
      ctx.strokeStyle = nextOutline
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.strokeRect(x0 - 1.5, Y(v.y_max) - 1.5, x1 - x0 + 3, Y(-v.y_max) - Y(v.y_max) + 3)
      ctx.setLineDash([])
    }
  }
  drawTrail(ctx, trail, v.frame, (q) => [X(q.x), Y(q.y)], theme, 2)
  // The drone's hitbox: hl along x (side-view scale k), hw across (scale ky).
  ctx.fillStyle = '#0ea5e9'
  ctx.beginPath()
  ctx.ellipse(X(d.x), Y(d.y), Math.max(2, d.hl * k), Math.max(2, d.hw * ky), 0, 0, Math.PI * 2)
  ctx.fill()
  if (overLabel != null) {
    const style: PillStyle = { font: LABEL_FONT, bg: labelBg, fg: okColor }
    const text = 'flying over'
    const pw = pillWidth(ctx, text, style)
    drawPill(ctx, text, clamp(overLabel - pw / 2, 4, w - 4 - pw), Y(-v.y_max) - PH - 2, pw, PH, style)
  }
  ctx.fillStyle = p.text
  ctx.globalAlpha = 0.7
  ctx.font = LABEL_FONT
  ctx.fillText('top view', 10, ty0 + 14)
  ctx.globalAlpha = 1
  ctx.restore()
}
