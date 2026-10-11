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
}

export function newFx(): FxState {
  return { t: 0, thrustGlow: 0, tween: 1, crashAt: -10, dodgeZ: 0 }
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
 * `size` is the body length in pixels; `tilt` in radians (positive = nose down).
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
  opts: { showScore?: boolean } = {},
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

const WIND_DIRS: Record<number, [number, number]> = {
  2: [0, -1], // toward north (up)
  3: [0, 1], // south
  4: [1, 0], // east
  5: [-1, 0], // west
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

export function drawDodge(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  v: DodgeView,
  fx: FxState,
  theme: Theme,
) {
  const p = PALETTE[theme]
  const sideH = Math.round(h * 0.7)
  const topH = h - sideH
  // Zoom on what matters: the drone and the obstacles ahead, with at least 40 m of lookahead.
  const ahead = v.obstacles.filter((o) => o.x1 > v.drone.x - 6 && o.x0 < v.drone.x + 50)
  const zTop = Math.max(v.drone.z + 2, ...ahead.map((o) => (o.deck ? o.deck[2] + 1.5 : o.top + 2)))
  const target = clamp(zTop + 1, 8, v.z_max + 2)
  fx.dodgeZ = fx.dodgeZ ? fx.dodgeZ + (target - fx.dodgeZ) * 0.04 : target
  const zView = fx.dodgeZ
  let k = sideH / zView // px per meter
  if (w / k < 44) k = w / 44
  const camX = v.drone.x - 5
  const X = (x: number) => (x - camX) * k
  const Z = (z: number) => sideH - 6 - z * k

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
    const alpha = o.gap ? 0.75 : 1
    ctx.globalAlpha = o.passed ? 0.55 : alpha
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
    if (o.deck) {
      const [d0, d1, ceil] = o.deck
      const th = Math.max(5, 0.8 * k)
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
      // Supports up to the top of the view.
      ctx.fillStyle = theme === 'dark' ? 'rgba(71,85,105,0.7)' : 'rgba(51,65,85,0.6)'
      ctx.fillRect(X(d0), 0, Math.max(3, 0.3 * k), Z(ceil) - th)
      ctx.fillRect(X(d1) - Math.max(3, 0.3 * k), 0, Math.max(3, 0.3 * k), Z(ceil) - th)
    }
    ctx.globalAlpha = 1
  }

  const d = v.drone
  const tilt = v.action === 'brake' || v.action === 'abort' ? -0.2 : clamp(d.v / 60, 0, 0.18)
  drawDroneSide(ctx, X(d.x), Z(d.z), Math.max(18, d.hl * 2 * k * 1.3), tilt, fx.t, theme, 0.4)

  if (v.crashed) {
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
  ctx.fillText(`${d.v.toFixed(1)} m/s · alt ${d.z.toFixed(1)} m · ${Math.max(0, v.finish_x - d.x).toFixed(0)} m to go`, 10, 18)
  ctx.restore()

  // Top view strip (lateral position y).
  const ty0 = sideH
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
  for (const o of v.obstacles) {
    const x0 = X(o.x0)
    const x1 = X(o.x1)
    if (x1 < -40 || x0 > w + 40) continue
    ctx.fillStyle = o.impassable ? '#b91c1c' : p.pipe[1]
    ctx.globalAlpha = o.passed ? 0.5 : 1
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
  }
  ctx.fillStyle = '#0ea5e9'
  ctx.beginPath()
  ctx.arc(X(d.x), Y(d.y), Math.max(4, d.hw * ky), 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = p.text
  ctx.globalAlpha = 0.7
  ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif'
  ctx.fillText('top view', 10, ty0 + 14)
  ctx.globalAlpha = 1
}
