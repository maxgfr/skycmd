// Software raycaster for the Doom-like game. It draws the JSON snapshot produced by the wasm
// module (`GameSession.snapshot().view` for the `doom` game) and keeps no game state of its own:
// only cosmetic state (weapon bob) lives in `DoomFx`.
//
// The 3D view is rendered column by column (DDA through the grid) into a low-resolution
// ImageData (320 columns, rows follow the view's aspect), then scaled to the canvas without
// smoothing. The HUD, the minimap and the banners are drawn on top at the canvas resolution.
//
// Graphics come from Freedoom (BSD-3-Clause, see /doom/README.txt and /doom/COPYING.txt).
// Every image has a code-drawn fallback, so the renderer works before (or without) the assets.
import { withBase } from 'vitepress'

// ---------------------------------------------------------------------------------------------
// Snapshot schema (produced by Rust).

export type DoomSpriteKind = 'imp' | 'zombie' | 'fireball' | 'health' | 'medikit' | 'ammo'
export type DoomSpriteState = 'idle' | 'chase' | 'attack' | 'pain' | 'dead' | 'none'

export interface DoomSprite {
  id: number
  kind: DoomSpriteKind
  x: number
  y: number
  /** 'none' for items and projectiles. */
  state: DoomSpriteState
  /** Monster facing / projectile direction. */
  angle: number
  /** Ticks spent in the current state (35 ticks per second). */
  anim: number
  /** Line of sight from the player (the renderer still does its own depth test). */
  visible: boolean
}

export interface DoomView {
  map: string
  /** 'easy' | 'normal' */
  skill: string
  /** Grid size in cells. */
  width: number
  height: number
  /**
   * width*height chars, row-major (index = y*width + x): '.' floor, '1'..'5' wall texture ids,
   * 'D' door, 'X' exit switch (a solid wall cell carrying the switch texture).
   */
  cells: string
  /** Every door cell; open in 0..1 (0 closed, 1 fully open and passable). */
  doors: { x: number; y: number; open: number }[]
  exit: { x: number; y: number; pressed: boolean }
  /**
   * Coordinates are in cell units: x east, y south. The facing vector is (cos angle, sin angle),
   * the right-hand side is (-sin angle, cos angle); turning right increases angle.
   */
  player: { x: number; y: number; angle: number; health: number; ammo: number; radius: number }
  sprites: DoomSprite[]
  /** The AI's next navigation waypoint (minimap only). */
  waypoint: { x: number; y: number } | null
  /** 35 Hz ticks since the level started. */
  tick: number
  kills: number
  monsters: number
  items: number
  items_total: number
  score: number
  /** Ticks since the last shot (large when none). */
  fire_tick: number
  /** Ticks since the player last took damage. */
  hurt_tick: number
  action: string
  vetoed: boolean
  vetoed_request: string | null
  dead: boolean
  completed: boolean
}

// ---------------------------------------------------------------------------------------------
// Assets.

/** A wall texture stored column-major (index = x * h + y) for fast vertical spans. */
export interface WallTex {
  w: number
  h: number
  col: Uint32Array
}

/** A floor / ceiling flat stored row-major; w and h are powers of two. */
export interface FlatTex {
  w: number
  h: number
  px: Uint32Array
}

/** One sprite frame: rect inside the sheet plus the Doom patch offsets (origin = feet). */
export interface SpriteFrame {
  x: number
  y: number
  w: number
  h: number
  /** Doom leftoffset: distance from the left edge to the sprite's center line. */
  ox: number
  /** Doom topoffset: distance from the top edge down to the floor contact point. */
  oy: number
}

/** A sprite sheet stored column-major (index = x * h + y), ABGR pixels with real alpha. */
export interface SpriteSheet {
  w: number
  h: number
  col: Uint32Array
  frames: SpriteFrame[]
}

export interface DoomAssets {
  /** Wall textures indexed by the cell digit ('1'..'5' -> walls[1]..walls[5]); walls[0] unused. */
  walls: WallTex[]
  door: WallTex
  exitOff: WallTex
  exitOn: WallTex
  floor: FlatTex
  ceil: FlatTex
  sprites: Record<DoomSpriteKind, SpriteSheet>
  weapon: SpriteSheet
  flash: SpriteSheet
  /** Image files that failed to load (their code-drawn fallback is used instead). */
  missing: string[]
}

type FrameRow = [x: number, w: number, h: number, ox: number, oy: number]

// Frame rects of the packed sheets in /doom (written by the asset build from Freedoom's
// buildcfg.txt offsets). Front rotation only.
const FRAMES: Record<DoomSpriteKind | 'weapon' | 'flash', FrameRow[]> = {
  // TROO A1 B1 C1 D1 (walk) E1 F1 G1 (attack) H1 (pain) I0 J0 K0 L0 M0 (death, M = corpse)
  imp: [
    [0, 48, 60, 23, 56], [49, 50, 62, 24, 58], [100, 48, 60, 25, 56], [149, 50, 62, 25, 58],
    [200, 29, 62, 14, 58], [230, 54, 61, 27, 57], [285, 71, 64, 37, 60], [357, 49, 63, 26, 59],
    [407, 49, 63, 26, 59], [457, 54, 62, 23, 58], [512, 45, 54, 23, 50], [558, 42, 43, 21, 39],
    [601, 57, 26, 29, 22],
  ],
  // POSS A1 B1 C1 D1 (walk) E1 F1 (attack) G1 (pain) H0 I0 J0 K0 L0 (death, L = corpse)
  zombie: [
    [0, 41, 57, 22, 53], [42, 41, 57, 21, 53], [84, 42, 57, 21, 53], [127, 42, 57, 21, 53],
    [170, 26, 56, 17, 52], [197, 27, 56, 18, 52], [225, 39, 55, 19, 51], [265, 47, 55, 26, 51],
    [313, 38, 42, 19, 38], [352, 44, 34, 22, 30], [397, 48, 27, 24, 23], [446, 50, 19, 25, 15],
  ],
  fireball: [[0, 15, 15, 8, 11], [16, 15, 15, 8, 11]], // BAL1 A0 B0
  health: [[0, 19, 10, 10, 10]], // STIM A0
  medikit: [[0, 29, 20, 14, 20]], // MEDI A0
  ammo: [[0, 34, 13, 17, 12]], // SBOX A0
  // Player sprites: offsets are Doom screen offsets (negative = position on the 320x200 screen).
  weapon: [[0, 67, 62, -122, -107]], // SHTG A0
  flash: [[0, 36, 47, -142, -98], [37, 58, 55, -131, -90]], // SHTF A0 B0
}

const KINDS: DoomSpriteKind[] = ['imp', 'zombie', 'fireball', 'health', 'medikit', 'ammo']

/** Packs an RGB color as a little-endian ImageData pixel (ABGR in a Uint32). */
function rgb(r: number, g: number, b: number, a = 255): number {
  return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v))
const num = (v: unknown, d = 0): number => (typeof v === 'number' && Number.isFinite(v) ? v : d)

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = url
  })
}

/** RGBA pixels of an image, row-major, as a Uint32Array view (null if unreadable). */
function imagePixels(img: HTMLImageElement): { w: number; h: number; px: Uint32Array } | null {
  const w = img.naturalWidth
  const h = img.naturalHeight
  if (!w || !h) return null
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d', { willReadFrequently: true })
  if (!g) return null
  g.drawImage(img, 0, 0)
  const data = g.getImageData(0, 0, w, h).data
  return { w, h, px: new Uint32Array(data.buffer, data.byteOffset, w * h) }
}

function toColumnMajor(w: number, h: number, px: Uint32Array): Uint32Array {
  const col = new Uint32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) col[x * h + y] = px[y * w + x]
  return col
}

const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0

// ---- Code-drawn fallbacks ---------------------------------------------------------------------

function fallbackWall(base: [number, number, number], style: 'brick' | 'panel' | 'door' | 'switch', on = false): WallTex {
  const w = 64
  const h = 128
  const col = new Uint32Array(w * h)
  const [r, g, b] = base
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let k = 1 - ((x * 7 + y * 13) % 5) * 0.03
      if (style === 'brick') {
        const row = y >> 4
        const bx = (x + (row & 1 ? 16 : 0)) & 31
        if ((y & 15) === 0 || bx === 0) k = 0.45
      } else if (style === 'panel' || style === 'switch') {
        if ((y & 63) === 0 || x === 0 || x === 63) k = 0.5
        if (style === 'switch' && x >= 22 && x < 42 && y >= 70 && y < 98) {
          const lit = x >= 27 && x < 37 && y >= 76 && y < 86
          col[x * h + y] = lit ? (on ? rgb(40, 220, 60) : rgb(220, 30, 30)) : rgb(170, 170, 170)
          continue
        }
      } else if (style === 'door') {
        if ((y & 15) === 0 || x < 3 || x > 60) k = 0.55
        if (y > 112) k = (x >> 3) & 1 ? 0.9 : 0.35
      }
      col[x * h + y] = rgb((r * k) | 0, (g * k) | 0, (b * k) | 0)
    }
  }
  return { w, h, col }
}

function fallbackFlat(a: [number, number, number], b: [number, number, number]): FlatTex {
  const px = new Uint32Array(64 * 64)
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const c = ((x >> 5) ^ (y >> 5)) & 1 ? a : b
      const k = x % 32 === 0 || y % 32 === 0 ? 0.7 : 1
      px[y * 64 + x] = rgb((c[0] * k) | 0, (c[1] * k) | 0, (c[2] * k) | 0)
    }
  }
  return { w: 64, h: 64, px }
}

/** A one-frame sheet with an ellipse (or box) silhouette, standing on its bottom edge. */
function fallbackSheet(w: number, h: number, color: [number, number, number], shape: 'ellipse' | 'box'): SpriteSheet {
  const col = new Uint32Array(w * h)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const nx = (x + 0.5) / w - 0.5
      const ny = (y + 0.5) / h - 0.5
      const inside = shape === 'box' ? true : nx * nx + ny * ny < 0.25
      if (!inside) continue
      const k = 1 - Math.abs(nx) * 0.8 - (shape === 'box' && (x === 0 || y === 0 || x === w - 1 || y === h - 1) ? 0.4 : 0)
      col[x * h + y] = rgb((color[0] * k) | 0, (color[1] * k) | 0, (color[2] * k) | 0)
    }
  }
  return { w, h, col, frames: [{ x: 0, y: 0, w, h, ox: w >> 1, oy: h }] }
}

function fallbackWeapon(): SpriteSheet {
  const s = fallbackSheet(40, 60, [90, 90, 96], 'box')
  s.frames[0] = { x: 0, y: 0, w: 40, h: 60, ox: -140, oy: -142 }
  return s
}

function fallbackFlash(): SpriteSheet {
  const s = fallbackSheet(40, 30, [255, 210, 80], 'ellipse')
  s.frames[0] = { x: 0, y: 0, w: 40, h: 30, ox: -140, oy: -118 }
  return s
}

const WALL_FALLBACK: [number, number, number][] = [
  [150, 110, 70],
  [150, 110, 70],
  [140, 140, 145],
  [80, 125, 80],
  [150, 80, 55],
  [60, 75, 110],
]

let fallbackCache: DoomAssets | null = null

function fallbackAssets(): DoomAssets {
  if (fallbackCache) return fallbackCache
  fallbackCache = {
    walls: WALL_FALLBACK.map((c, i) => fallbackWall(c, i === 2 || i === 5 ? 'panel' : 'brick')),
    door: fallbackWall([120, 95, 70], 'door'),
    exitOff: fallbackWall([120, 120, 125], 'switch'),
    exitOn: fallbackWall([120, 120, 125], 'switch', true),
    floor: fallbackFlat([70, 70, 74], [60, 60, 64]),
    ceil: fallbackFlat([50, 50, 54], [44, 44, 48]),
    sprites: {
      imp: fallbackSheet(36, 56, [150, 100, 60], 'ellipse'),
      zombie: fallbackSheet(30, 54, [90, 120, 80], 'ellipse'),
      fireball: fallbackSheet(14, 14, [255, 170, 40], 'ellipse'),
      health: fallbackSheet(16, 10, [220, 220, 230], 'box'),
      medikit: fallbackSheet(26, 18, [235, 235, 240], 'box'),
      ammo: fallbackSheet(30, 12, [150, 60, 40], 'box'),
    },
    weapon: fallbackWeapon(),
    flash: fallbackFlash(),
    missing: [],
  }
  // The fireball floats: its origin is the center.
  const fb = fallbackCache.sprites.fireball.frames[0]
  fb.oy = fb.h >> 1
  return fallbackCache
}

function sheetFrom(img: { w: number; h: number; px: Uint32Array }, rows: FrameRow[]): SpriteSheet {
  const col = toColumnMajor(img.w, img.h, img.px)
  // Real alpha: anything below half opacity is transparent, the rest opaque.
  for (let i = 0; i < col.length; i++) col[i] = col[i] >>> 24 < 128 ? 0 : (col[i] | 0xff000000) >>> 0
  const frames = rows
    .filter(([x, w, h]) => x + w <= img.w && h <= img.h)
    .map(([x, w, h, ox, oy]) => ({ x, y: 0, w, h, ox, oy }))
  return { w: img.w, h: img.h, col, frames }
}

/**
 * Loads the Freedoom images from /doom (respecting the site base). Always resolves: an image
 * that fails to load keeps its code-drawn fallback and is listed in `missing`.
 */
export async function loadDoomAssets(): Promise<DoomAssets> {
  const fb = fallbackAssets()
  const assets: DoomAssets = {
    walls: fb.walls.slice(),
    door: fb.door,
    exitOff: fb.exitOff,
    exitOn: fb.exitOn,
    floor: fb.floor,
    ceil: fb.ceil,
    sprites: { ...fb.sprites },
    weapon: fb.weapon,
    flash: fb.flash,
    missing: [],
  }
  if (typeof document === 'undefined') return assets

  const get = async (file: string) => {
    try {
      const img = await loadImage(withBase(`/doom/${file}`))
      const px = img ? imagePixels(img) : null
      if (!px) assets.missing.push(file)
      return px
    } catch {
      assets.missing.push(file)
      return null
    }
  }
  const wall = async (file: string, set: (t: WallTex) => void) => {
    const p = await get(file)
    if (p) set({ w: p.w, h: p.h, col: toColumnMajor(p.w, p.h, p.px) })
  }
  const flat = async (file: string, set: (t: FlatTex) => void) => {
    const p = await get(file)
    if (p && isPow2(p.w) && isPow2(p.h)) set({ w: p.w, h: p.h, px: new Uint32Array(p.px) })
  }
  const sheet = async (file: string, rows: FrameRow[], set: (s: SpriteSheet) => void) => {
    const p = await get(file)
    if (!p) return
    const s = sheetFrom(p, rows)
    if (s.frames.length) set(s)
  }

  const jobs: Promise<void>[] = []
  for (let i = 1; i <= 5; i++) jobs.push(wall(`wall${i}.png`, (t) => (assets.walls[i] = t)))
  jobs.push(wall('door.png', (t) => (assets.door = t)))
  jobs.push(wall('exitsw0.png', (t) => (assets.exitOff = t)))
  jobs.push(wall('exitsw1.png', (t) => (assets.exitOn = t)))
  jobs.push(flat('floor.png', (t) => (assets.floor = t)))
  jobs.push(flat('ceil.png', (t) => (assets.ceil = t)))
  const files: Record<DoomSpriteKind, string> = {
    imp: 'imp.png',
    zombie: 'zombie.png',
    fireball: 'fireball.png',
    health: 'health.png',
    medikit: 'medikit.png',
    ammo: 'ammo.png',
  }
  for (const k of KINDS) jobs.push(sheet(files[k], FRAMES[k], (s) => (assets.sprites[k] = s)))
  jobs.push(sheet('weapon.png', FRAMES.weapon, (s) => (assets.weapon = s)))
  jobs.push(sheet('weapon_flash.png', FRAMES.flash, (s) => (assets.flash = s)))
  await Promise.all(jobs)
  return assets
}

// ---------------------------------------------------------------------------------------------
// Cosmetic state.

export interface DoomFx {
  /** Seconds since start (banner pulse, waypoint blink); the caller advances it. */
  t: number
  /** Weapon bob phase in radians; drawDoom advances it as the player moves. */
  bob: number
}

export function newDoomFx(): DoomFx {
  return { t: 0, bob: 0 }
}

/** Movement tracking for the bob, kept outside DoomFx so its shape stays minimal. */
const bobState = new WeakMap<DoomFx, { x: number; y: number; tick: number; amp: number }>()

function updateBob(fx: DoomFx, x: number, y: number, tick: number, moving: boolean) {
  const s = bobState.get(fx)
  if (!s) {
    bobState.set(fx, { x, y, tick, amp: 0 })
    return 0
  }
  if (tick !== s.tick) {
    const d = Math.hypot(x - s.x, y - s.y)
    const ticks = Math.max(1, tick - s.tick)
    // Ignore teleports and level restarts.
    if (d < 1 && tick > s.tick) {
      fx.bob += d * 9
      const target = moving && d / ticks > 0.004 ? 1 : 0
      s.amp += (target - s.amp) * 0.25
    } else {
      s.amp = 0
    }
    s.x = x
    s.y = y
    s.tick = tick
  }
  return s.amp
}

// ---------------------------------------------------------------------------------------------
// Low-resolution frame buffer.

/** Columns of the low-resolution view (Doom's 320). */
const RW = 320
/** Wall height in cell units; the eye sits halfway. */
const WALL_H = 1
const EYE = 0.5
/** Partly open doors remembered per column (drawn back to front, they also clip sprites). */
const MAX_DOORS = 4

interface FrameBuffer {
  w: number
  h: number
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  img: ImageData
  px: Uint32Array
  /** Perpendicular distance of the opaque wall per column. */
  zbuf: Float32Array
  doorN: Uint8Array
  doorDist: Float32Array
  /** First screen row below each door (sprites behind it are hidden above this row). */
  doorClip: Int16Array
}

const buffers = new Map<number, FrameBuffer>()

function frameBuffer(h: number): FrameBuffer | null {
  let fb = buffers.get(h)
  if (fb) return fb
  const canvas = document.createElement('canvas')
  canvas.width = RW
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  const img = ctx.createImageData(RW, h)
  fb = {
    w: RW,
    h,
    canvas,
    ctx,
    img,
    px: new Uint32Array(img.data.buffer),
    zbuf: new Float32Array(RW),
    doorN: new Uint8Array(RW),
    doorDist: new Float32Array(RW * MAX_DOORS),
    doorClip: new Int16Array(RW * MAX_DOORS),
  }
  if (buffers.size > 6) buffers.clear()
  buffers.set(h, fb)
  return fb
}

// Cell kinds in the parsed grid.
const EMPTY = 0
const DOOR = 6
const EXIT = 7

let grid = new Uint8Array(0)
let doorOpen = new Float32Array(0)
let gridKey = ''

function parseGrid(view: DoomView, W: number, H: number) {
  const n = W * H
  if (grid.length < n) {
    grid = new Uint8Array(n)
    doorOpen = new Float32Array(n)
  }
  const cells = typeof view.cells === 'string' ? view.cells : ''
  const key = `${W}x${H}:${cells}`
  if (key !== gridKey) {
    gridKey = key
    for (let i = 0; i < n; i++) {
      const c = cells.charCodeAt(i)
      if (c === 46 || c === 32 || Number.isNaN(c)) grid[i] = EMPTY // '.', ' ', past the end
      else if (c >= 49 && c <= 53) grid[i] = c - 48 // '1'..'5'
      else if (c === 68) grid[i] = DOOR // 'D'
      else if (c === 88) grid[i] = EXIT // 'X'
      else grid[i] = 1
    }
  }
  doorOpen.fill(0, 0, n)
  for (const d of Array.isArray(view.doors) ? view.doors : []) {
    const x = Math.floor(num(d?.x, -1))
    const y = Math.floor(num(d?.y, -1))
    if (x >= 0 && y >= 0 && x < W && y < H) doorOpen[y * W + x] = clamp(num(d.open), 0, 1)
  }
}

/** Light level (0..256) for a distance in cells: Doom-like diminishing light. */
function light(d: number): number {
  return (clamp(1.12 - d * 0.085, 0.16, 1) * 256) | 0
}

/** Multiplies the RGB channels of an ABGR pixel by s/256. */
function shade(c: number, s: number): number {
  return (
    0xff000000 |
    ((((c >>> 16) & 255) * s) >> 8) << 16 |
    ((((c >>> 8) & 255) * s) >> 8) << 8 |
    (((c & 255) * s) >> 8)
  )
}

function renderScene(fb: FrameBuffer, view: DoomView, A: DoomAssets, fx: DoomFx, bobAmp: number) {
  const { w: SW, h: SH, px: out, zbuf, doorN, doorDist, doorClip } = fb
  const W = Math.max(1, Math.floor(num(view.width, 1)))
  const H = Math.max(1, Math.floor(num(view.height, 1)))
  parseGrid(view, W, H)

  const pl = view.player ?? { x: 1.5, y: 1.5, angle: 0, health: 0, ammo: 0, radius: 0.25 }
  const posX = num(pl.x, 1.5)
  const posY = num(pl.y, 1.5)
  const ang = num(pl.angle)
  const dirX = Math.cos(ang)
  const dirY = Math.sin(ang)
  // Camera plane = right-hand side, length tan(45 deg) = 1 for a 90 degree horizontal FOV.
  const planeX = -dirY
  const planeY = dirX
  const proj = SW / 2
  const hor = SH / 2

  // ---- Floor and ceiling (flat casting, one row at a time) ----
  const fl = A.floor
  const ce = A.ceil
  const rLx = dirX - planeX
  const rLy = dirY - planeY
  const rRx = dirX + planeX
  const rRy = dirY + planeY
  for (let y = 0; y < SH; y++) {
    const p = y + 0.5 - hor
    const isFloor = p > 0
    const tex = isFloor ? fl : ce
    const rowDist = (isFloor ? EYE : WALL_H - EYE) * proj / Math.abs(p)
    const s = (light(rowDist) * (isFloor ? 1 : 0.9)) | 0
    // Flats: 64 texels = half a cell (Doom flats are 64 units, a cell is 128).
    const k = tex.w * 2
    const stepX = (rowDist * (rRx - rLx) / SW) * k
    const stepY = (rowDist * (rRy - rLy) / SW) * k
    let fx0 = (posX + rowDist * rLx) * k + stepX * 0.5
    let fy0 = (posY + rowDist * rLy) * k + stepY * 0.5
    const mw = tex.w - 1
    const mh = tex.h - 1
    const tw = tex.w
    const tp = tex.px
    let o = y * SW
    for (let x = 0; x < SW; x++) {
      const tx = Math.floor(fx0) & mw
      const ty = Math.floor(fy0) & mh
      out[o++] = shade(tp[ty * tw + tx], s)
      fx0 += stepX
      fy0 += stepY
    }
  }

  // ---- Walls and doors, one ray per column ----
  const exitPressed = !!view.exit?.pressed
  const maxSteps = W + H + 4
  for (let x = 0; x < SW; x++) {
    const camX = (2 * (x + 0.5)) / SW - 1
    const rdx = dirX + planeX * camX
    const rdy = dirY + planeY * camX
    let mapX = Math.floor(posX)
    let mapY = Math.floor(posY)
    const ddx = rdx === 0 ? 1e30 : Math.abs(1 / rdx)
    const ddy = rdy === 0 ? 1e30 : Math.abs(1 / rdy)
    const stepX = rdx < 0 ? -1 : 1
    const stepY = rdy < 0 ? -1 : 1
    let sideX = rdx < 0 ? (posX - mapX) * ddx : (mapX + 1 - posX) * ddx
    let sideY = rdy < 0 ? (posY - mapY) * ddy : (mapY + 1 - posY) * ddy
    let side = 0
    let nd = 0
    let dist = 64
    let kind = 1
    let hitSide = 0
    // Partial door hits (near to far): distance, far-boundary distance, side, open.
    let d0 = 0, f0 = 0, s0 = 0, o0 = 0, d1 = 0, f1 = 0, s1 = 0, o1 = 0
    let d2 = 0, f2 = 0, s2 = 0, o2 = 0, d3 = 0, f3 = 0, s3 = 0, o3 = 0
    for (let i = 0; i < maxSteps; i++) {
      if (sideX < sideY) {
        sideX += ddx
        mapX += stepX
        side = 0
      } else {
        sideY += ddy
        mapY += stepY
        side = 1
      }
      const d = side === 0 ? sideX - ddx : sideY - ddy
      if (mapX < 0 || mapY < 0 || mapX >= W || mapY >= H) {
        dist = d
        kind = 1
        hitSide = side
        break
      }
      const idx = mapY * W + mapX
      const k = grid[idx]
      if (k === EMPTY) continue
      if (k === DOOR) {
        const o = doorOpen[idx]
        if (o >= 0.999) continue
        if (o > 0.001) {
          const far = Math.min(sideX, sideY)
          if (nd === 0) { d0 = d; f0 = far; s0 = side; o0 = o }
          else if (nd === 1) { d1 = d; f1 = far; s1 = side; o1 = o }
          else if (nd === 2) { d2 = d; f2 = far; s2 = side; o2 = o }
          else if (nd === 3) { d3 = d; f3 = far; s3 = side; o3 = o }
          if (nd < MAX_DOORS) nd++
          continue
        }
      }
      dist = d
      kind = k
      hitSide = side
      break
    }
    dist = Math.max(dist, 1e-4)
    zbuf[x] = dist
    const tex =
      kind === DOOR ? A.door : kind === EXIT ? (exitPressed ? A.exitOn : A.exitOff) : A.walls[kind] ?? A.walls[1]
    drawWallColumn(out, SW, SH, x, dist, hitSide, rdx, rdy, posX, posY, tex, kind < DOOR, 0, hor, proj)

    // Doors from far to near; each also records the row below which it no longer hides sprites.
    doorN[x] = nd
    for (let j = nd - 1; j >= 0; j--) {
      const d = j === 0 ? d0 : j === 1 ? d1 : j === 2 ? d2 : d3
      const far = j === 0 ? f0 : j === 1 ? f1 : j === 2 ? f2 : f3
      const sd = j === 0 ? s0 : j === 1 ? s1 : j === 2 ? s2 : s3
      const op = j === 0 ? o0 : j === 1 ? o1 : j === 2 ? o2 : o3
      const bottom = drawWallColumn(out, SW, SH, x, Math.max(d, 1e-4), sd, rdx, rdy, posX, posY, A.door, false, op, hor, proj)
      let clip = bottom
      // Underside of the raised door, visible when its bottom edge is above the eye.
      const zb = op * WALL_H
      if (zb > EYE) {
        const yFar = Math.min(SH, Math.ceil(hor - ((zb - EYE) * proj) / Math.max(far, 1e-4) - 0.5))
        const s = (light(d) * 0.45) | 0
        const c = shade(rgb(70, 62, 56), s)
        for (let y = Math.max(0, bottom); y < yFar; y++) out[y * SW + x] = c
        clip = Math.max(clip, yFar)
      }
      doorDist[x * MAX_DOORS + j] = d
      doorClip[x * MAX_DOORS + j] = clamp(clip, 0, SH)
    }
  }

  drawSprites(fb, view, A, posX, posY, dirX, dirY, planeX, planeY, hor, proj)
  drawWeapon(fb, view, A, fx, bobAmp)
}

/**
 * Draws one textured wall column. `zBottom` (0..1) raises the bottom edge of a door, whose
 * texture slides up with it. Returns the first screen row below the drawn span.
 */
function drawWallColumn(
  out: Uint32Array,
  SW: number,
  SH: number,
  x: number,
  dist: number,
  side: number,
  rdx: number,
  rdy: number,
  posX: number,
  posY: number,
  tex: WallTex,
  repeat: boolean,
  zBottom: number,
  hor: number,
  proj: number,
): number {
  let wx = side === 0 ? posY + dist * rdy : posX + dist * rdx
  wx -= Math.floor(wx)
  // Keep textures left-to-right as seen from the front of each face.
  if ((side === 0 && rdx < 0) || (side === 1 && rdy > 0)) wx = 1 - wx
  // Plain walls keep square texels (a 64x128 texture repeats twice across a cell); doors and
  // the switch span the cell once.
  const reps = repeat ? Math.max(1, Math.round(tex.h / tex.w)) : 1
  const tx = Math.min(tex.w - 1, Math.floor(wx * tex.w * reps) % tex.w)
  const scale = proj / dist
  const yTopF = hor - (WALL_H - EYE) * scale
  const yBotF = hor - (zBottom * WALL_H - EYE) * scale
  const y0 = Math.max(0, Math.ceil(yTopF - 0.5))
  const y1 = Math.min(SH, Math.ceil(yBotF - 0.5))
  if (y1 <= y0) return Math.max(0, y1)
  const th = tex.h
  const step = th / (WALL_H * scale)
  // Texture row at the top edge: 0 for walls, `open * height` for a raised door.
  let v = zBottom * th + (y0 + 0.5 - yTopF) * step
  const s = (light(dist) * (side === 1 ? 0.8 : 1)) | 0
  const col = tex.col
  const base = tx * th
  let o = y0 * SW + x
  for (let y = y0; y < y1; y++) {
    let tv = v | 0
    if (tv >= th) tv = th - 1
    else if (tv < 0) tv = 0
    out[o] = shade(col[base + tv], s)
    o += SW
    v += step
  }
  return y1
}

// ---- Sprites -----------------------------------------------------------------------------------

/** World units per sprite texel (monsters about 0.9 of the wall height, items 0.3-0.4). */
const SPRITE_SCALE: Record<DoomSpriteKind, number> = {
  imp: 0.016,
  zombie: 0.016,
  fireball: 0.018,
  health: 0.03,
  medikit: 0.019,
  ammo: 0.026,
}

/** Frame index in the sheet for a sprite's state and animation ticks. */
function spriteFrame(s: DoomSprite, tick: number): number {
  const a = Math.max(0, Math.floor(num(s.anim)))
  if (s.kind === 'imp') {
    switch (s.state) {
      case 'chase': return Math.floor(a / 6) % 4 // A-D
      case 'attack': return 4 + Math.min(2, Math.floor(a / 8)) // E F G
      case 'pain': return 7 // H
      case 'dead': return 8 + Math.min(4, Math.floor(a / 7)) // I..M (corpse)
      default: return 0
    }
  }
  if (s.kind === 'zombie') {
    switch (s.state) {
      case 'chase': return Math.floor(a / 8) % 4 // A-D
      case 'attack': return a < 10 ? 4 : a < 18 ? 5 : 4 // E F E
      case 'pain': return 6 // G
      case 'dead': return 7 + Math.min(4, Math.floor(a / 5)) // H..L (corpse)
      default: return 0
    }
  }
  if (s.kind === 'fireball') return Math.floor((a || tick) / 4) % 2
  return 0
}

const order: { s: DoomSprite; depth: number }[] = []

function drawSprites(
  fb: FrameBuffer,
  view: DoomView,
  A: DoomAssets,
  posX: number,
  posY: number,
  dirX: number,
  dirY: number,
  planeX: number,
  planeY: number,
  hor: number,
  proj: number,
) {
  const sprites = Array.isArray(view.sprites) ? view.sprites : []
  order.length = 0
  for (const s of sprites) {
    if (!s || !A.sprites[s.kind]) continue
    const depth = (num(s.x) - posX) * dirX + (num(s.y) - posY) * dirY
    if (depth > 0.12) order.push({ s, depth })
  }
  order.sort((a, b) => b.depth - a.depth)
  const tick = num(view.tick)
  const { w: SW, h: SH, px: out, zbuf, doorN, doorDist, doorClip } = fb
  for (const { s, depth } of order) {
    const sheet = A.sprites[s.kind]
    // A code-drawn fallback has no death frames: hide the corpse rather than a standing blob.
    if (s.state === 'dead' && sheet.frames.length === 1 && (s.kind === 'imp' || s.kind === 'zombie')) continue
    const fi = Math.min(sheet.frames.length - 1, spriteFrame(s, tick))
    const f = sheet.frames[fi]
    if (!f) continue
    const k = SPRITE_SCALE[s.kind]
    const lat = (num(s.x) - posX) * planeX + (num(s.y) - posY) * planeY
    const ppu = proj / depth
    const sx = proj * (1 + lat / depth)
    // Feet on the floor (Doom topoffset); the fireball floats with its center at mid height.
    const zTop = s.kind === 'fireball' ? EYE + (f.h * k) / 2 : f.oy * k
    const left = sx - f.ox * k * ppu
    const wpx = f.w * k * ppu
    const top = hor - (zTop - EYE) * ppu
    const hpx = f.h * k * ppu
    const x0 = Math.max(0, Math.ceil(left - 0.5))
    const x1 = Math.min(SW, Math.ceil(left + wpx - 0.5))
    const y0 = Math.max(0, Math.ceil(top - 0.5))
    const y1 = Math.min(SH, Math.ceil(top + hpx - 0.5))
    if (x1 <= x0 || y1 <= y0) continue
    const bright = s.kind === 'fireball'
    const lv = bright ? 256 : light(depth)
    const col = sheet.col
    const sh = sheet.h
    const vStep = f.h / hpx
    for (let x = x0; x < x1; x++) {
      if (depth >= zbuf[x]) continue
      // Rows hidden by nearer, partly open doors.
      let clip = y0
      const n = doorN[x]
      for (let j = 0; j < n; j++) {
        if (doorDist[x * MAX_DOORS + j] < depth) clip = Math.max(clip, doorClip[x * MAX_DOORS + j])
      }
      if (clip >= y1) continue
      const tx = Math.min(f.w - 1, Math.floor(((x + 0.5 - left) / wpx) * f.w))
      const base = (f.x + tx) * sh + f.y
      let v = (clip + 0.5 - top) * vStep
      let o = clip * SW + x
      for (let y = clip; y < y1; y++) {
        const tv = Math.min(f.h - 1, v | 0)
        const c = col[base + tv]
        if (c >>> 24) out[o] = lv === 256 ? c : shade(c, lv)
        o += SW
        v += vStep
      }
    }
  }
}

/** Draws a player sprite frame at Doom screen coordinates (320x200, bottom-anchored). */
function blitPlayerSprite(fb: FrameBuffer, sheet: SpriteSheet, fi: number, dx: number, dy: number) {
  const f = sheet.frames[Math.min(sheet.frames.length - 1, fi)]
  if (!f) return
  const { w: SW, h: SH, px: out } = fb
  // psprite position: x = 1 - leftoffset, y = WEAPONTOP (32) - topoffset, on a 200-row screen.
  const left = Math.round(1 - f.ox + dx)
  const top = Math.round(32 - f.oy + dy + (SH - 200))
  const col = sheet.col
  const sh = sheet.h
  for (let tx = 0; tx < f.w; tx++) {
    const x = left + tx
    if (x < 0 || x >= SW) continue
    const base = (f.x + tx) * sh + f.y
    for (let ty = 0; ty < f.h; ty++) {
      const y = top + ty
      if (y < 0 || y >= SH) continue
      const c = col[base + ty]
      if (c >>> 24) out[y * SW + x] = c
    }
  }
}

function drawWeapon(fb: FrameBuffer, view: DoomView, A: DoomAssets, fx: DoomFx, bobAmp: number) {
  if (view.dead) return
  const fire = num(view.fire_tick, 999)
  const bx = Math.cos(fx.bob) * 9 * bobAmp
  const by = Math.abs(Math.sin(fx.bob)) * 7 * bobAmp
  // Recoil: kick down on the shot, settle over ~8 ticks.
  const recoil = fire < 8 ? (8 - fire) * 1.6 : 0
  blitPlayerSprite(fb, A.weapon, 0, bx, by + recoil)
  if (fire < 4) blitPlayerSprite(fb, A.flash, fire < 2 ? 0 : 1, bx, by + recoil)
}

// ---------------------------------------------------------------------------------------------
// Canvas overlays: HUD strip, minimap, tints and banners.

const HUD_FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

function drawHud(ctx: CanvasRenderingContext2D, w: number, y0: number, hh: number, view: DoomView) {
  const g = ctx.createLinearGradient(0, y0, 0, y0 + hh)
  g.addColorStop(0, '#3a3633')
  g.addColorStop(1, '#1f1c1a')
  ctx.fillStyle = g
  ctx.fillRect(0, y0, w, hh)
  ctx.fillStyle = '#5c5550'
  ctx.fillRect(0, y0, w, Math.max(1, hh * 0.05))

  const pl = view.player
  const cells: { label: string; value: string; color: string }[] = [
    { label: 'HEALTH', value: `${Math.max(0, Math.round(num(pl?.health)))}%`, color: '#ef3b2c' },
    { label: 'AMMO', value: `${Math.max(0, Math.round(num(pl?.ammo)))}`, color: '#ef3b2c' },
    { label: 'KILLS', value: `${num(view.kills)}/${num(view.monsters)}`, color: '#f2c14e' },
    { label: 'ITEMS', value: `${num(view.items)}/${num(view.items_total)}`, color: '#f2c14e' },
  ]
  const mapW = Math.min(w * 0.26, 180)
  const colW = (w - mapW) / cells.length
  const big = Math.max(10, Math.min(hh * 0.5, colW * 0.24))
  const small = Math.max(8, Math.min(hh * 0.22, colW * 0.13))
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  cells.forEach((c, i) => {
    const cx = colW * (i + 0.5)
    if (i > 0) {
      ctx.fillStyle = 'rgba(0,0,0,0.35)'
      ctx.fillRect(colW * i, y0 + hh * 0.15, 1, hh * 0.7)
    }
    ctx.font = `700 ${big}px ${HUD_FONT}`
    ctx.fillStyle = 'rgba(0,0,0,0.6)'
    ctx.fillText(c.value, cx + 1, y0 + hh * 0.58 + 1)
    ctx.fillStyle = c.color
    ctx.fillText(c.value, cx, y0 + hh * 0.58)
    ctx.font = `600 ${small}px ${HUD_FONT}`
    ctx.fillStyle = '#b8b0a8'
    ctx.fillText(c.label, cx, y0 + hh * 0.88)
  })
  // Map name and skill.
  const mx = w - mapW / 2
  ctx.fillStyle = 'rgba(0,0,0,0.3)'
  ctx.fillRect(w - mapW, y0 + hh * 0.05, mapW, hh * 0.95)
  ctx.font = `700 ${Math.max(9, Math.min(hh * 0.3, mapW * 0.12))}px ${HUD_FONT}`
  ctx.fillStyle = '#e8e2da'
  ctx.fillText(String(view.map ?? '').toUpperCase().slice(0, 16), mx, y0 + hh * 0.5)
  ctx.font = `600 ${small}px ${HUD_FONT}`
  ctx.fillStyle = '#b8b0a8'
  ctx.fillText(String(view.skill ?? '').toUpperCase(), mx, y0 + hh * 0.85)
}

let miniCache: { key: string; canvas: HTMLCanvasElement } | null = null

function drawMinimap(ctx: CanvasRenderingContext2D, w: number, viewH: number, view: DoomView, fx: DoomFx) {
  const W = Math.max(1, Math.floor(num(view.width, 1)))
  const H = Math.max(1, Math.floor(num(view.height, 1)))
  const maxSide = Math.min(w * 0.34, viewH * 0.42, 240)
  const cs = Math.max(1, maxSide / Math.max(W, H))
  const mw = cs * W
  const mh = cs * H
  const pad = Math.max(6, Math.min(w, viewH) * 0.02)
  const ox = w - mw - pad
  const oy = pad
  const dpr = Math.max(1, ctx.getTransform().a)

  // Static layer (walls), cached per map and size.
  const key = `${W}x${H}:${cs.toFixed(3)}:${dpr}:${view.cells}`
  if (!miniCache || miniCache.key !== key) {
    const c = miniCache?.canvas ?? document.createElement('canvas')
    c.width = Math.ceil(mw * dpr)
    c.height = Math.ceil(mh * dpr)
    const g = c.getContext('2d')
    if (g) {
      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, mw, mh)
      g.fillStyle = 'rgba(12, 10, 10, 0.62)'
      g.fillRect(0, 0, mw, mh)
      const wallColors = ['#9a8f86', '#b08a5a', '#9aa0a6', '#6f9a6f', '#b0644a', '#6f86b0']
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const k = grid[y * W + x]
          if (k === EMPTY || k === DOOR || k === EXIT) continue
          g.fillStyle = wallColors[k] ?? wallColors[0]
          g.fillRect(x * cs, y * cs, cs + 0.02, cs + 0.02)
        }
      }
    }
    miniCache = { key, canvas: c }
  }
  ctx.save()
  ctx.globalAlpha = 0.9
  ctx.drawImage(miniCache.canvas, ox, oy, mw, mh)
  ctx.globalAlpha = 1
  ctx.strokeStyle = 'rgba(255,255,255,0.25)'
  ctx.lineWidth = 1
  ctx.strokeRect(ox - 0.5, oy - 0.5, mw + 1, mh + 1)

  // Doors: red when closed, green when open.
  for (const d of Array.isArray(view.doors) ? view.doors : []) {
    const o = clamp(num(d?.open), 0, 1)
    ctx.fillStyle = `rgb(${Math.round(200 - 140 * o)}, ${Math.round(70 + 130 * o)}, 60)`
    const inset = cs * 0.15
    ctx.fillRect(ox + num(d?.x) * cs + inset, oy + num(d?.y) * cs + inset, cs - 2 * inset, cs - 2 * inset)
  }
  // Exit switch.
  if (view.exit) {
    ctx.fillStyle = view.exit.pressed ? '#3ee070' : '#ffd23f'
    ctx.fillRect(ox + num(view.exit.x) * cs, oy + num(view.exit.y) * cs, cs, cs)
  }
  // Items and live monsters (visible ones only), projectiles.
  const r = Math.max(1.5, cs * 0.28)
  for (const s of Array.isArray(view.sprites) ? view.sprites : []) {
    if (!s) continue
    const x = ox + num(s.x) * cs
    const y = oy + num(s.y) * cs
    if (s.kind === 'imp' || s.kind === 'zombie') {
      if (!s.visible || s.state === 'dead') continue
      ctx.fillStyle = s.kind === 'imp' ? '#ff4d3a' : '#ff9a3a'
      ctx.beginPath()
      ctx.arc(x, y, r * 1.2, 0, Math.PI * 2)
      ctx.fill()
    } else if (s.kind === 'fireball') {
      ctx.fillStyle = '#ffcc33'
      ctx.beginPath()
      ctx.arc(x, y, r * 0.8, 0, Math.PI * 2)
      ctx.fill()
    } else {
      ctx.fillStyle = s.kind === 'ammo' ? '#d8b04a' : '#4ad8ff'
      ctx.fillRect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6)
    }
  }
  const pl = view.player
  const px = ox + num(pl?.x) * cs
  const py = oy + num(pl?.y) * cs
  // Waypoint with a faint line from the player.
  if (view.waypoint) {
    const wx = ox + num(view.waypoint.x) * cs
    const wy = oy + num(view.waypoint.y) * cs
    ctx.strokeStyle = 'rgba(80, 220, 255, 0.55)'
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(px, py)
    ctx.lineTo(wx, wy)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = '#50dcff'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(wx, wy, r * (1.3 + 0.25 * Math.sin(fx.t * 6)), 0, Math.PI * 2)
    ctx.stroke()
  }
  // Player arrow.
  const a = num(pl?.angle)
  const L = Math.max(5, cs * 0.9)
  ctx.translate(px, py)
  ctx.rotate(a)
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = 'rgba(0,0,0,0.7)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(L * 0.75, 0)
  ctx.lineTo(-L * 0.45, L * 0.42)
  ctx.lineTo(-L * 0.2, 0)
  ctx.lineTo(-L * 0.45, -L * 0.42)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()
  ctx.restore()
}

function drawBanner(ctx: CanvasRenderingContext2D, w: number, viewH: number, view: DoomView, fx: DoomFx) {
  const size = Math.max(16, Math.min(w * 0.075, viewH * 0.14))
  const cy = viewH * 0.38
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.fillRect(0, cy - size * 1.05, w, size * 2.1)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.font = `800 ${size}px ${HUD_FONT}`
  ctx.lineWidth = Math.max(2, size * 0.1)
  ctx.strokeStyle = '#2a0a00'
  ctx.strokeText('LEVEL COMPLETE', w / 2, cy - size * 0.2)
  const pulse = 0.85 + 0.15 * Math.sin(fx.t * 4)
  ctx.fillStyle = `rgba(255, ${Math.round(190 * pulse)}, 40, 1)`
  ctx.fillText('LEVEL COMPLETE', w / 2, cy - size * 0.2)
  ctx.font = `600 ${size * 0.32}px ${HUD_FONT}`
  ctx.fillStyle = '#e8e2da'
  ctx.fillText(
    `KILLS ${num(view.kills)}/${num(view.monsters)}   ITEMS ${num(view.items)}/${num(view.items_total)}   SCORE ${num(view.score)}`,
    w / 2,
    cy + size * 0.62,
  )
}

// ---------------------------------------------------------------------------------------------

let warned = false

/**
 * Draws one frame of the Doom-like game into a canvas of CSS size w x h (the context is
 * expected to be already scaled for devicePixelRatio, as `fitCanvas` does). `assets` may be
 * null (code-drawn fallbacks). Never throws.
 */
export function drawDoom(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  view: DoomView,
  assets: DoomAssets | null,
  fx: DoomFx,
  opts: { minimap: boolean; showHud?: boolean },
): void {
  try {
    drawDoomFrame(ctx, w, h, view ?? ({} as DoomView), assets ?? fallbackAssets(), fx ?? newDoomFx(), opts ?? { minimap: false })
  } catch (e) {
    if (!warned) {
      warned = true
      console.error('drawDoom failed', e)
    }
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, w, h)
  }
}

function drawDoomFrame(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  view: DoomView,
  A: DoomAssets,
  fx: DoomFx,
  opts: { minimap: boolean; showHud?: boolean },
) {
  const showHud = opts.showHud ?? true
  const hudH = showHud ? Math.round(clamp(h * 0.11, 26, 56)) : 0
  const viewH = Math.max(1, h - hudH)
  const rh = clamp(Math.round((RW * viewH) / Math.max(1, w)), 100, 400)
  const fb = frameBuffer(rh)
  if (!fb) return

  const pl = view.player
  const moving = /forward|back|strafe/.test(String(view.action ?? ''))
  const bobAmp = updateBob(fx, num(pl?.x), num(pl?.y), num(view.tick), moving)
  renderScene(fb, view, A, fx, bobAmp)
  fb.ctx.putImageData(fb.img, 0, 0)

  ctx.save()
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(fb.canvas, 0, 0, w, viewH)
  ctx.restore()

  const hurt = num(view.hurt_tick, 999)
  if (hurt < 10 && !view.dead) {
    ctx.fillStyle = `rgba(255, 0, 0, ${0.42 * (1 - hurt / 10)})`
    ctx.fillRect(0, 0, w, viewH)
  }
  if (opts.minimap) drawMinimap(ctx, w, viewH, view, fx)
  if (view.completed) drawBanner(ctx, w, viewH, view, fx)
  if (showHud) drawHud(ctx, w, viewH, hudH, view)
  if (view.dead) {
    ctx.fillStyle = 'rgba(110, 0, 0, 0.5)'
    ctx.fillRect(0, 0, w, h)
  }
}
