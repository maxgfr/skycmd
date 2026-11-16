// The tool list of the Router page (s6-route, docs/s6-nano3-prereg.md): parsing the lines a visitor
// types into the options `ToolRouter.route` reads, as the training and test slates were rendered
// (`sky_data::route`), the "mask names" reading of §6, and the page's presets. Pure functions, so
// that tests/router.test.ts checks them without a browser.

/** The router was trained and tested on slates of up to 7 tools (and *none*). */
export const MAX_TOOLS = 7

/** A description is cut to its first sentence and at most this many words (`DESC_WORDS`). */
export const DESC_WORDS = 30

/** The option the router appends to every tool list (`sky_infer::route::NONE_OPTION`). */
export const NONE_OPTION = 'none of these tools fits the request'

/** Whitespace runs collapsed to one space, trimmed (`sky_data::route::collapse_ws`). */
export function collapseWs(s: string): string {
  return s.split(/\s+/).filter(Boolean).join(' ')
}

/** The first sentence of a description (up to the first `.`, `!` or `?` followed by a space or the
 * end), then at most 30 words: `sky_data::route::cut_description`, the rule of every slate. */
export function cutDescription(d: string): string {
  const s = collapseWs(d)
  const chars = Array.from(s)
  let end = chars.length
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i]
    if ((c === '.' || c === '!' || c === '?') && (i + 1 === chars.length || chars[i + 1] === ' ')) {
      end = i + 1
      break
    }
  }
  return chars
    .slice(0, end)
    .join('')
    .split(' ')
    .filter(Boolean)
    .slice(0, DESC_WORDS)
    .join(' ')
}

/** `name: description`, or the name alone without a description (`option_text`). */
export function optionText(name: string, desc: string): string {
  return desc ? `${name}: ${desc}` : name
}

export interface Tool {
  name: string
  /** The description as the router reads it (cut). */
  desc: string
  /** The description was longer than its first sentence or 30 words, and was cut. */
  cut: boolean
}

export interface ToolList {
  tools: Tool[]
  /** Lines beyond the 7th, left out. */
  dropped: number
  /** Names given twice (the router would see two options with the same name). */
  duplicates: string[]
}

/** The tool list of a textarea: one tool per non-empty line, `name: description`; at most 7. */
export function parseTools(text: string): ToolList {
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
  const tools: Tool[] = lines.slice(0, MAX_TOOLS).map((l) => {
    const at = l.indexOf(':')
    const name = collapseWs(at >= 0 ? l.slice(0, at) : l)
    const raw = collapseWs(at >= 0 ? l.slice(at + 1) : '')
    const desc = cutDescription(raw)
    return { name, desc, cut: desc !== raw }
  })
  const seen = new Set<string>()
  const duplicates: string[] = []
  for (const t of tools) {
    if (seen.has(t.name) && !duplicates.includes(t.name)) duplicates.push(t.name)
    seen.add(t.name)
  }
  return { tools: tools.filter((t) => t.name), dropped: Math.max(0, lines.length - MAX_TOOLS), duplicates }
}

/** A seeded generator in [0, 1) (mulberry32). */
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** The names-masked reading of §6: every tool renamed `tool_<k>`, distinct k drawn from 1 to 99 as
 * the masked slates were (`sky_data::route::slates::masked`), the descriptions kept. The draw is
 * seeded, so the same list always gets the same names. */
export function maskNames(tools: Tool[], seed = 7): Tool[] {
  const ks = Array.from({ length: 99 }, (_, i) => i + 1)
  const r = rng(seed)
  for (let i = ks.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1))
    ;[ks[i], ks[j]] = [ks[j], ks[i]]
  }
  return tools.map((t, i) => ({ ...t, name: `tool_${ks[i]}` }))
}

/** The options sent to `ToolRouter.route` (without *none*, which the router appends). */
export function renderTools(tools: Tool[]): string[] {
  return tools.map((t) => optionText(t.name, t.desc))
}

export interface RouterRequest {
  text: string
  /** The tool the preset was written for, or `none`. Our own label, not a test answer. */
  intended: string
}

export interface RouterPreset {
  id: string
  label: string
  tools: string
  requests: RouterRequest[]
}

/** Presets written for this page (not rows of any dataset): three tool lists, each with requests
 * that should go to a tool and one that none of the tools serves. */
export const ROUTER_PRESETS: RouterPreset[] = [
  {
    id: 'home',
    label: 'Home automation',
    tools: [
      'lights_on: Turns on the lights in a given room.',
      'set_thermostat: Sets the target temperature of the heating.',
      'lock_door: Locks a door of the house.',
      'play_music: Plays a song, an artist or a playlist on the speakers.',
      'start_vacuum: Starts the robot vacuum cleaner.',
    ].join('\n'),
    requests: [
      { text: "it's freezing in here, put the heating to 22 degrees", intended: 'set_thermostat' },
      { text: 'turn on the kitchen lights please', intended: 'lights_on' },
      { text: 'put on some jazz in the living room', intended: 'play_music' },
      { text: 'what is the capital of Peru?', intended: 'none' },
    ],
  },
  {
    id: 'dev',
    label: 'Developer tools',
    tools: [
      'run_tests: Runs the test suite of the repository.',
      'create_branch: Creates a new git branch from the current commit.',
      'open_pull_request: Opens a pull request for the current branch.',
      'search_code: Searches the code base for a symbol or a piece of text.',
      'deploy_staging: Deploys the current build to the staging environment.',
      'read_logs: Shows the latest logs of a running service.',
    ].join('\n'),
    requests: [
      { text: 'where is the function parse_config defined?', intended: 'search_code' },
      { text: 'ship this build to staging', intended: 'deploy_staging' },
      { text: 'check that nothing is broken before I merge', intended: 'run_tests' },
      { text: 'order a pizza for the team', intended: 'none' },
    ],
  },
  {
    id: 'support',
    label: 'Support desk',
    tools: [
      "reset_password: Sends a password reset link to the user's email address.",
      'refund_order: Refunds a paid order to the original payment method.',
      'track_shipment: Gives the delivery status of a shipped order.',
      'update_address: Changes the shipping address of an account.',
      "cancel_subscription: Cancels the user's subscription at the end of the billing period.",
      'talk_to_agent: Transfers the conversation to a human support agent.',
      'apply_coupon: Applies a discount code to the current cart.',
    ].join('\n'),
    requests: [
      { text: "my package still hasn't arrived, where is it?", intended: 'track_shipment' },
      { text: 'I forgot my password and cannot log in', intended: 'reset_password' },
      { text: 'I want my money back for order 4417', intended: 'refund_order' },
      { text: 'can you recommend a good book for the holidays?', intended: 'none' },
    ],
  },
]

/** The page's published numbers (runs/s6-nano3/eval-fresh/results.md, gate.json). */
export const ROUTE_RESULT = {
  system: 0.8868,
  bm25: 0.8169,
  minilm: 0.9048,
  masked: 0.2497,
  bytes: 3692224,
  wasmMedianMs: 30.9,
} as const
