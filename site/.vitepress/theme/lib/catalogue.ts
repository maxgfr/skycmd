// The Models catalogue: one entry per model family, grouped by what a visitor can do with it.
// It drives /models/ and the "What works today" strip of the home page.
//
// Update it with every new result, like site/novelty/index.md ("What we bring"). Every number
// here is copied from site/results/index.md, site/novelty/index.md, hf/cards/*.md or
// site/public/models/manifest.json, and the `source` of each entry says which; never round,
// change or add one. tests/catalogue.test.ts checks the sizes against the manifest, that every
// model of the manifest has an entry, that every "Get it" link has a card in hf/cards/ and that
// every page linked exists.

export type Status = 'passed' | 'second-look' | 'ungated' | 'bench-only' | 'part' | 'failed' | 'progress'

export type GroupId = 'works' | 'part' | 'results' | 'progress'

export interface Link {
  /** A path of this site (without the base), or an absolute URL. */
  link: string
  /** What the link opens, read by screen readers after the action ("Try it: Router page"). */
  label: string
}

export interface CatalogueEntry {
  /** The card's anchor on /models/ and the model's name. */
  id: string
  name: string
  group: GroupId
  status: Status
  /** Why the status is what it is, in one sentence, when the badge alone would mislead. */
  statusNote?: string
  /** One plain sentence: what the model is for. */
  purpose: string
  /** As the site writes it (decimal units); checked against `manifest` bytes by the test. */
  size?: string
  /** The manifest entries this card stands for; the first one gives the size. */
  manifest?: string[]
  /** The single most useful fact or lesson, with its number and what it was measured on. */
  brings: string
  /** The limit a visitor must know before using it. */
  caveat?: string
  /** On the home page: the key number of the "What works today" strip. */
  home?: string
  /** On the home page: one of the lessons of "What failed, and what it taught". */
  lesson?: string
  /** The subject of `lesson` when it is not the whole card (one model of a pair). */
  lessonName?: string
  try?: Link
  /** The Hugging Face model page, when hf/cards/<name>.md exists. */
  get?: string
  read?: Link
  /** Where the numbers of this entry come from (for maintainers; not shown). */
  source: string[]
}

export const STATUS: Record<Status, { label: string; tone: 'success' | 'neutral' | 'warning' | 'danger' | 'info'; meaning: string }> = {
  passed: {
    label: 'Passed',
    tone: 'success',
    meaning: 'It cleared a gate written down before it was trained and tested.',
  },
  'second-look': {
    label: 'Passed on a second look',
    tone: 'success',
    meaning: 'It cleared its gate, but it was designed after a first version failed on the same test records.',
  },
  ungated: {
    label: 'No gate, measured',
    tone: 'neutral',
    meaning: 'Its numbers are measured and published, but no gate was fixed for this file before it was tested.',
  },
  'bench-only': {
    label: 'Passed on its bench only',
    tone: 'warning',
    meaning: 'It cleared its gate on generated data and fails on real data, measured after publication.',
  },
  part: {
    label: 'Failed gate, usable as a part',
    tone: 'warning',
    meaning: 'It missed its gate; what it does measure makes it worth using inside another tool.',
  },
  failed: {
    label: 'Failed gate, a result',
    tone: 'danger',
    meaning: 'It missed its gate. The file is published as the result of its experiment, not for use.',
  },
  progress: {
    label: 'In progress',
    tone: 'info',
    meaning: 'Pre-registered, with no test result yet.',
  },
}

export const GROUPS: Record<GroupId, { title: string; verb: string }> = {
  works: { title: 'Works: use it', verb: 'What it brings' },
  part: { title: 'Useful as a part, with a failed gate', verb: 'What it brings' },
  results: { title: 'Published as results, not for use', verb: 'What it taught' },
  progress: { title: 'In progress', verb: 'Being tested' },
}

const HF = 'https://huggingface.co/maxgfr/'
const DOCS = 'https://github.com/maxgfr/skycmd/blob/main/docs/'

export const CATALOGUE: CatalogueEntry[] = [
  // Works: use it
  {
    id: 's6-route',
    name: 's6-route',
    group: 'works',
    status: 'passed',
    statusNote: "Passed on fresh requests; the system was defined after its predecessor's test was read.",
    purpose: 'Picks the tool that fits a request from a list of tools you write at run time, or says that none fits.',
    size: '3.7 MB',
    manifest: ['s6-route'],
    brings:
      'The right option on 0.8868 of 945 fresh requests about tools never seen in training, against 0.8169 for BM25 and 0.9048 for all-MiniLM-L6-v2 (91.6 MB), in 30.9 ms per request in WASM.',
    caveat: 'It needs distinctive tool names: with every tool renamed it falls to 0.2497.',
    home: '0.8868 on tools it never saw; BM25 0.8169',
    try: { link: '/router/', label: 'the Router page' },
    get: HF + 's6-route',
    read: { link: '/articles/a-3-7-mb-tool-router', label: 'the article on the tool router' },
    source: ['site/results/index.md#s6-route-passed', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-micro',
    name: 's6-micro',
    group: 'works',
    status: 'second-look',
    statusNote: 'Its first version failed its calibration guard; this one was designed after that reading and judged on the same test records.',
    purpose: 'Plays four browser games and answers typed questions written in free text, in one file.',
    size: '2.4 MB',
    manifest: ['s6-micro'],
    brings:
      "0.553 on typed_decisions_bench (gate 0.45) with an ECE of 0.0588 (guard 0.08), and every game at s3-pico's score.",
    caveat: 'Expect mistakes on free-text questions: 0.401 on When2Call and 0.092 on NOTAM subjects.',
    try: { link: '/live/#s6-nano-the-questions-it-was-trained-for', label: 'the Live page, where its menu lists s6-micro' },
    get: HF + 's6-micro',
    read: { link: '/results/#s6-micro-v2-passed-on-a-second-look', label: 'its section of Results' },
    source: ['site/results/index.md#s6-micro-v2-passed-on-a-second-look', 'hf/spaces.json', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-nano',
    name: 's6-nano',
    group: 'works',
    status: 'passed',
    purpose: 'Plays the four browser games and answers typed questions written in free text, in under 1 MB.',
    size: '715 KB',
    manifest: ['s6-nano'],
    brings:
      "The four games at the oracle's level, and 0.480 on typed_decisions_bench, above the majority class (0.398) where s3-pico (0.382) stayed below.",
    caveat:
      'A constant answer per task passes its gate too: answering each question with its most common training answer scores 0.4815.',
    home: "The four games at the oracle's level, and 0.480 on typed questions",
    try: { link: '/live/#s6-nano-the-questions-it-was-trained-for', label: 'the Live page' },
    get: HF + 's6-nano',
    read: { link: '/results/#s6-nano-passed-published-at-full-depth', label: 'its section of Results' },
    source: ['site/results/index.md#s6-nano-passed-published-at-full-depth', 'site/public/models/manifest.json'],
  },
  {
    id: 's3-pico',
    name: 's3-pico',
    group: 'works',
    status: 'ungated',
    statusNote: 'Its run had no pre-registered gate; the scores are 30 seeded episodes per game with the shipped file.',
    purpose: 'Plays four browser games (Flappy Drone, Rescue Run, Obstacle Dodge and a Doom-like) from the game state.',
    size: '251 KB',
    manifest: ['s3-pico'],
    brings:
      "The four games at the oracle's level (Flappy 200/200 pipes, Dodge 87%, the oracle's rate), at 49.6 µs per Flappy frame in WASM; trained from scratch in about 47 minutes on an Apple M5.",
    home: "Four games at the oracle's level, 49.6 µs per frame",
    try: { link: '/play/', label: 'the Play page' },
    get: HF + 's3-pico',
    read: { link: '/tutorial/21-s3-recursive-tiny', label: 'chapter 21 of the tutorial' },
    source: ['site/novelty/index.md (section 1)', 'hf/cards/s3-pico.md', 'site/public/models/manifest.json'],
  },
  {
    id: 's5-lite',
    name: 's5-lite',
    group: 'works',
    status: 'ungated',
    statusNote:
      'The pre-registered s5 bets failed; this checkpoint is a longer run added for the demo outside them, so its numbers are reported, not claimed.',
    purpose: 'Answers questions you write yourself, in English, with each option defined in plain words.',
    size: '20.7 MB',
    manifest: ['s5-lite'],
    brings:
      'Reordering the options or renaming their keys cannot change an answer, by construction (exact tests to 1e-6, natively and in WASM); 0.696 on the typed-decisions test, where oscar-1-17m scores 0.682 on the same records.',
    caveat: '26% of its answers flip when the definitions are paraphrased, and four test sets fall below their majority class.',
    home: 'Option order and key names cannot change its answer',
    try: { link: '/live/', label: 'the Live page' },
    get: HF + 's5-lite',
    read: { link: '/tutorial/23-s5-lite', label: 'chapter 23 of the tutorial' },
    source: ['site/results/index.md#s5-october-2026', 'hf/cards/s5-lite.md', 'site/public/models/manifest.json'],
  },

  // Useful as a part, with a failed gate
  {
    id: 's6-secrets-verify-pico',
    name: 's6-secrets-verify-pico',
    group: 'part',
    status: 'part',
    statusNote: 'It missed one gate of seven (G1) by 0.027; the six others hold.',
    purpose:
      'Judges whether a value found by a secret-scanning rule is a real credential, from the value, its name and the lines around it.',
    size: '250 KB',
    manifest: ['s6-secrets-verify-pico'],
    brings:
      'Line F1 0.8561 on 221 real test repositories of CredData, above every rule scanner tested (the best, Betterleaks, 0.7332), with 0.35% of clean real files flagged; its gate asked for 0.8832.',
    caveat:
      'secretgate 1.9.0 offers it as an experimental option, off by default, where it can only add redactions. It has no demo: its input comes from secretgate\'s rules, which the browser does not run.',
    get: HF + 's6-secrets-verify-pico',
    read: { link: '/articles/a-250-kb-secrets-verifier', label: 'the article on the secrets verifier' },
    source: ['site/results/index.md#s6-secrets-verify-pico-failed', 'site/articles/index.md', 'site/public/models/manifest.json'],
  },

  // Published as results, not for use
  {
    id: 's6-route-nano',
    name: 's6-route-nano',
    group: 'results',
    status: 'failed',
    purpose: "Picks a tool from a list given at run time, with each tool's BM25 score in its input.",
    size: '2.4 MB',
    manifest: ['s6-route-nano'],
    brings:
      'It tied BM25, the score it reads (0.8395 against 0.8500 on 2,000 slates of tools never seen in training), at 38.9 ms per request in WASM where its gate asked for 2 ms. Its file, unchanged, is the model half of s6-route.',
    lesson: 'tied BM25, the lexical score it reads (0.8395 against 0.8500); its file became half of s6-route, which passed.',
    get: HF + 's6-route-nano',
    read: { link: '/articles/a-3-7-mb-tool-router', label: 'the article on the tool router' },
    source: ['site/results/index.md#s6-route-nano-failed', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-secrets-pico',
    name: 's6-secrets-pico',
    group: 'results',
    status: 'bench-only',
    purpose: 'Says whether a short text exposes a secret, which service issued it and what kind of value it is.',
    size: '250 KB',
    manifest: ['s6-secrets-pico'],
    brings:
      'A generated bench does not carry over to real files: it beat a rule scanner there (secret 0.9990 against 0.977), then flagged 59.8% of 719 clean real files at p ≥ 0.9.',
    lesson: 'beat a rule scanner on its generated bench (0.9990 against 0.977) and flagged 59.8% of clean real files.',
    try: { link: '/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text', label: 'the Live page, on its bench texts' },
    get: HF + 's6-secrets-pico',
    read: { link: '/articles/a-250-kb-secrets-verifier', label: 'the article on the secrets verifier' },
    source: ['site/results/index.md#s6-secrets-real-text', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-micro-v1',
    name: 's6-micro-v1',
    group: 'results',
    status: 'failed',
    purpose: 'The first version of s6-micro: four games and typed questions in free text.',
    size: '2.4 MB',
    manifest: ['s6-micro-v1'],
    brings:
      'Accuracy is not the whole gate: 0.547 on typed_decisions_bench cleared its bar of 0.45, and an ECE of 0.0821 missed the 0.08 guard. Its second version, with twice the decision-training minutes, passed on a second look.',
    try: { link: '/play/', label: 'the Play page, where it is marked as failed' },
    get: HF + 's6-micro-v1',
    read: { link: '/results/#s6-micro-v1-failed-on-calibration-only', label: 'its section of Results' },
    source: ['site/results/index.md#s6-micro-v1-failed-on-calibration-only', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-ear-nano',
    name: 's6-ear-nano',
    group: 'results',
    status: 'failed',
    purpose: 'Hears the last 10 s of audio and answers typed questions about it: is the turn over, which keyword was said.',
    size: '899 KB',
    manifest: ['s6-ear-nano'],
    brings:
      'Its pretraining on Mimi speech codes did not help: a control trained without it did better on all four test sets. End of turn 0.5681, against 0.9263 for Smart Turn v3.2 on the same clips.',
    try: { link: '/ear/', label: 'the Listen page' },
    get: HF + 's6-ear-nano',
    read: { link: '/results/#s6-ear-nano-failed', label: 'its section of Results' },
    source: ['site/results/index.md#s6-ear-nano-failed', 'site/public/models/manifest.json'],
  },
  {
    id: 's6-cast-pico',
    name: 's6-cast-pico',
    group: 'results',
    status: 'failed',
    purpose: 'Forecasts the next 64 steps of a time series as nine deciles.',
    size: '143 KB',
    manifest: ['s6-cast-pico'],
    brings:
      'nGMASE 0.9471 and nWQL 0.6646 on GIFT-Eval, against bars of 0.838 and 0.587; our port of TinyCast, a published model of 146,505 parameters, scores 0.7856 and 0.5570 on the same harness.',
    try: { link: '/forecast/', label: 'the Forecast page, where it is unticked by default' },
    get: HF + 's6-cast-pico',
    read: { link: '/results/#s6-cast-pico-failed', label: 'its section of Results' },
    source: ['site/results/index.md#s6-cast-pico-failed', 'site/public/models/manifest.json'],
  },
  {
    id: 's4-micro',
    name: 's4-micro',
    group: 'results',
    status: 'failed',
    purpose: 'Generic decisions with options in free text, distilled from Qwen3.5-4B in one night.',
    size: '2.4 MB',
    manifest: ['s4-micro'],
    brings:
      'Loops do not replace language pretraining: 0.343 on the typed-decisions bench against a gate of 0.55, below the majority class (0.398); a fine-tuned 4.4M BERT-Tiny reaches 0.794 on Banking77, against 0.303.',
    try: { link: '/live/#proof-1-keys-and-order-cannot-matter-guaranteed', label: 'proof 1 of the Live page, where it reads the keys' },
    get: HF + 's4-micro',
    read: { link: '/tutorial/22-s4-generic-decisions', label: 'chapter 22 of the tutorial' },
    source: ['site/novelty/index.md (section 7)', 'hf/cards/s4-micro.md', 'site/public/models/manifest.json'],
  },
  {
    id: 's2-nano',
    name: 's2-nano and s2-nano-doom',
    group: 'results',
    status: 'ungated',
    statusNote: 'Earlier game models, replaced by s3-pico.',
    purpose: 'Play Flappy Drone, Rescue Run and Obstacle Dodge; s2-nano-doom, its fine-tune, plays the Doom-like.',
    size: '702 KB',
    manifest: ['s2-nano', 's2-nano-doom'],
    brings:
      'Calibrate early exits by playing: at exit thresholds fitted on i.i.d. questions, s2-nano crashed in every Flappy episode; with thresholds chosen in the game it flew 191.9 of 200 pipes on average.',
    lesson: 'crashed in every Flappy episode at exit thresholds fitted on questions, and flew 191.9 of 200 pipes once they were chosen by playing.',
    lessonName: 's2-nano',
    try: { link: '/play/', label: 'the Play page' },
    get: HF + 's2-nano',
    read: { link: '/novelty/#_3-a-lesson-calibrate-early-exits-where-the-model-is-used', label: 'section 3 of What we bring' },
    source: ['site/novelty/index.md (section 3)', 'hf/cards/s2-nano.md', 'site/public/models/manifest.json'],
  },
  {
    id: 's1-pico',
    name: 's1-pico',
    group: 'results',
    status: 'ungated',
    statusNote: 'The first game model, replaced by s3-pico.',
    purpose: 'Plays three browser games in the standard System One format; the same file runs in llama.cpp and wllama.',
    size: '545 KB',
    manifest: ['s1-pico'],
    brings:
      'Labelling states is not playing: it labelled 82% of Flappy test states correctly and crashed in all 30 episodes; relative clearances in its input and two DAgger rounds took it to 200/200 pipes.',
    try: { link: '/play/', label: 'the Play page' },
    get: HF + 's1-pico',
    read: { link: '/tutorial/20-dagger-lessons', label: 'chapter 20 of the tutorial' },
    source: ['hf/cards/s1-pico.md', 'site/public/models/manifest.json'],
  },

  // In progress: no numbers before a result, except the pilot reading of s6-sleep v3
  {
    id: 's6-ear-v2',
    name: 's6-ear v2',
    group: 'progress',
    status: 'progress',
    purpose: 'Detects the end of a speaker\'s turn for a voice agent, from the last seconds of audio.',
    brings:
      'Whether a small ternary student distilled from Smart Turn v3.2, with pitch and energy in its input and trained on this one task, reaches a fixed accuracy bar on the test clips where s6-ear-nano failed.',
    read: { link: DOCS + 's6-ear2-prereg.md', label: 'its pre-registration' },
    source: ['docs/s6-ear2-prereg.md'],
  },
  {
    id: 's6-sleep-v3',
    name: 's6-sleep v3',
    group: 'progress',
    status: 'progress',
    purpose: 'A fast-weight memory that lets a decision model carry a fact across frames.',
    brings:
      'Whether the memory helps a nano model on unseen episodes. Its pilot (pico size, development seeds, no test seed read) says the memory works: recall 1.0000 with the memory, 0.2542 for the control without one and 0.2567 with the memory zeroed. The nano runs come next.',
    read: { link: DOCS + 's6-sleep3-prereg.md', label: 'its pre-registration and amendment 2' },
    source: ['docs/s6-sleep3-prereg.md (amendment 2)'],
  },
  {
    id: 's6-cast-v2',
    name: 's6-cast v2',
    group: 'progress',
    status: 'progress',
    purpose: 'Forecasts time series, as s6-cast-pico tried to.',
    brings:
      "Whether the same encoder becomes a usable zero-shot forecaster with a computed-periodicity front end, training data that covers GIFT-Eval's frequency bands and more training time. Not run yet.",
    read: { link: DOCS + 's6-cast2-prereg.md', label: 'its pre-registration' },
    source: ['docs/s6-cast2-prereg.md', 'site/results/index.md (In short)'],
  },
]

export function entriesOf(group: GroupId): CatalogueEntry[] {
  return CATALOGUE.filter((e) => e.group === group)
}

export function isExternal(link: string): boolean {
  return /^https?:\/\//.test(link)
}
