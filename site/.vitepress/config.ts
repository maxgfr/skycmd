import { defineConfig } from 'vitepress'

const chapters: [string, string][] = [
  ['01-why-tiny-decision-models', '01 Why tiny decision models'],
  ['02-setup-apple-silicon', '02 Setup on Apple Silicon'],
  ['03-typed-decisions', '03 Typed decisions: schema, unknown and the veto rule'],
  ['04-game-cores-and-oracles', '04 Game cores and oracles'],
  ['05-hugging-face-data', '05 Pulling data from Hugging Face in Rust'],
  ['06-glm-top-up', '06 GLM top-up via z.ai'],
  ['07-tokenizer', '07 A tokenizer with small vocabularies'],
  ['08-encoder-and-decision-head', '08 The encoder and the decision head'],
  ['09-muon-wsd-training-loop', '09 Muon, WSD and the mlx-rs training loop'],
  ['10-pretraining-and-decision-training', '10 Pretraining and decision training'],
  ['11-calibration-and-evaluation', '11 Calibration and evaluation'],
  ['12-size-ladder', '12 The size ladder'],
  ['13-gguf-export-int8', '13 GGUF export and int8 quantization'],
  ['14-wasm-inference-simd128', '14 A WASM inference engine with SIMD128'],
  ['15-wiring-the-games', '15 Wiring the games'],
  ['16-llama-cpp-and-wllama', '16 Bonus: llama.cpp /v1/systemone and wllama'],
  ['17-scaling-and-troubleshooting', '17 Scaling up and troubleshooting'],
  ['18-doom', '18 A Doom-like shooter'],
  ['19-s2-architecture', '19 The s2 architecture'],
  ['20-dagger-lessons', '20 What DAgger taught us'],
  ['21-s3-recursive-tiny', '21 s3: a 251 KB model that plays four games'],
]

const groups: [string, number, number][] = [
  ['Foundations', 0, 4],
  ['Data', 4, 7],
  ['Model and training', 7, 12],
  ['Shipping', 12, 17],
  ['Going further', 17, 21],
]

export default defineConfig({
  base: '/skycmd/',
  lang: 'en-US',
  title: 'skycmd',
  description:
    'Tiny System One decision models, trained from scratch in Rust and playing games live in your browser through WASM.',
  cleanUrls: false,
  lastUpdated: false,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/skycmd/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#0ea5e9' }],
    ['meta', { property: 'og:title', content: 'skycmd: tiny decision models in the browser' }],
    [
      'meta',
      {
        property: 'og:description',
        content: 'A bidirectional encoder with a typed decision head, scoring options in one forward pass.',
      },
    ],
  ],
  themeConfig: {
    logo: { src: '/favicon.svg', alt: '' },
    nav: [
      { text: 'Play', link: '/play/' },
      { text: 'Tutorial', link: '/tutorial/' },
      { text: 'Bench', link: '/bench/' },
      { text: 'Install', link: '/install/' },
      { text: 'What\'s new', link: '/novelty/' },
      { text: 'Sources', link: '/sources/' },
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/maxgfr/skycmd' }],
    sidebar: {
      '/tutorial/': [
        { text: 'Tutorial', items: [{ text: 'Overview', link: '/tutorial/' }] },
        ...groups.map(([text, a, b]) => ({
          text,
          collapsed: false,
          items: chapters.slice(a, b).map(([slug, title]) => ({ text: title, link: `/tutorial/${slug}` })),
        })),
      ],
    },
    outline: { level: [2, 3] },
    search: { provider: 'local' },
    footer: {
      message: 'Apache-2.0. Civil use only. No trackers, no cookies: scores stay in your browser.',
      copyright: 'skycmd',
    },
    docFooter: { prev: 'Previous', next: 'Next' },
  },
  vite: {
    build: { target: 'es2022', chunkSizeWarningLimit: 900 },
  },
})
