import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { defineConfig, type HeadConfig } from 'vitepress'

const SITE_URL = 'https://maxgfr.github.io/skycmd/'
const SITE_DESCRIPTION =
  'Tiny decision models trained on one laptop and run in your browser through WebAssembly, with pre-registered gates and every result published, failures included.'

/** The page's URL on the deployed site (cleanUrls is off: chapters end in .html). */
function pageUrl(relativePath: string): string {
  return SITE_URL + relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
}

/** VitePress writes the description into `<meta name="description">` without escaping quotes. */
function safeQuotes(text: string): string {
  return text.replace(/"([^"]*)"/g, '\u201c$1\u201d').replace(/"/g, '\u201d')
}

/** A page without a `description` in its front matter gets its first paragraph, as plain text. */
function firstParagraph(source: string): string {
  const body = source
    .replace(/^---\n[\s\S]*?\n---\n/, '')
    .replace(/<script[\s\S]*?<\/script>/g, '')
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/```[\s\S]*?```/g, '')
  for (const block of body.split(/\n\s*\n/)) {
    const b = block.trim()
    if (!b || /^(#|<|:::|\||!\[|- |\d+\. |>)/.test(b)) continue
    const text = b
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/<[^>]+>/g, '')
      .replace(/[`*_]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    if (text.length < 40 || text.endsWith(':')) continue
    if (text.length <= 200) return text
    const cut = text.slice(0, 200)
    return cut.slice(0, cut.lastIndexOf(' ')).replace(/[,;:]$/, '') + '…'
  }
  return SITE_DESCRIPTION
}

const chapters: [string, string][] = [
  ['01-why-tiny-decision-models', '01 Why tiny decision models'],
  ['02-setup-apple-silicon', '02 Setup on Apple Silicon'],
  ['03-typed-decisions', '03 Typed decisions: schema, unknown and the veto rule'],
  ['04-game-cores-and-oracles', '04 Game cores and oracles'],
  ['05-hugging-face-data', '05 Pulling data from Hugging Face in Rust'],
  ['06-glm-top-up', '06 GLM top-up'],
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
  ['22-s4-generic-decisions', '22 s4: a generic decision model distilled from Qwen3.5'],
  ['23-s5-lite', '23 s5-lite: invariant by construction, tested in the open'],
  ['24-reproduce-s5-lite', '24 Reproducing s5-lite step by step'],
  ['25-hugging-face', '25 Publishing on Hugging Face'],
  ['26-s6-ear', '26 s6-ear: a decision model that listens'],
  ['27-s6-nano', '27 s6-nano: one small file for the games and for free-text questions'],
  ['28-s6-secrets-pico', '28 s6-secrets-pico: is a secret exposed in this text?'],
]

const groups: [string, number, number][] = [
  ['Foundations', 0, 4],
  ['Data', 4, 7],
  ['Model and training', 7, 12],
  ['Shipping', 12, 17],
  ['Going further', 17, 21],
  ['Decision models for any question (s4, s5)', 21, 25],
  ['The s6 family', 25, 28],
]

// Every maxgfr.github.io site shares one origin, so one localStorage: VitePress keeps the theme
// under 'vitepress-theme-appearance', a key all its sites there share. skycmd keeps its own. The
// first visit after the rename takes the old value over (left in place: other sites may use it).
const APPEARANCE_KEY = 'skycmd:appearance'
const SHARED_APPEARANCE_KEY = 'vitepress-theme-appearance'
const readAppearance = `(()=>{try{let v=localStorage.getItem('${APPEARANCE_KEY}');if(v===null){v=localStorage.getItem('${SHARED_APPEARANCE_KEY}');if(v)localStorage.setItem('${APPEARANCE_KEY}',v)}return v}catch{return null}})()`

export default defineConfig({
  base: '/skycmd/',
  appearance: { storageKey: APPEARANCE_KEY },
  // The no-flash theme script VitePress inlines in every page reads the shared key: point it at ours.
  transformHtml: (html) =>
    html.replace(/localStorage\.getItem\((['"])vitepress-theme-appearance\1\)/, readAppearance),
  lang: 'en-US',
  title: 'skycmd',
  description: SITE_DESCRIPTION,
  cleanUrls: false,
  lastUpdated: false,
  sitemap: { hostname: SITE_URL },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/skycmd/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#0ea5e9' }],
  ],
  transformPageData(pageData, { siteConfig }) {
    if (pageData.isNotFound) return
    if (pageData.frontmatter.description) {
      pageData.description = safeQuotes(pageData.frontmatter.description)
      return
    }
    try {
      pageData.description = safeQuotes(firstParagraph(readFileSync(join(siteConfig.srcDir, pageData.relativePath), 'utf8')))
    } catch {
      /* a page with no source file keeps the site's description */
    }
  },
  // Canonical URL, Open Graph and Twitter card tags, per page.
  transformHead({ pageData, title, description }) {
    if (pageData.isNotFound) return []
    const url = pageUrl(pageData.relativePath)
    const image = SITE_URL + 'og-image.png'
    const head: HeadConfig[] = [
      ['link', { rel: 'canonical', href: url }],
      ['meta', { property: 'og:type', content: pageData.relativePath === 'index.md' ? 'website' : 'article' }],
      ['meta', { property: 'og:site_name', content: 'skycmd' }],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { property: 'og:image', content: image }],
      ['meta', { property: 'og:image:width', content: '1200' }],
      ['meta', { property: 'og:image:height', content: '630' }],
      ['meta', { property: 'og:image:alt', content: 'skycmd: tiny decision models, trained on one laptop, run in your browser through WebAssembly.' }],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['meta', { name: 'twitter:image', content: image }],
    ]
    return head
  },
  themeConfig: {
    logo: { src: '/favicon.svg', alt: '' },
    // Five entries: the catalogue, the demos, the evidence, the reading and the install guide.
    // Every page keeps its URL; only the menu groups them.
    nav: [
      { text: 'Models', link: '/models/', activeMatch: '^/models/' },
      {
        text: 'Demos',
        activeMatch: '^/(play|live|router|forecast|ear|secrets-verify|memory)/',
        items: [
          { text: 'Play: four games', link: '/play/' },
          { text: 'Live: your own questions', link: '/live/' },
          { text: 'Router: pick a tool', link: '/router/' },
          { text: 'Forecast: time series', link: '/forecast/' },
          { text: 'Listen: audio', link: '/ear/' },
          { text: 'Secrets verifier (failed gate)', link: '/secrets-verify/' },
          { text: 'Memory (failed gate)', link: '/memory/' },
        ],
      },
      {
        text: 'Results',
        activeMatch: '^/(results|bench|novelty|paper)/',
        items: [
          { text: 'Results', link: '/results/' },
          { text: 'Bench', link: '/bench/' },
          { text: 'What we bring', link: '/novelty/' },
          { text: 'Technical report', link: '/paper/' },
        ],
      },
      {
        text: 'Learn',
        activeMatch: '^/(tutorial|articles|sources|idle-training)/',
        items: [
          { text: 'Tutorial', link: '/tutorial/' },
          { text: 'Articles', link: '/articles/' },
          { text: 'Train while you work', link: '/idle-training/' },
          { text: 'Sources', link: '/sources/' },
        ],
      },
      { text: 'Install', link: '/install/', activeMatch: '^/install/' },
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
