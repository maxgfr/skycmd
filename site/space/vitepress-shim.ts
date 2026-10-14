// Stand-in for the one VitePress helper the site components use, so that they run in a plain
// static page (a Hugging Face Space). Files shipped with the Space (wasm module, models, game
// textures, results) resolve next to the page; every other link points at the project site.
const SITE = 'https://maxgfr.github.io/skycmd'
const LOCAL = ['/pkg/', '/models/', '/doom/', '/results/s5.json']

export function withBase(path: string): string {
  // Absolute, from the page: a dynamic `import()` of a relative path would resolve against the
  // bundled module in assets/, not against the page.
  if (LOCAL.some((p) => path.startsWith(p))) return new URL(`.${path}`, document.baseURI).href
  return `${SITE}${path}`
}
