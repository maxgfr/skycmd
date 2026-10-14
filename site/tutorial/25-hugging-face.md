# 25 Publishing on Hugging Face

Every model of this site is also on [Hugging Face](https://huggingface.co/maxgfr). Each one gets:

- **a model repository:** the card and the GGUF file;
- **a static Space:** a demo that runs entirely in the visitor's browser, with no server and no key.

This chapter shows how both are built and published from the repository.

## Model cards

The cards are in `hf/cards/<model>.md`, in English. Each one has the YAML header Hugging Face reads (license, language, tags, base model) and the same sections:

- what the model is and what is different about it;
- the results, failures included, with their sources;
- the limitations;
- usage with the skycmd engine;
- the training data and their licenses;
- the files.

`{{USER}}` stands for the namespace and is replaced at publish time.

## One demo app for every Space

The demos of this site are Vue components:

- `LiveDecide` and `LiveProof` for decisions;
- `PlayGame` for the games.

`site/space/` reuses them in a plain static page built with Vite. They need only one thing from VitePress, `withBase`, and a shim replaces it:

`site/space/vitepress-shim.ts`

```ts
const SITE = 'https://maxgfr.github.io/skycmd'
const LOCAL = ['/pkg/', '/models/', '/doom/', '/results/s5.json']

export function withBase(path: string): string {
  // Absolute, from the page: a dynamic `import()` of a relative path would resolve against the
  // bundled module in assets/, not against the page.
  if (LOCAL.some((p) => path.startsWith(p))) return new URL(`.${path}`, document.baseURI).href
  return `${SITE}${path}`
}
```

The comment records the one bug of the first try. `./pkg/sky_web.js` passed to `import()` resolves against the script, not the page, and every model failed to load with a 404.

Each Space ships the same build with its own `space.json`, which names the model, the demos and the game to open first. Two builds are needed:

```sh
scripts/build-wasm.sh                 # the Rust engine as wasm32 + SIMD128, into site/public/pkg
cd site && npm run space:build        # the demo app, into target/hf-space-app
```

## Publish

```sh
pip install -U huggingface_hub && hf auth login   # in your own terminal: a token with write access
python3 scripts/hf-publish.py --dry-run          # assembles target/hf/<model>/{model,space}
python3 scripts/hf-publish.py                    # creates or updates the repos and Spaces
python3 scripts/hf-publish.py --only s5-lite     # one model
```

The script assembles everything under `target/hf/` first, so you can inspect it. For each entry of `hf/spaces.json`:

- **the model repository** gets the card and `<model>.gguf`;
- **the Space** gets `README.md` (`sdk: static`), the demo app, the wasm module, `models/` with the GGUF files and a manifest, the Freedoom textures for the games, and `space.json`;
- **both** are created if missing, then uploaded with `huggingface_hub`.

The s5-lite Space also carries s4-micro, the model that reads the keys, for the "keys and order" proof.

Keep the token in your own terminal, never in a chat or a commit. `hf auth login` stores it in `~/.cache/huggingface/token`, and the script reads it from there without printing it.

## Check

A static Space is served at `https://<user>-<model>.static.hf.space/`. Open each one in a real browser (or a headless Chrome) and check three things:

- the model loads;
- the demos answer;
- the console has no error.
