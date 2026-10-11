// skycmd: tiny System One decision models (GGUF) in the browser, Node >= 18, Deno and Bun.
//
//   import SkyCmd from "skycmd";
//   const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s1-pico.gguf");
//   const r = model.decide({ state: "alt 12 vy -3", questions: { ... } }); // POST /v1/systemone body
//
// No bundler needed: the WebAssembly engine (sky_infer_bg.wasm) is loaded next to this file.

import initWasm, { SkyEngine } from "./sky_infer.js";

const WASM_FILE = "sky_infer_bg.wasm";
const QTYPES = ["choice", "score", "noul"];

let ready = null;

/** True for `file:` URLs. */
function isFileUrl(u) {
  return u instanceof URL && u.protocol === "file:";
}

/** `node:fs/promises` without letting bundlers try to resolve it for the browser. */
async function readLocal(pathOrUrl) {
  const spec = "node:fs/promises";
  const fs = await import(/* @vite-ignore */ /* webpackIgnore: true */ spec);
  return new Uint8Array(await fs.readFile(pathOrUrl));
}

/** Resolve a string source to a URL, or keep it as a local path (Node, Deno, Bun). */
function resolveSource(s) {
  if (/^[a-z][a-z0-9+.-]+:/i.test(s)) return new URL(s); // has a scheme (not a Windows drive)
  const base = globalThis.location?.href;
  if (base) return new URL(s, base);
  return s; // relative or absolute file path
}

async function fetchBytes(url) {
  if (isFileUrl(url)) return readLocal(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`skycmd: cannot fetch ${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** Turn any supported source into bytes. */
async function toBytes(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    if (!source.ok) throw new Error(`skycmd: HTTP ${source.status} for ${source.url}`);
    return new Uint8Array(await source.arrayBuffer());
  }
  if (typeof Blob !== "undefined" && source instanceof Blob) {
    return new Uint8Array(await source.arrayBuffer());
  }
  if (source instanceof URL) return fetchBytes(source);
  if (typeof source === "string") {
    const r = resolveSource(source);
    return typeof r === "string" ? readLocal(r) : fetchBytes(r);
  }
  throw new TypeError(
    "skycmd: expected a URL, a path, a Response, a Blob, an ArrayBuffer or a Uint8Array",
  );
}

/**
 * Load the WebAssembly engine once. Called by `SkyCmd.load`; call it yourself only to point at a
 * wasm file served from another place (`init(url | bytes | WebAssembly.Module)`).
 */
export function init(wasm) {
  if (!ready) {
    ready = (async () => {
      let input = wasm;
      if (input === undefined) {
        const url = new URL(WASM_FILE, import.meta.url);
        // Node cannot fetch file: URLs; Deno and Bun can, but reading the file works everywhere.
        input = isFileUrl(url) ? await readLocal(url) : url;
      } else if (typeof input === "string" || input instanceof URL) {
        input = await toBytes(input);
      }
      await initWasm({ module_or_path: input });
    })();
    ready.catch(() => {
      ready = null; // allow a retry after a failed download
    });
  }
  return ready;
}

/** A loaded decision model (GGUF: s1 `modern-bert` or s2 `skycmd-s2`). */
export class SkyCmd {
  #engine;
  #info = null;

  /** Use `await SkyCmd.load(...)`. */
  constructor(engine) {
    if (!(engine instanceof SkyEngine)) {
      throw new TypeError("skycmd: use `await SkyCmd.load(urlOrBytes)`");
    }
    this.#engine = engine;
  }

  /**
   * Load a model from a URL, a path (Node, Deno, Bun), a `Response`, a `Blob` or bytes.
   * `options.q8` computes with int8 weights (smaller memory, slightly different numbers);
   * `options.wasm` overrides where the engine's .wasm file comes from.
   */
  static async load(source, options = {}) {
    const [, bytes] = await Promise.all([init(options.wasm), toBytes(source)]);
    const engine = options.q8
      ? SkyEngine.withOptions(bytes, true, !!options.fast)
      : new SkyEngine(bytes);
    return new SkyCmd(engine);
  }

  /**
   * Answer a System One request, the body of `POST /v1/systemone`:
   * `{ state, questions: { id: { type, instructions, criteria? } } }`. Returns
   * `{ model, answers, usage }` (an object), like the HTTP server.
   */
  decide(request) {
    const text = typeof request === "string" ? request : JSON.stringify(request);
    return JSON.parse(this.#live().decide_json(text));
  }

  /** Same as `decide` with JSON text in and out. */
  decideJson(request) {
    return this.#live().decide_json(request);
  }

  /**
   * Probabilities (after the temperature) of already rendered option texts, in order. The
   * question and option tokens are cached, so a game loop that changes only `state` is fast.
   * `type` is "choice", "score" or "noul"; noul options are `["false: ...", "true: ..."]`.
   */
  decideRaw(type, instructions, options, state) {
    checkType(type);
    return Array.from(this.#live().decide(type, instructions, Array.from(options), state));
  }

  /** Raw scores (before the temperature), same arguments as `decideRaw`. */
  scores(type, instructions, options, state) {
    checkType(type);
    return Array.from(this.#live().scores(type, instructions, Array.from(options), state));
  }

  /** Model description: name, sizes, temperatures. */
  get info() {
    this.#info ??= JSON.parse(this.#live().model_info());
    return this.#info;
  }

  /** Model name (`general.name`). */
  get name() {
    return this.info.name;
  }

  /** Number of parameters. */
  get nParams() {
    return this.#live().n_params();
  }

  /** Free the WebAssembly memory of this model. The object cannot be used afterwards. */
  free() {
    this.#engine?.free();
    this.#engine = null;
  }

  #live() {
    if (!this.#engine) throw new Error("skycmd: this model was freed");
    return this.#engine;
  }
}

// `using model = await SkyCmd.load(...)` where explicit resource management exists.
if (typeof Symbol.dispose === "symbol") {
  SkyCmd.prototype[Symbol.dispose] = function () {
    this.free();
  };
}

function checkType(type) {
  if (!QTYPES.includes(type)) {
    throw new TypeError(`skycmd: unknown question type ${JSON.stringify(type)} (choice, score, noul)`);
  }
}

/** Same as `SkyCmd.load`. */
export const load = (source, options) => SkyCmd.load(source, options);

export default SkyCmd;
