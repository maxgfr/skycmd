// A host for the secrets verifier of sky-web (s6-secrets-verify-pico) that needs neither a
// bundler nor the wasm-bindgen glue (sky_web.js): it instantiates sky_web_bg.wasm (SIMD128) or
// sky_web_relaxed_bg.wasm (relaxed SIMD) itself and calls the `SecretVerifier` exports. It runs
// in Node 21 and later, in Deno and in browsers (Web APIs only: WebAssembly, TextEncoder,
// TextDecoder, performance.now).
//
// The module's only reach out of its sandbox is the five imports below: an Error constructor, a
// string reader, a throw, a clock and a table initialiser. wasm-bindgen names them by a stem and
// a hash of the build; `checkImports` refuses a module that asks for anything else. Pin the
// .wasm by its sha256 (MANIFEST.json of the engine assets) before calling this.
//
//   import { loadVerifier } from './verify-host.mjs'
//   const v = loadVerifier(wasmBytes, modelBytes, dictionaryBytes)
//   const { p_real, state_tokens, truncated, micros } = v.judge(candidate)   // an sg-cand-1 record
//   const { tau_f1 } = v.thresholds()                                        // real when p_real >= tau_f1
//
// The candidate JSON is documented in crates/sky-web/src/verify.rs. In a browser's main thread,
// compile with `await WebAssembly.compile(bytes)` and call `verifierFromModule`.

const IMPORT_MODULE = './sky_web_bg.js'

// stem -> what the import does; a name is the stem followed by a hexadecimal hash
const STEMS = {
  __wbg_Error_: 'error',
  __wbg___wbindgen_string_get_: 'stringGet',
  __wbg___wbindgen_throw_: 'throw',
  __wbg_now_: 'now',
  __wbindgen_init_externref_table: 'initTable',
}

function roleOf(name) {
  if (name === '__wbindgen_init_externref_table') return 'initTable'
  for (const [stem, role] of Object.entries(STEMS)) {
    if (stem.endsWith('_') && name.startsWith(stem) && /^[0-9a-f]{8,}$/.test(name.slice(stem.length))) return role
  }
  return null
}

/** The import name of each role; throws unless the module asks for exactly the five imports. */
export function checkImports(module) {
  const names = {}
  for (const i of WebAssembly.Module.imports(module)) {
    const role = i.module === IMPORT_MODULE && i.kind === 'function' ? roleOf(i.name) : null
    if (role === null || names[role]) throw new Error(`the WebAssembly module asks for an unexpected import: ${i.module} ${i.name}`)
    names[role] = i.name
  }
  if (Object.keys(names).length !== 5) throw new Error('the WebAssembly module is not a sky-web build')
  return names
}

/** A verifier from compiled sky-web module, the model's GGUF bytes and the word list's bytes. */
export function verifierFromModule(module, modelBytes, wordsBytes) {
  const names = checkImports(module)
  let wasm
  const decoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true })
  const encoder = new TextEncoder()
  // the memory may grow on any allocation: never keep a view of it
  const bytes = () => new Uint8Array(wasm.memory.buffer)
  const readString = (ptr, len) => decoder.decode(bytes().subarray(ptr >>> 0, (ptr >>> 0) + len))
  // the callee takes ownership of what is passed and frees it
  const pass = (data) => {
    const ptr = wasm.__wbindgen_malloc(data.length, 1) >>> 0
    bytes().set(data, ptr)
    return [ptr, data.length]
  }
  const passString = (s) => pass(encoder.encode(s))
  const takeRef = (idx) => {
    const value = wasm.__wbindgen_externrefs.get(idx)
    wasm.__externref_table_dealloc(idx)
    return value
  }
  const asError = (v) => (v instanceof Error ? v : new Error(String(v)))
  // a returned string is copied out, then freed
  const takeString = (ptr, len) => {
    try {
      return readString(ptr, len)
    } finally {
      wasm.__wbindgen_free(ptr, len, 1)
    }
  }
  const imports = {
    [IMPORT_MODULE]: {
      [names.error]: (ptr, len) => new Error(readString(ptr, len)),
      [names.stringGet]: (retptr, obj) => {
        const [ptr, len] = typeof obj === 'string' ? passString(obj) : [0, 0]
        const view = new DataView(wasm.memory.buffer)
        view.setInt32(retptr + 4, len, true)
        view.setInt32(retptr, ptr, true)
      },
      [names.throw]: (ptr, len) => {
        throw new Error(readString(ptr, len))
      },
      [names.now]: () => performance.now(),
      [names.initTable]: () => {
        const table = wasm.__wbindgen_externrefs
        const offset = table.grow(4)
        table.set(0, undefined)
        table.set(offset, undefined)
        table.set(offset + 1, null)
        table.set(offset + 2, true)
        table.set(offset + 3, false)
      },
    },
  }
  wasm = new WebAssembly.Instance(module, imports).exports
  wasm.__wbindgen_start()

  const created = wasm.secretverifier_new(...pass(modelBytes), ...pass(wordsBytes))
  if (created[2]) throw asError(takeRef(created[1]))
  let self = created[0]
  const live = () => {
    if (!self) throw new Error('the verifier was freed')
    return self
  }
  const asJson = (x) => (typeof x === 'string' ? x : JSON.stringify(x))
  // a method returning Result<String, JsError>: [ptr, len, error ref, is error]
  const call = (fn, arg) => {
    const ret = fn(live(), ...passString(asJson(arg)))
    if (ret[3]) throw asError(takeRef(ret[2]))
    return takeString(ret[0], ret[1])
  }
  return {
    /** One candidate (object or JSON text) -> {p_real, state_tokens, truncated, micros}. */
    judge: (candidate) => JSON.parse(call(wasm.secretverifier_judge, candidate)),
    /** An array of candidates (or its JSON text) -> an array of answers, in order. */
    judgeMany: (candidates) => JSON.parse(call(wasm.secretverifier_judgeMany, candidates)),
    /** The state text the model reads for a candidate. */
    state: (candidate) => call(wasm.secretverifier_state, candidate),
    /** {tau_f1, tau_f05, tau_f2}: the operating thresholds stored in the model file. */
    thresholds: () => JSON.parse(takeString(...wasm.secretverifier_thresholds(live()))),
    /** {name, n_params, file_bytes, window, words}. */
    info: () => JSON.parse(takeString(...wasm.secretverifier_info(live()))),
    /** Releases the model and the word list (the instance's memory stays allocated). */
    free: () => {
      if (self) wasm.__wbg_secretverifier_free(self, 0)
      self = 0
    },
  }
}

/** Synchronous compile and load (Node, workers). */
export function loadVerifier(wasmBytes, modelBytes, wordsBytes) {
  return verifierFromModule(new WebAssembly.Module(wasmBytes), modelBytes, wordsBytes)
}
