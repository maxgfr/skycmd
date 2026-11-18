// The models of one Space inside the global Space (site/space/HubView.vue), which ships the full
// manifest: while a model's demo is open there, every reader of the manifest (lib/wasm.ts) and of
// forecast.json (lib/forecast.ts) sees only the models that model's own Space lists, so that the
// demo shows what its own Space shows. `universe` holds every model the global Space lists: a name
// outside it (TinyCast, a reference forecaster) is never left out. No scope, the default: the site
// and the single-model Spaces read every model. Pure, tested by tests/space.test.ts.

let scope: { names: Set<string>; universe: Set<string> } | null = null

/** Restricts the manifest readers to `names`; null lifts the restriction. */
export function setManifestScope(names: string[] | null, universe: string[] = []): void {
  scope = names ? { names: new Set(names), universe: new Set(universe) } : null
}

/** The test of the scope in force now, or null when there is none. A reader takes it when it is
 * called and applies it once its file has arrived, so a demo being unmounted keeps its own. */
export function scopeFilter(): ((name: string) => boolean) | null {
  const s = scope
  return s ? (name) => s.names.has(name) || !s.universe.has(name) : null
}
