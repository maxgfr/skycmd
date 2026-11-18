// The `space.json` of a Space (scripts/hf-publish.py writes it) and the pure parts of the global
// Space, `maxgfr/skycmd` (HubView.vue): its list of models grouped as the site's Models catalogue
// groups them, and the model a URL hash opens. Pure: no import but types, tested by
// tests/space.test.ts.
import type { CatalogueEntry, GroupId } from '../.vitepress/theme/lib/catalogue'

/** The `space.json` of one model's Space (`space_config` in scripts/hf-publish.py). */
export interface SpaceConfig {
  model: string
  title: string
  tagline: string
  /** `secrets`: the API-key demo of s6-secrets-pico (manifest `task: secrets`). `ear`: the Listen
   * demo (manifest `task: ear`). `forecast`: the Forecast demo (`models/forecast.json`). `router`:
   * the tool router of the Router page (manifest `task: router`, two files fused by `ToolRouter`).
   * `turn`: the end-of-turn demo of the Listen page (s6-ear2-nano through `TurnDetector`).
   * `router-v2`: the same router demo showing the s6-route-nano file alone (it failed its gate; the
   * Space also carries s6-route's two files, which `ToolRouter` needs). `secrets-verify`: the
   * Secrets verifier demo (s6-secrets-verify-pico through `SecretVerifier`, with its word list).
   * `memory`: the Memory demo (s6-sleep3-nano and its control, two files of one manifest entry
   * `task: memory`, through `RecallProbe`; it failed its gate). */
  kind: 'decision' | 'game' | 'secrets' | 'secrets-verify' | 'ear' | 'forecast' | 'router' | 'router-v2' | 'turn' | 'memory'
  proofs?: boolean
  model_repo: string
  notes?: string[]
  /** Game shown first (game Spaces): flappy, rescue, dodge or doom. */
  game?: string | null
  /** The model's pre-registered gate (from the manifest). The Space of a model with a gate opens
   * with a notice that gives its verdict, passed or failed, with the numbers and the link to the
   * result; a model published without a gate has none. */
  gate?: 'passed' | 'failed' | null
  /** `fixed-before-training` when no pre-registration document exists (lib/gate.ts). */
  gate_kind?: 'pre-registered' | 'fixed-before-training' | null
  gate_note?: string | null
  gate_url?: string | null
}

/** One model's Space inside the global one: its own `space.json`, the models its own Space lists
 * (the model and its `extra_models`) and the address of that Space. */
export interface HubSpace {
  config: SpaceConfig
  models: string[]
  space_url: string
}

/** The `space.json` of the global Space (kind `hub`). */
export interface HubConfig {
  kind: 'hub'
  title: string
  tagline: string
  /** Counted from the manifest when the Space is assembled. */
  notes?: string[]
  spaces: HubSpace[]
}

/** One model of the list: its Space and the catalogue card that describes it. */
export interface HubItem {
  name: string
  space: HubSpace
  card: CatalogueEntry
}

export interface HubGroup {
  id: GroupId
  title: string
  items: HubItem[]
}

/** The models of the global Space under the groups of the Models catalogue, in the catalogue's
 * order (a card that stands for two models, s2-nano and s2-nano-doom, gives both in its manifest
 * order). A Space whose model has no card is left out; empty groups are left out. */
export function hubGroups(
  spaces: HubSpace[],
  catalogue: CatalogueEntry[],
  groups: Record<GroupId, { title: string }>,
): HubGroup[] {
  const byModel = new Map(spaces.map((s) => [s.config.model, s]))
  return (Object.keys(groups) as GroupId[])
    .map((id) => ({
      id,
      title: groups[id].title,
      items: catalogue
        .filter((card) => card.group === id)
        .flatMap((card) =>
          (card.manifest ?? []).flatMap((name) => {
            const space = byModel.get(name)
            return space ? [{ name, space, card }] : []
          }),
        ),
    }))
    .filter((g) => g.items.length > 0)
}

/** The hash that opens a model's demo: `#s6-route`. */
export function hashOf(space: HubSpace): string {
  return `#${space.config.model}`
}

/** The Space a URL hash opens (`#s6-route`, percent-encoded or not), or null: the list. */
export function spaceOfHash(hash: string, spaces: HubSpace[]): HubSpace | null {
  let name = hash.replace(/^#/, '')
  try {
    name = decodeURIComponent(name)
  } catch {
    /* a malformed escape: the name as it is */
  }
  return spaces.find((s) => s.config.model === name) ?? null
}

/** Every model the global Space lists (the scope's universe in lib/scope.ts). */
export function hubModels(spaces: HubSpace[]): string[] {
  return [...new Set(spaces.flatMap((s) => s.models))]
}
