// The verdict notice of a model with a pre-registered gate (manifest `gate`, `gate_note`,
// `gate_url`): a bold lead, the note, then the link to the result. A Space opens with it, and the
// site's demos show it next to the selected model. A model published without a gate gets none.
//
// The spaces between the three parts live in the strings built here, not in the templates: Vue's
// template compiler drops a text node made only of whitespace at the start of a `<template>`, which
// ran the sentences together ("not for use.It missed …"). components/GateNotice.vue renders it.

export interface GateFields {
  gate?: 'passed' | 'failed' | null
  gate_note?: string | null
  gate_url?: string | null
}

export interface GateNotice {
  tone: 'passed' | 'failed'
  /** The bold lead, a whole sentence. */
  lead: string
  /** What follows the lead: each sentence with its leading space, and a space before the link. */
  note: string
  /** The result the link opens ("The result."), when the manifest gives one. */
  url?: string
}

/** The notice of `entry`, or null for a model without a gate. `subject` names the model in the
 * lead (the site's demos, which list several models); `extra` is a sentence added after the note. */
export function gateNotice(entry: GateFields | null | undefined, subject?: string, extra?: string): GateNotice | null {
  const tone = entry?.gate
  if (tone !== 'passed' && tone !== 'failed') return null
  const lead =
    tone === 'failed'
      ? `${subject ? `${subject} failed` : 'Failed'} its pre-registered gate: published as a result, not for use.`
      : `${subject ? `${subject} passed` : 'Passed'} its pre-registered gate.`
  const url = entry?.gate_url || undefined
  const note = (entry?.gate_note ? ` ${entry.gate_note}.` : '') + (extra ? ` ${extra}` : '') + (url ? ' ' : '')
  return { tone, lead, note, url }
}

/** The notice as one plain string, as a reader sees it. */
export function gateNoticeText(n: GateNotice): string {
  return n.lead + n.note + (n.url ? 'The result.' : '')
}
