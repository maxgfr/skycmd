// The sections of a decision Space (SpaceView.vue, kind `decision`): the four demos of LiveDecide,
// then, with `proofs`, the three proofs of LiveProof. Pure, tested by tests/space.test.ts.

export const DECISION = [
  {
    mode: 'playground',
    title: 'Playground',
    text: 'Any question, any options, any type: choice, score (levels from lowest to highest) or noul (a true/false statement).',
  },
  {
    mode: 'tools',
    title: 'Tool router for an agent',
    text: 'Describe the tools in one line each; the model picks the tool for the request, or no_tool.',
  },
  {
    mode: 'triage',
    title: 'Inbox and ticket triage',
    text: 'Your own categories, with an "unsure" way out.',
  },
  {
    mode: 'guard',
    title: 'Guardrail on a written rule',
    text: 'A noul question: "The message violates this rule: …". The model returns p(true).',
  },
] as const

export const PROOFS = [
  {
    mode: 'guarantee',
    title: 'Keys and order cannot matter (guaranteed)',
    text: 'Every key moves to another definition, then the order is reversed: the iso model gives each definition the same probability, up to float rounding. s4-micro, which reads the keys, moves.',
  },
  {
    mode: 'paraphrase',
    title: 'Other words for the same options (learned)',
    text: 'Robustness to paraphrased definitions is trained, not guaranteed; measured on held-out paraphrases it is far from perfect.',
  },
  {
    mode: 'dutch',
    title: 'A Dutch book on the model’s own beliefs',
    text: 'For each option the model is asked "The correct answer is: x" and "… is not: x" separately. A coherent model has s + n = 1 and Σ s = 1.',
  },
] as const
