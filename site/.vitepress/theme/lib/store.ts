// Live numbers shared by the landing page components (hero game, counters, feature cards).
import { reactive } from 'vue'

export const live = reactive({
  /** Name of the model playing in the hero, or '' for the oracle. */
  model: '',
  params: 0,
  bytes: 0,
  /** Rolling median of a full AI decision inside WASM (both questions + physics). */
  medianUs: 0,
  /** Decisions actually taken per second by the hero game. */
  decisionsPerS: 0,
  /** Best score of the hero game in this visit. */
  best: 0,
  status: 'loading' as 'loading' | 'model' | 'oracle' | 'error',
})
