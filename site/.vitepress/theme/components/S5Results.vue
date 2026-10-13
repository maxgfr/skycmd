<script setup lang="ts">
// The s5 results page: every number comes from public/results/s5.json, written by
// `sky-s5-report summary` from the pre-registered analyses (docs/s5-prereg.md); nothing is
// transcribed by hand. A missing section means the experiment has not run yet.
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

type J = any // eslint-disable-line @typescript-eslint/no-explicit-any

const data = ref<J | null>(null)
const failed = ref('')

onMounted(async () => {
  try {
    const r = await fetch(withBase('/results/s5.json'))
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    data.value = await r.json()
  } catch (e) {
    failed.value = (e as Error).message
  }
})

const reports = computed<J>(() => data.value?.reports ?? {})
const shipped = computed<string | null>(() => data.value?.shipped ?? null)
const p1 = computed<J>(() => reports.value.pari1 ?? null)
const p2 = computed<J>(() => reports.value.pari2 ?? null)
const gates = computed<J>(() => (shipped.value ? reports.value[`gates-${shipped.value}-q8_0`] : null) ?? null)
const robust = computed<J>(() => (shipped.value ? reports.value[`robust-${shipped.value}-q8_0`] : null) ?? null)
const coherence = computed<J>(() => (shipped.value ? reports.value[`coherence-${shipped.value}-q8_0`] : null) ?? null)
const selective = computed<J>(() => (shipped.value ? reports.value[`selective-${shipped.value}-q8_0`] : null) ?? null)
const selectiveAll = computed<J>(() =>
  (shipped.value ? reports.value[`selective-allcal-${shipped.value}-q8_0`] : null) ?? null,
)
const phase5 = computed<[string, J][]>(() =>
  ['f-ap', 'f-b'].filter((a) => reports.value[`phase5-${a}`]).map((a) => [a, reports.value[`phase5-${a}`]]),
)
const runs = computed<[string, J][]>(() => Object.entries(data.value?.runs ?? {}))

function n(x: unknown, d = 3): string {
  return typeof x === 'number' && Number.isFinite(x) ? x.toFixed(d) : '–'
}
function iv(i: J, d = 3): string {
  if (!i) return '–'
  return `${n(i.value, d)} [${n(i.lo, d)}, ${n(i.hi, d)}]`
}
function diff(p: J, d = 3): string {
  return p ? iv(p.diff, d) : '–'
}
function yes(b: unknown): string {
  return b === true ? 'pass' : b === false ? 'fail' : '–'
}
function when(t: number | null | undefined): string {
  return t ? new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC' : '–'
}
const ARM: Record<string, string> = { 'f-ap': 'A′ (Laya format, random names)', 'f-b': 'B* (iso)' }
</script>

<template>
  <div class="s5r">
    <p v-if="failed" class="s5r-note">The results file is not published yet ({{ failed }}).</p>
    <template v-else-if="data">
      <p class="s5r-note">
        {{ data.test_log.passes }} logged test passes, pre-registration commit
        <code v-for="c in data.test_log.prereg_commits" :key="c">{{ c.slice(0, 7) }}</code>
        <template v-if="!data.test_log.prereg_commits.length"> (none yet)</template>;
        first {{ when(data.test_log.first_unix) }}, last {{ when(data.test_log.last_unix) }}.
        Shipped model: <strong>{{ shipped ?? 'not chosen yet' }}</strong>.
      </p>

      <h3>Bet 1: invariance by construction (B* against A′, 3 seeds each)</h3>
      <template v-if="p1">
        <p>
          Verdict: <strong :class="p1.success ? 'ok' : 'ko'">{{ p1.success ? 'success' : 'failure' }}</strong>
          (both non-inferiority criteria are required).
        </p>
        <table>
          <thead>
            <tr><th>criterion</th><th>A′</th><th>B*</th><th>B* − A′ [95% CI]</th><th>margin</th><th>verdict</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>accuracy, fitted bench</td>
              <td>{{ n(p1.primary.bench_accuracy.paired.reference) }}</td>
              <td>{{ n(p1.primary.bench_accuracy.paired.variant) }}</td>
              <td>{{ diff(p1.primary.bench_accuracy.paired) }}</td>
              <td>−{{ n(p1.primary.bench_accuracy.margin, 2) }}</td>
              <td>{{ yes(p1.primary.bench_accuracy.non_inferior) }}</td>
            </tr>
            <tr>
              <td>ECE, fitted bench</td>
              <td>{{ n(p1.primary.bench_ece.paired.reference) }}</td>
              <td>{{ n(p1.primary.bench_ece.paired.variant) }}</td>
              <td>{{ diff(p1.primary.bench_ece.paired) }}</td>
              <td>+{{ n(p1.primary.bench_ece.margin, 2) }}</td>
              <td>{{ yes(p1.primary.bench_ece.non_inferior) }}</td>
            </tr>
          </tbody>
        </table>
        <details>
          <summary>Secondary comparisons (Holm-adjusted)</summary>
          <table>
            <thead><tr><th>metric</th><th>A′</th><th>B*</th><th>B* − A′ [95% CI]</th><th>p</th><th>p (Holm)</th></tr></thead>
            <tbody>
              <tr v-for="(v, k) in p1.secondary_holm" :key="k">
                <td>{{ k }}</td><td>{{ n(v.reference) }}</td><td>{{ n(v.variant) }}</td>
                <td>{{ iv(v.diff) }}</td><td>{{ n(v.p_value, 4) }}</td><td>{{ n(v.p_holm, 4) }}</td>
              </tr>
            </tbody>
          </table>
        </details>
      </template>
      <p v-else class="s5r-note">Not analysed yet.</p>

      <h3>Bet 2: coherence across questions (λ* against λ = 0)</h3>
      <template v-if="p2">
        <p>
          Verdict: <strong :class="p2.success ? 'ok' : 'ko'">{{ p2.success ? 'success' : 'failure' }}</strong>
          <template v-if="!p2.success && p2.trained_identities_only"> (gain on the trained identities only)</template>.
          D competence at λ = 0: {{ iv(p2.competence.d_accuracy_lambda0) }} against a required lower bound of
          {{ n(p2.competence.required_lower_bound) }} ({{ p2.competence.competent ? 'measurable' : 'not measurable' }}).
        </p>
        <table>
          <thead><tr><th>criterion</th><th>value [95% CI]</th><th>required</th><th>verdict</th><th>status</th></tr></thead>
          <tbody>
            <tr><td>MAD_neg of λ*</td><td>{{ iv(p2.primary.mad_neg.interval) }}</td><td>&lt; 0.03</td><td>{{ yes(p2.primary.mad_neg.pass) }}</td><td>learned</td></tr>
            <tr><td>Σ s of λ*</td><td>{{ iv(p2.primary.partition_sum.interval) }}</td><td>[0.97, 1.03]</td><td>{{ yes(p2.primary.partition_sum.pass) }}</td><td>learned</td></tr>
            <tr><td>bench accuracy, λ* − λ0</td><td>{{ diff(p2.primary.bench_accuracy.paired) }}</td><td>≥ −0.02</td><td>{{ yes(p2.primary.bench_accuracy.non_inferior) }}</td><td>result</td></tr>
            <tr>
              <td>monotonicity violations, λ* − λ0</td>
              <td>{{ diff(p2.primary.monotonicity_violation_share.paired) }} ({{ n(p2.primary.monotonicity_violation_share.relative_reduction * 100, 1) }}% fewer)</td>
              <td>−30%, CI &lt; 0</td>
              <td>{{ yes(p2.primary.monotonicity_violation_share.pass) }}</td>
              <td>held out of training</td>
            </tr>
          </tbody>
        </table>
        <details>
          <summary>Secondary comparisons (Holm-adjusted)</summary>
          <table>
            <thead><tr><th>metric</th><th>λ = 0</th><th>λ*</th><th>difference [95% CI]</th><th>p (Holm)</th></tr></thead>
            <tbody>
              <tr v-for="(v, k) in p2.secondary_holm" :key="k">
                <td>{{ k }}</td><td>{{ n(v.reference) }}</td><td>{{ n(v.variant) }}</td><td>{{ iv(v.diff) }}</td><td>{{ n(v.p_holm, 4) }}</td>
              </tr>
            </tbody>
          </table>
        </details>
      </template>
      <p v-else class="s5r-note">Not analysed yet.</p>

      <h3>Phase 5: calibration after 8-bit quantization</h3>
      <table v-if="phase5.length">
        <thead><tr><th>arm</th><th>ECE f32</th><th>ECE q8_0</th><th>q8_0 − f32 [95% CI]</th><th>≤ +0.01</th><th>accuracy q8_0 − f32</th></tr></thead>
        <tbody>
          <tr v-for="[a, v] in phase5" :key="a">
            <td>{{ ARM[a] ?? a }}</td>
            <td>{{ n(v.primary.bench_ece.paired.reference) }}</td>
            <td>{{ n(v.primary.bench_ece.paired.variant) }}</td>
            <td>{{ diff(v.primary.bench_ece.paired) }}</td>
            <td>{{ yes(v.primary.bench_ece.non_inferior) }}</td>
            <td>{{ diff(v.primary.bench_accuracy.paired) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="s5r-note">Not analysed yet.</p>

      <h3>s5-lite gates (shipped model, q8_0)</h3>
      <template v-if="gates">
        <table>
          <thead><tr><th>gate</th><th>value</th><th>verdict</th></tr></thead>
          <tbody>
            <tr><td>fitted bench ≥ 0.65</td><td>{{ n(gates.gates.bench_ge_0_65.value) }}</td><td>{{ yes(gates.gates.bench_ge_0_65.pass) }}</td></tr>
            <tr><td>English demo cases ≥ 90%</td><td>{{ n(gates.gates.demo_en_ge_0_90.value) }}</td><td>{{ yes(gates.gates.demo_en_ge_0_90.pass) }}</td></tr>
            <tr><td>no set below its majority baseline</td><td>{{ gates.gates.no_set_below_majority.below.join(', ') || 'none below' }}</td><td>{{ yes(gates.gates.no_set_below_majority.pass) }}</td></tr>
            <tr><td>sets above 0.735 (investigated)</td><td colspan="2">{{ gates.gates.above_0_735_investigate.join(', ') || 'none' }}</td></tr>
          </tbody>
        </table>
        <details>
          <summary>Every evaluation set</summary>
          <table>
            <thead><tr><th>set</th><th>n</th><th>accuracy [95% CI]</th><th>majority</th><th>ECE</th><th>NLL</th></tr></thead>
            <tbody>
              <template v-for="(v, k) in gates.sets" :key="k">
                <tr v-if="v.accuracy">
                  <td>{{ k }}</td><td>{{ v.n }}</td><td>{{ iv(v.accuracy) }}</td><td>{{ n(v.majority) }}</td><td>{{ n(v.ece) }}</td><td>{{ n(v.nll) }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </details>
      </template>
      <p v-else class="s5r-note">Not evaluated yet.</p>

      <h3>Robustness and coherence of the shipped model</h3>
      <template v-if="robust">
        <table>
          <thead><tr><th>test</th><th>value [95% CI]</th><th>status</th></tr></thead>
          <tbody>
            <tr><td>paraphrased definitions: answer flips (two calls)</td><td>{{ iv(robust.paraphrase.flip_rate) }}</td><td>learned</td></tr>
            <tr><td>JevOut-lite: flips under the worst distractor</td><td>{{ iv(robust.jevout_lite.flip_rate) }}</td><td>learned</td></tr>
            <tr v-for="(v, k) in robust.reassign" :key="k">
              <td>label reassignment {{ k }}: flips</td><td>{{ iv(v.flip_rate) }}</td><td>guaranteed (keys never read)</td>
            </tr>
            <tr v-for="(v, k) in robust.policybench" :key="k"><td>{{ k }} accuracy</td><td>{{ iv(v.accuracy) }}</td><td>result</td></tr>
          </tbody>
        </table>
      </template>
      <table v-if="coherence">
        <thead><tr><th>coherence (2609.33209 identities)</th><th>value [95% CI]</th><th>status</th></tr></thead>
        <tbody>
          <tr><td>MAD_neg |s + n − 1|</td><td>{{ iv(coherence.mad_neg) }}</td><td>learned</td></tr>
          <tr><td>Σ s</td><td>{{ iv(coherence.partition_sum) }}</td><td>learned</td></tr>
          <tr><td>|s − c|</td><td>{{ iv(coherence.xf_abs) }}</td><td>learned</td></tr>
          <tr><td>monotonicity violations</td><td>{{ iv(coherence.held_out.monotonicity_violation_share) }}</td><td>held out of training</td></tr>
          <tr><td>additivity |d − (s_i + s_j)|</td><td>{{ iv(coherence.held_out.additivity_abs) }}</td><td>held out of training</td></tr>
          <tr><td>description invariance |d − n|</td><td>{{ iv(coherence.held_out.description_abs) }}</td><td>held out of training</td></tr>
          <tr><td>D questions accuracy (majority {{ n(coherence.held_out.d_majority) }})</td><td>{{ iv(coherence.held_out.d_accuracy) }}</td><td>held out of training</td></tr>
        </tbody>
      </table>
      <p v-if="!robust && !coherence" class="s5r-note">Not evaluated yet.</p>

      <h3>Phase 7: escalation with a controlled error (offline)</h3>
      <template v-if="selective">
        <table>
          <thead><tr><th>α</th><th>LTT coverage</th><th>LTT error of the accepted</th><th>conformal singletons coverage</th><th>their error</th></tr></thead>
          <tbody>
            <tr v-for="(v, k) in selective.by_alpha" :key="k">
              <td>{{ k }}</td><td>{{ n(v.ltt.coverage) }}</td><td>{{ n(v.ltt.error_accepted) }}</td>
              <td>{{ n(v.conformal_singletons.coverage) }}</td><td>{{ n(v.conformal_singletons.error_accepted) }}</td>
            </tr>
          </tbody>
        </table>
        <p class="s5r-note">
          Thresholds from the typed-decisions calibration split (exchangeable with the bench).
          AURC {{ n(selective.aurc) }}, coverage at ≤ 5% error {{ n(selective.coverage_at_5pct_error) }}.
          <template v-if="selectiveAll">
            With every English calibration record (more data, no formal guarantee under shift), LTT at 5%
            accepts {{ n(selectiveAll.by_alpha['5%']?.ltt.coverage) }} of the bench with an error of
            {{ n(selectiveAll.by_alpha['5%']?.ltt.error_accepted) }}.
          </template>
        </p>
      </template>
      <p v-else class="s5r-note">Not evaluated yet.</p>

      <h3>Every run (validation only)</h3>
      <table v-if="runs.length">
        <thead><tr><th>run</th><th>variant</th><th>init</th><th>lr</th><th>λ</th><th>seed</th><th>val acc</th><th>val NLL</th><th>val ECE</th><th>min</th></tr></thead>
        <tbody>
          <tr v-for="[k, v] in runs" :key="k">
            <td>{{ k }}</td><td>{{ v.variant }}</td><td>{{ v.init }}</td><td>{{ v.lr }}</td><td>{{ v.coh }}</td><td>{{ v.seed }}</td>
            <td>{{ n(v.all?.accuracy) }}</td><td>{{ n(v.all?.nll) }}</td><td>{{ n(v.all?.ece) }}</td><td>{{ n(v.minutes, 1) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="s5r-note">No run yet.</p>
    </template>
    <p v-else class="s5r-note">Loading the results…</p>
  </div>
</template>

<style scoped>
.s5r table {
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  display: block;
  overflow-x: auto;
}
.s5r-note {
  color: var(--vp-c-text-2);
  font-size: 14px;
}
.s5r code {
  margin: 0 2px;
}
.s5r .ok {
  color: var(--vp-c-green-1, #16a34a);
}
.s5r .ko {
  color: var(--vp-c-danger-1);
}
.s5r details {
  margin: 8px 0 16px;
}
</style>
