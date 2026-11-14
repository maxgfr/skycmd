# Where a sub-megabyte, millisecond, calibrated decision model is useful: a research dossier for skycmd s6

Research date: 14 November 2026. Three load-bearing sources were opened a second time the same day and match what is written here: the GitHub post of 13 October 2026 (section 1.1), the cbx-secrets-detector-17M model card (section 1.1) and the TinyCast paper (section 5.1). Nothing in this file is a decision: experiments that follow from it get their own pre-registration. Each fact gives its source URL and the date of that source where one is known. Text marked **[Inference]** is my own reasoning, not a claim from a source. "Not verified" means I looked and could not confirm it.

---

## 0. Summary

- **[Inference]** The demand is not for a general decision model. Large "decision" models already score 90–95% on an open bounded-decision benchmark (section 3.2). The demand is for a **cheap verifier placed behind a high-recall rule or signal**, in places where a 50–800 MB model cannot go: pre-commit hooks, agent hooks, browser extensions, low-RAM devices, microcontrollers. In October 2026 the largest secret-scanning product replaced its LLM pipeline with exactly this design (section 1.2).
- The strongest case is **secret detection reshaped as a filter on rule candidates**, using the candidate plus its context. The open competition appeared two weeks ago (a 17M span model) and is about 70x larger than s6-secrets-pico (section 1).
- The second case is a **voice end-of-turn model built on mel or prosody features** rather than codec tokens, distilled from Smart Turn v3.2 (BSD-2) using CC-BY-4.0 data (section 4).
- Forecasting is already occupied at the bottom of the size range by TinyCast, which has 146K parameters (section 5). s6-cast-pico has to beat it on GIFT-Eval, or it has no reason to exist.

---

## 1. Secret detection with ML

### 1.1 Facts: tools and their reported numbers

| Tool / model | Kind | Size | Reported result | Source (date) |
|---|---|---|---|---|
| SecretBench comparison (9 tools) | rules + vendors | – | Precision: GitHub Secret Scanner 75%, Gitleaks 46%, Commercial X 25%. Recall: Gitleaks 88%, SpectralOps 67%, TruffleHog 52%. Five tools below 7% precision. FP rates 25–99%. | https://arxiv.org/pdf/2307.00714 (2023) |
| CredData benchmark (Samsung) | rules + ML | – | Run on train+test, "Updated: April 2022". CredSweeper P 0.917 / R 0.808 / F1 0.859; gitleaks 0.526/0.244/0.334; detect-secrets 0.142/0.381/0.206; truffleHog 0.25/0.009/0.018; truffleHog3 0.150/0.547/0.235 | https://github.com/Samsung/CredData (README, table dated Apr 2022) |
| CredSweeper ML validator | BiLSTM on last 50 chars of the candidate + 91 handcrafted features, filters rule hits | `ml_model.onnx` = 11,545,911 bytes (11.5 MB), checked through the GitHub API on 14 Nov 2026 | Thresholds `lowest…highest` set from F-0.25…F-4 on the CredData test set. Latest release v1.19.6, 5 Nov 2026 | https://credsweeper.readthedocs.io/en/v1.6.4/overall_architecture.html ; https://github.com/Samsung/CredSweeper |
| cbx-secrets-detector-17M (CobuildX-AI) | token classifier with spans (Ettin-17M encoder), Apache-2.0 | 16.8M params (F32 safetensors, about 67 MB, my estimate) | CredData test (168 repos, 3,493 secrets, 17,961 non-secret lines), span F1: **0.805** (P 0.80/R 0.81); Gitleaks 8.30.1 0.442; detect-secrets 1.5.0 0.460; TruffleHog 3.99.2 0.042 (offline, no verification). Issue reports: 0.836 vs Gitleaks 0.678. Latency 37 ms per 1,000 tokens on an i7-10700 with 8 threads; 6 ms on a GPU. 32M variant: CredData F1 0.820 | https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M (created 30 Oct 2026 per HF API) |
| GitHub Copilot secret scanning (generic passwords), v1 | LLM pipeline: GPT-3.5 few-shot, then GPT-4 confirmation | – | "94% reduction in false positives" in some organizations. No absolute P/R published | https://github.blog/engineering/platform-security/finding-leaked-passwords-with-ai-how-we-built-copilot-secret-scanning/ (4 Mar 2025) |
| GitHub secret-detection model, v2 | **Stage 1: high-recall regex candidates. Stage 2: ModernBERT classifier** on candidate value + file path + secret type + bounded context. ONNX, 8-bit | Size not disclosed ("compact") | Against the LLM pipeline: offline precision +13%, recall +19%, owner-weighted acceptance +11.5%, median scan time −86%, serving cost "more than 98% lower". Rows grouped by repo/value before splitting. Hard synthetic negatives: docs, test fixtures, public keys, checksums, near-miss pairs | https://github.blog/ai-and-ml/githubs-model-for-secret-detection-finding-more-secrets-with-less-noise/ (13 Oct 2026) ; https://github.blog/changelog/2026-10-07-purpose-built-model-for-leaked-secret-detection/ (7 Oct 2026) |
| GitHub push protection for AI-detected secrets | same classifier at push time | – | "more than doubled the number of pushes where push protection caught a secret". Public preview, off by default, costs AI credits | https://github.blog/changelog/2026-10-28-push-protection-beyond-predefined-secret-patterns/ (28 Oct 2026) |
| GitGuardian FP Remover | in-house transformer, run after generic detectors | Self-hosted: 3 vCPU and 2.5 GiB RAM per replica | v1 removed up to 50% of FPs (Jun 2024). v2 found about 60% more FPs. Docs: precision about 95%. Jul 2026: flags 25–40% of generic findings on non-code sources (Slack, Jira…) as FPs; non-code filtering uses an LLM | https://docs.gitguardian.com/secrets-detection/secrets-detection-engine/machine_learning ; https://blog.gitguardian.com/ai-false-positive-remover-v2/ ; https://docs.gitguardian.com/releases/saas/2026/07/08/changelog |
| IssueGuard (Chrome extension + backend) | 761 regexes, then CodeBERT-base on ±200 chars | about 125M (CodeBERT-base, size not stated in paper) | Test P 92.49 / R 92.91 / F1 92.70. Regex alone: P 6.8 / R 100 / F1 12.8. TruffleHog 69.69, Gitleaks 63.60. On 178 real repos, Secret-class F1 64.2. Inference about 10 ms (GPU), about 198 ms end to end | https://arxiv.org/html/2602.08072v1 (8 Feb 2026) |
| Fine-tuned 7–8B LLMs on SecretBench | LLM as a filter on candidates | 7–8B | LLaMA-3.1-8B F1 0.985 on a balanced 3,000-sample test (the paper's confusion matrix is internally inconsistent) | https://arxiv.org/html/2504.18784v1 (26 Apr 2025) |
| Betterleaks (successor to Gitleaks, MIT) | rules + "token efficiency" (BPE tokens per char) in place of entropy | – | Author's claim: 98.6% recall on CredData against 70.4% for entropy. LLM-assisted classification planned for v2 | https://www.bleepingcomputer.com/news/security/betterleaks-a-new-open-source-secrets-scanner-to-replace-gitleaks/ (Mar 2026) |
| OpenAI Privacy Filter | PII + secrets token classifier, Apache-2.0 | 1.5B total / 50M active MoE. q4f16 ONNX 772 MB for WebGPU | Model card reports CredData secret P 0.747 / R 0.965 / span F1 0.617 (quoted in a search snippet; I did not open the tables) | https://arxiv.org/abs/2608.18274 (18 Aug 2026) ; https://github.com/montevive/openai-privacy-filter |
| Kingfisher (MongoDB, Apache-2.0, Rust + Hyperscan) | rules + **live validation** + blast-radius mapping | – | v2.15.0, 28 Oct 2026. Nosey Parker archived (last release May 2025). Gitleaks "feature complete" since 21 May 2026 | https://www.mongodb.com/company/blog/product-release-announcements/introducing-kingfisher-real-time-secret-detection-validation ; https://appsecsanta.com/kingfisher |

Where rules fail, according to the sources:
- Generic passwords and custom tokens have no fixed format. GitHub says regex "generates excessive noise" on them (Mar 2025 post).
- Entropy flags UUIDs, hashes and base64 payloads (Betterleaks write-up).
- On CredData, recall of the default rule sets is low (gitleaks 0.24–0.29).
- GitGuardian's 2026 report: 55.4% of leaked secrets came from specific detectors in 2025 (46.3% in 2024), so about 45% were generic. 46% of critical secrets "cannot be validated". It also reports Claude Code–assisted commits leaking at 3.2% against a 1.5% baseline. I took these from the report summary, not the PDF tables. https://www.gitguardian.com/state-of-secrets-sprawl-report-2026 (17 Mar 2026) ; https://thehackernews.com/2026/03/the-state-of-secrets-sprawl-2026-9.html

### 1.2 What the field converged on
The sources independently describe the same architecture: **a high-recall rule finds candidate spans, then a learned classifier judges each candidate in context.** This is the design of GitHub (Oct 2026), CredSweeper, IssueGuard, GitGuardian FP Remover and the SecretBench LLM paper. The rule supplies the span, so the model never needs to output one.

### 1.3 Datasets

| Dataset | Content | Licence / access | Source |
|---|---|---|---|
| CredData (Samsung) | 297 repos, 19.46M lines, 73,842 labelled lines, 4,583 True. Values **obfuscated** (masked, replaced or randomized). Paths obfuscated | Repo Apache-2.0, but **each file keeps its project's licence** (looked up by RepoID). Files are fetched with `download_data.py`. cbx used only the MIT/Apache/BSD repos (116) for training | https://github.com/Samsung/CredData |
| SecretBench | 97,479 candidates (15,084 true), 818 repos, 49 languages | Repo MIT, but **access needs a signed data-protection agreement**. Data in BigQuery/GCS | https://github.com/setu1421/SecretBench |
| FPSecretBench | false positives from 9 tools on SecretBench | same agreement | https://github.com/setu1421/FPSecretBench |
| Secret leaks in issue reports (Ahmed/Wahab et al.) | 54,148 candidates, 5,881 true (IssueGuard numbers) | **CC-BY-4.0, open**, Zenodo 10.5281/zenodo.19622962, published 16 Apr 2026, 134 MB zip (checked through the Zenodo API). Contains real secrets: cbx replaced them before training | https://zenodo.org/records/19622962 ; https://arxiv.org/abs/2410.23657 |

### 1.4 Not verified
- The size of GitHub's ModernBERT model, and the "<2 ms per batch" figure (from a secondary article; I did not find it on the changelog page).
- Current CredSweeper numbers on CredData: the README table dates from 2022, and the CI benchmark file returned 404.
- Betterleaks' 98.6% recall: the author's own measurement.
- OpenAI Privacy Filter's CredData numbers: from a snippet only.

### 1.5 [Inference] What would make a 250 KB model a real addition
1. **A false-positive filter on rule candidates**, not a detector of whole texts. Input: candidate value, key or variable name, file path or extension, and about 100–200 chars of context. Output: calibrated P(real) with F-beta presets, as CredSweeper does. The span problem disappears because the rule gives it. The "right line 69%" problem disappears for the same reason.
2. **Generic secrets are the target**: passwords, connection strings, custom tokens. For provider keys, regex plus live validation (Kingfisher, GitHub partner patterns) already wins. The "which service" head on synthetic data is not a differentiator.
3. **The size argument is real but narrow.** CredSweeper's filter is 11.5 MB, cbx-17M is about 67 MB, the ModernBERT behind GitHub's classifier is unknown but its base is 149M parameters, and OpenAI's filter is 772 MB. A 250 KB WASM filter fits a browser paste guard, a git pre-commit or agent hook, and an editor extension, all with no download step. Section 7 shows those products ship regex-only today, or disable their ML on low-RAM machines.
4. Credibility needs **CredData test numbers on the cbx split, against Gitleaks/Betterleaks + CredSweeper-ML + cbx-17M**, plus the CC-BY issue-report test set. The synthetic bench is not enough.

---

## 2. Prompt-injection, jailbreak and guardrail classifiers

### 2.1 Facts

| Model | Size | Result | Latency | Licence | Source (date) |
|---|---|---|---|---|---|
| Llama Prompt Guard 2 86M | mDeBERTa-base. 278.8M total incl. embeddings | EN AUC .998, recall 97.5% at 1% FPR, multilingual AUC .995. AgentDojo APR 81.2% | 92.4 ms (A100, 512 tok) | Llama 4 Community, gated | https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M (Apr 2025) |
| Llama Prompt Guard 2 22M | DeBERTa-xsmall. **70.8M total (49.4M of it embeddings)** | AUC .995, recall 88.7% at 1% FPR, multilingual .942, AgentDojo 78.4% | 19.3 ms (A100) | same | same ; https://huggingface.co/PinataCloud/Llama-Doorman-22M |
| Llama-Doorman-22M (PinataCloud) | fine-tune of PG2-22M, 87.3 MB int8 ONNX, runs in onnxruntime-web | On agent tool output: 3.6% attacks wrongly cleared, 0.4% false alarms raised, about 0.24 s/check. Benchmarked against 13 models including LLMs | ~0.24 s | Llama 4 Community | https://huggingface.co/PinataCloud/Llama-Doorman-22M (no date on card) |
| ProtectAI deberta-v3-base-prompt-injection-v2 | 184M | PINT 79.14% (2025-05-02). Qualifire benchmark 65.2% (CourtGuard paper) | – | Apache-2.0 | https://github.com/lakeraai/pint-benchmark ; https://arxiv.org/html/2510.19844 |
| Lakera Guard (commercial) | – | PINT 95.22% (PINT is Lakera's own and partly private) | – | proprietary | https://github.com/lakeraai/pint-benchmark |
| Independent CPU test of PG2 | – | 105 cases: 22M 76%, 86M 82%. Both miss the same hard categories | ONNX CPU 73 ms (22M), 151 ms (86M) | – | https://github.com/crunchtools/trentina/issues/17 |
| Claude Code auto mode (two stages) | stage 1 is a "fast single-token filter" | Stage 1 FPR 8.5% on 10k real calls; full pipeline 0.4%. FNR on overeager actions 17%, on synthetic exfiltration 5.7%. An independent stress test found 81% FNR on ambiguous authorizations | – | – | https://www.digitalapplied.com/blog/claude-code-auto-mode-classifier-numbers-published (summarizes Anthropic's post) ; https://arxiv.org/abs/2604.04978 (Apr 2026) |
| vLLM Semantic Router jailbreak guard | ModernBERT | – | 3–8 ms (docs) | Apache-2.0 | https://vllm-semantic-router.com/docs/overview/architecture/system-architecture/ |

Robustness findings:
- "Defenses Against Prompt Attacks Learn Surface Heuristics" (12 Jan 2026): position bias (suffix rejection rises from <10% to 90%), single trigger tokens raise false refusals by up to 50%, topic shift costs up to 40% accuracy. https://arxiv.org/abs/2601.07185
- PIDS-Bench (14 Sep 2026): a detector with held-out F1 >0.98 still misclassifies about 1/3 of externally sourced benign security-adjacent prompts. No detector met F1 ≥0.95 together with hard-benign FPR ≤0.10. https://arxiv.org/abs/2609.15017
- Embedding + random forest classifiers can beat fine-tuned baselines on AUC (https://arxiv.org/pdf/2410.22284, Oct 2024). NVIDIA reports an embedding + classical-classifier jailbreak detector with FPR <1% (https://arxiv.org/pdf/2412.01547, Dec 2024).
- I found no published model2vec/static-embedding injection classifier with numbers. A Medium post describes one, without a benchmark.

### 2.2 Datasets (licences checked through the HF API on 14 Nov 2026)
deepset/prompt-injections (Apache-2.0), jackhhao/jailbreak-classification (Apache-2.0), Lakera/gandalf_ignore_instructions (MIT), microsoft/llmail-inject-challenge (MIT), leolee99/NotInject (MIT, over-defense test set), reshabhs/SPML_Chatbot_Prompt_Injection (MIT). PINT: partly proprietary; access by request (README). Qualifire benchmark: licence not verified (HF API returned no JSON).

### 2.3 [Inference] Room for a sub-1 MB first pass
- There is size room: the smallest credible open model (PG2-22M) is 70.8M parameters and 87 MB, because of its 128k-vocabulary embedding table.
- This is an adversarial task, and the 2026 papers show that even 100M+ models learn surface cues. A 1M-parameter model will learn more of them.
- The only defensible role is a **recall-tuned stage 1 in a cascade** (as in Claude auto mode, where stage 1 alone has 8.5% FPR). It would run on tool output inside an agent hook, and escalate rather than decide.
- It must be judged by three numbers: attack recall at a fixed escalation rate on held-out sources (BIPIA, llmail-inject, PINT if granted), FPR on NotInject, and robustness to obfuscation.
- It must never be sold as "protection". Medium priority, high reputational risk.

---

## 3. Intent / tool routing, "should I call a tool", "which model"

### 3.1 Facts
- **Model2Vec (static embeddings, MIT).** Over 14 SetFit datasets with 1,000 training examples each, average accuracy is tfidf 70.8, potion + logreg 78.0, potion full fine-tune 79.2, SetFit (MiniLM-L6) 82.6, bge-base + logreg 82.8. Throughput (CPU, samples/s): tfidf 108k, m2v full fine-tune 24.7k, SetFit 716, bge-base 118. massive_scenario: tfidf 77.3, m2v 84.4, SetFit 83.5. https://minish.ai/packages/model2vec/results/ (undated)
- Against fastText (15 datasets), Model2Vec wins on average, "smaller than expected". fastText with no preprocessing runs at 25k samples/s with score 78.5. https://minish.ai/blog/2025-07-28-fasttext/ (28 Jul 2025)
- Model2Vec v0.10.0 (11 Oct 2026) reran a 26-dataset benchmark with fastText and MiniLM baselines. The numbers were not on the page. https://minish.ai/blog/2026-10-12-model2vec-0100/
- **Sizes (HF API, 14 Nov 2026):** potion-base-2M 7.56 MB f32, potion-base-4M 15.1 MB, potion-base-8M 30.2 MB. **[Inference]** int8 or vocab quantization would put potion-2M near 2 MB, which is still 3–8x the s6 files.
- **MASSIVE intents (60 classes).** Supervised XLM-R base (270M): en-US 88.3%, locale average 85.1%. Zero-shot cross-lingual from English (XLM-R base): average 70.6%. https://arxiv.org/pdf/2204.08582 (2022, Table 3)
- **Zero-shot intent with LLMs (41 open-weight models).** Average/top accuracy: CLINC150 0.468/0.836, Banking77 0.388/0.708. https://arxiv.org/pdf/2607.27421 (Jul 2026). **[Inference]** s6-nano's 0.43 English "zero-shot" on MASSIVE, which amendment 14 says is not really zero-shot, sits below the average LLM. That is not a selling point.
- **When2Call** (NVIDIA, NAACL 2025): four-way choice of tool call / ask follow-up / say unable / answer directly. Dataset CC-BY-4.0. Artificial Analysis runs 3,423 multiple-choice items. https://arxiv.org/abs/2504.18851 ; https://huggingface.co/datasets/nvidia/When2Call. Not verified: per-model scores for models under 1B.
- **LLM routing ("which model").** RouteLLM: ">2x" cost reduction in some cases (abstract), https://arxiv.org/abs/2406.18665 (2024). "Rethinking predictive routing": tuned kNN matches or beats learned routers, and degrades 2.63 pts under shift against 6.67 for the alternatives, https://arxiv.org/html/2505.12601v2 (2025). "The Routing Plateau": on RouterBench the top 15 routers sit within 0.23 pp, kNN scores 0.768 against 0.762–0.772, https://arxiv.org/html/2606.07587 (Jun 2026).
- **Arch-Router-1.5B**: 93.17% routing accuracy at 51±12 ms on an L40S, on synthetic data. https://arxiv.org/pdf/2506.16655 (Jun 2025)
- **Routers shipped in products.** The vLLM Semantic Router runs ModernBERT intent, PII and jailbreak classifiers in Rust (Candle), at 3–15 ms per classifier. https://arxiv.org/pdf/2510.08731 ; https://vllm-semantic-router.com/docs/overview/architecture/system-architecture/

### 3.2 An open benchmark that matches the s6 interface
**Decision Bench** (Atlan, 24 Sep 2026; MIT harness; rows keep their source licences): 1,071 rows, 35 tasks, a probability for every option, scored on accuracy, calibration, latency and cost. Results from PRs: Amos 1 large (27B) 94.87%, Celeris-1 Decision 94.12%, Microsoft Decision 1 90.73% on the 949 text-only rows. https://huggingface.co/datasets/atlanai/decision-bench ; https://github.com/atlanai/decision-bench. **[Inference]** This shows that "general bounded decisions with probabilities" is now a product category served by 7–27B models at about 95%. A 1.5M model cannot compete there; its value has to come from size and locality on narrow tasks.

### 3.3 [Inference] What s6 would need to reach
For any narrow routing or intent task, the honest baseline is **potion-base-2M (or fastText) + logistic regression trained on the same labels, at a matched file size**, together with a kNN over the same embeddings. If s6 does not beat that at matched bytes and latency, a user should pick Model2Vec, which is mature, MIT-licensed and already has WASM/JS ports. I could not verify that last part with a measured browser number.

---

## 4. End-of-turn / turn-taking for voice agents

### 4.1 Facts

| System | Input | Size | Reported accuracy | Latency | Licence / data | Source (date) |
|---|---|---|---|---|---|---|
| Pipecat Smart Turn v3.2 | **raw 16 kHz audio, up to 8 s**, run after Silero VAD detects silence | 8M params. Whisper-tiny encoder + linear head. 8.7 MB int8 ONNX (32 MB fp32) | Speko, 200 real English clips: **94.0% acc, 1.0% false cutoff** at END-recall ≥85%. Synthetic TTS clips: 62.4%, 57.7% false cutoff | "as little as 10 ms on some CPUs", about 65 ms on a cloud instance. Speko: 32 ms warm CPU | BSD-2. Data smart-turn-data-v3.2 train (270k) / test, **CC-BY-4.0** | https://github.com/pipecat-ai/smart-turn ; https://huggingface.co/pipecat-ai/smart-turn-v3 ; https://speko.ai/blog/voice-agent-turn-taking-benchmark (7 Aug 2026) |
| LiveKit text turn detectors (v0.4.1 intl, v1.2.2 EN) | transcript | multilingual based on Qwen2.5-0.5B, 396 MB | Speko real clips: 87.0% / 12.0% FC (intl), 82.0% / 23.0% (EN) | 29 ms + STT wait; 5 ms (EN) | LiveKit model licence ("other") | https://docs.livekit.io/agents/logic/turns/turn-detector/ ; Speko as above |
| LiveKit Turn Detector v1 | **audio → adapter → LM** (semantic + acoustic), 14 languages. "v1-mini" open-weight for CPU | not stated | per-language chart only, no table | not stated | – | https://livekit.com/blog/solving-end-of-turn-detection (17 Jun 2026) |
| Krisp turn-taking v1 / v2 / TP v3 | audio only | v1: 6.1M params, 65 MB | v1 balanced acc 0.82. v2 0.823 clean / 0.768 noisy. TP v3 88.05 against Smart Turn v3.2 77.41 (Krisp's own test) | CPU | proprietary SDK | https://krisp.ai/blog/turn-taking-for-voice-ai/ ; https://krisp.ai/blog/krisp-turn-taking-v2-voice-ai-viva-sdk/ |
| Turnsense | transcript | SmolLM2-135M | Speko real 86.0% / 16% FC; TTS 98.8% | 127 ms | – | Speko |
| TEN Turn Detection | transcript | 7B LLM + an ASR step | "good accuracy, heavy resources" (Easy Turn paper) | – | Apache-2.0 | https://arxiv.org/pdf/2509.23938 ; https://huggingface.co/TEN-framework/TEN_Turn_Detection |
| VAD + silence timer | audio | Silero VAD about 2 MB, <1 ms per 30 ms chunk, MIT | 46.9% acc (Speko) | – | MIT | https://github.com/snakers4/silero-vad |
| "Less can be More" ablation (Amazon) | streaming acoustic/prosodic/text | head 261K trainable on frozen Zipformer2 (70M) | acoustic+prosodic F1 **0.930**, FA 7.8%, median 400 ms. Prosody-only (5 features: f0, Δf0, voicing, speech, silence duration) F1 0.828. Adding text made it *worse* (APT 0.909) | A10G | proprietary telephony corpus | https://arxiv.org/html/2609.11066 (10 Sep 2026) |
| VAP (Voice Activity Projection) | stereo audio. CPC + transformers | small | real time on CPU | – | – | https://arxiv.org/abs/2401.04868 (Jan 2024) |

### 4.2 [Inference] Lessons for the s6-ear redesign
- **The working designs read audio features** (log-mel through a Whisper-tiny or Zipformer encoder, or explicit prosody), or else transcripts. I found **no working end-of-turn system built on discrete speech-codec tokens**. That is consistent with the control model beating s6-ear. Codec tokens are optimized for reconstruction, and pitch contour, the strongest cue, may be poorly exposed at low bitrates. This is a hypothesis, not a sourced fact.
- Redesign: 80-bin log-mel (or 40-bin) plus the five prosodic features of 2609.11066 → small causal conv/ternary transformer → P(end) per 40 ms frame, with a "hold for N frames" rule. Gate it behind Silero VAD the way Smart Turn is.
- **Distill from Smart Turn v3.2** (BSD-2 permits it) on the CC-BY-4.0 Smart Turn data, and evaluate on smart-turn-data-v3.2-test. Report accuracy and false-cutoff at END-recall ≥85%, Speko's protocol.
- The bar: Smart Turn v3.2 at 8.7 MB, and a silence timer. A 1 MB model at about 1–3 ms in WASM that comes within 2–3 points of Smart Turn would be the only option small enough to ship inside a web page's voice widget with no server.
- Known trap: audio models fail on TTS speech (Speko). Test on real speech only, and say so.

---

## 5. Tiny time-series forecasters

### 5.1 Facts (GIFT-Eval: ratios to seasonal naive, lower is better; 97 configurations)

| Model | Params | MASE | CRPS/WQL | Note | Source |
|---|---|---|---|---|---|
| **TinyCast** | **146,505** (138.1 KiB as INT8 on a Cortex-M7) | 0.774 | 0.545 | dilated causal convs + **FFT-detected periods, phase folding**. Trained on GIFT-Eval-Pretrain + KernelSynth + synthetic. Firmware INT8: 0.833 / 0.581 | https://arxiv.org/html/2608.15767v1 (16 Aug 2026) |
| Reverso-Nano | 200K | 0.760 | point only | | same |
| TTM-R3 | 1.4M | 0.724 | 0.520 | saw GIFT-Eval data in pretraining | same ; https://huggingface.co/IUseAMouse/timessm-2.5m |
| TimeSSM-2.5M | 2.5M | 0.799 bare / 0.754 with resampling | 0.545 / 0.513 | Apache-2.0 | https://huggingface.co/IUseAMouse/timessm-2.5m (leaderboard snapshot 6 Sep 2026) |
| Toto-2.0-4m | 4.1M | 0.757 | 0.524 | | TinyCast paper |
| FlowState | 3M / 9.1M / 10.6M | 0.712 / 0.726 / 0.704 | 0.496 / 0.502 / 0.490 | SSM, IBM | https://arxiv.org/html/2508.05287v1 |
| Chronos-Bolt tiny | 9M (34.6 MB) | – | 0.601 | Apache-2.0 | timessm card ; HF API |
| FLAIR (training-free) | 0 | 0.838 | 0.587 | | TinyCast paper |
| AutoARIMA / AutoTheta / AutoETS | 0 | 1.074 / 1.090 / 1.212 | 0.912 / 1.244 / 7.489 | statsforecast | TinyCast paper |
| TTM-r2 (granite) | about 805K–1M, 3.24 MB file | – | – | point forecasts only, Apache-2.0; IBM "will not be maintaining" | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2 |

GIFT-Eval data: the HF entries Salesforce/GiftEval and GiftEvalPretrain are tagged Apache-2.0 (HF API), but they aggregate datasets that each have their own terms. I did not verify per-dataset licences.

### 5.2 [Inference]
A sub-1 MB forecaster is only worth using if it beats **TinyCast (nWQL 0.545 / nGMASE 0.774 at 146K)** and **FLAIR at zero parameters (0.587 / 0.838)**, zero-shot, with no GIFT-Eval leakage, and ideally also on Chronos-ZS or fev-bench, which TinyCast treats as the cleaner test. Classical statsforecast baselines are a low bar on GIFT-Eval: they lose to seasonal naive on MASE. s6-cast-pico should copy TinyCast's computed-periodicity front end, then compete on browser and WASM convenience and calibration (interval coverage). Its niche is crowded but measurable.

---

## 6. The technique

### 6.1 Facts
- **Ternary QAT at small scale.** 1.58-bit training "on par with or sometimes better than" 16/32-bit for MLPs, GNNs, encoder-only and encoder-decoder transformers, with a possible regularization effect (Nielsen, Galke, Schneider-Kamp, ICAART 2025). https://arxiv.org/abs/2411.05882
- **Ternary kernels.** bitnet.cpp offers I2_S (2-bit multiply-add) and TL1/TL2 lookup tables (2 or 3 weights → 4/5-bit index into precomputed activation sums). Up to 6.25x over fp16 and 2.32x over T-MAC, measured on LLMs. https://arxiv.org/pdf/2502.11880 (ACL 2025) ; https://arxiv.org/pdf/2410.16144
- **Recursion.** TRM (7M params): ARC-AGI-1 44.6%, Sudoku-Extreme 87.4% against HRM (27M) at 40.3% / 55.0%. https://arxiv.org/pdf/2510.04871 (Oct 2025). ARC Prize's HRM analysis: a same-size transformer in the HRM pipeline matched it, and removing the **outer refinement loop / deep supervision** collapsed performance. https://arcprize.org/blog/hrm-analysis (Aug 2025; read through quotes, not opened directly). Both results are on puzzle tasks, not text classification.
- **Early exit.** CalexNet trains and calibrates lightweight early-exit branches jointly (https://arxiv.org/pdf/2509.08318, title and abstract level only). CALM-style confidence-based exit: https://arxiv.org/abs/2207.07061 (2022).
- **Distillation and calibration.** Soft labels act as implicit regularization against overconfidence (cited in https://arxiv.org/html/2603.12270v1). *Verbalized* LLM confidence used as soft labels for BERT: "ECE improves slightly, Brier worsens" (https://arxiv.org/pdf/2605.11954, May 2026). Teacher calibration matters (https://arxiv.org/pdf/2508.20224, 2025; findings not read). Classic temperature scaling: https://arxiv.org/abs/1706.04599.
- **Static embeddings.** Model2Vec distills a sentence transformer into a per-token table, so inference is lookup + mean, numpy-only (section 3.1).
- **Browser.** Relaxed SIMD (`i32x4.relaxed_dot_i8x16_i7x16_add_s` etc.) has shipped in Chrome/Edge 114 and Firefox 146 (9 Dec 2025). It is **not in stable Safari**, only Technology Preview 250 (Aug 2026). https://github.com/web-platform-dx/developer-signals/issues/205 ; https://github.com/google-ai-edge/LiteRT/issues/10903 (15 Oct 2026). Chrome says relaxed SIMD speeds workloads 1.5–3x (https://developer.chrome.com/blog/io24-webassembly-webgpu-1, 2024). For models under about 100M params, WASM often beats WebGPU because of transfer overhead: about 9.6x on a 22M embedder (https://github.com/huggingface/transformers.js/issues/1750, Aug 2026; https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/). Native int8 only helps where a dot-product instruction exists (https://arxiv.org/pdf/2607.23227, Jul 2026).

### 6.2 Not verified
- Any ternary result specifically below 10M params on text classification, beyond Nielsen et al.'s small settings. I did not read their tables.
- Any published WASM ternary-kernel benchmark.

---

## 7. Where tiny on-device classifiers are deployed and wanted

- **Browser paste guards for AI chats.** Airlock, VamiGuard (Apache-2.0, "<5 ms", regex), PasteSecure, LLM Guard, Strac, AIovert, anonym.legal. **Almost all are regex-only.** DFKI's Privacy Guardrail runs optional in-browser transformer NER through ORT-Web, recommends **≥16 GB RAM + WebGPU**, and **auto-disables local AI at ≤2 GB RAM**. https://github.com/dfki-dsa/pii-guardrail-browser-extension/ ; https://github.com/sumanthkatta-dev/airlock-dlp-security ; https://vamiguard.com/ (2026). OpenAI Privacy Filter in the browser is a 772 MB download (section 1.1).
- **Coding-agent hooks.** AgentShield, agent-guard and agent-seatbelt-lite are PreToolUse guards built on regex or parsers. I found no published ML classifier. https://github.com/AI-AgentLens/agentshield-oss ; https://github.com/ly-solutions/agent-guard. Claude Code auto mode uses a fast stage-1 filter plus a reasoning stage-2 (section 2.1). Agent commits leak secrets at about twice the baseline rate (GitGuardian, section 1.1).
- **Issue trackers.** IssueGuard (Chrome extension), with the model on a FastAPI backend (section 1.1).
- **Browsers.** Chrome's client-side phishing detection runs a TFLite model in the renderer (https://www.chromium.org/developers/design-documents/safebrowsing/). An AsiaCCS 2024 study found that of 300 phishing sites only 66% were classified and 14% of those correctly (https://yancomm.net/papers/2024%20-%20AsiaCCS%20-%20Deep%20Dive%20into%20Client-Side%20Anti-Phishing.pdf). Model size not verified.
- **Voice.** Silero VAD (2 MB, MIT) is the archetype of a tiny on-device model with real adoption. Smart Turn has 8 MB.
- **Microcontrollers.** TinyCast INT8 at 138 KiB on an STM32H7 (section 5).
- Not researched in depth for lack of public numbers: keyboards (Gboard), SMS spam filter extensions, IoT anomaly detection.

**[Inference]** The repeated gap is "**regex today, because the ML option is 50 MB–1 GB or needs a GPU**". It appears in paste guards, agent hooks and pre-commit hooks, and it points at secrets and credentials first. PII names and addresses need real NER, which is much harder at under 1 MB.

---

## 8. Ranked shortlist (build or change)

### 1. s6-secrets v2: a calibrated verifier for rule candidates (generic secrets first)
- **User and job.** Developers and agents at commit or paste time: pre-commit and PreToolUse hooks, browser paste guards, editor extensions. It answers "is this rule hit a real credential?" in <1 ms, offline, with a threshold the user picks (F-beta presets).
- **Credible benchmark.** CredData test split (cbx protocol: 168 repos, 3,493 secrets), line or candidate level. The CC-BY issue-report test set (949 secrets / 8,132 non-secrets).
- **Baselines to beat.** Gitleaks 8.30 (span F1 0.442) and Betterleaks; detect-secrets (0.460); CredSweeper with ML (11.5 MB; 2022 F1 0.859, rerun it); cbx-secrets-detector-17M (0.805, about 67 MB, 37 ms CPU). To be credible: clearly above Gitleaks/Betterleaks as a filter on their own candidates, and within a few points of CredSweeper-ML and cbx-17M at 40–250x smaller.
- **Data.** CredData MIT/Apache/BSD repos for training, per-file licences checked, as cbx did. The Zenodo issue-report set (CC-BY-4.0, real values must be replaced). Synthetic hard negatives following GitHub's list (public keys, checksums, fixtures, docs, placeholders). Soft labels from an open LLM teacher on unlabelled public code. SecretBench needs a DPA, so do not rely on it for a published model.
- **Main risk.** CredData values are **obfuscated or randomized**, so a value-shape model may learn the obfuscation rather than real secrets. Mitigations: lean on context features (key name, path, assignment form), use token-efficiency and entropy as features, and validate on the issue-report set, which has real values. Second risk: a regex scanner with live validation already covers provider keys.
- **Why first.** The architecture is validated by GitHub (Oct 2026), CredSweeper, IssueGuard and GitGuardian. Public benchmarks and permissive data exist. The size gap to every competitor is 40x or more. The current model's two weaknesses (span, line) vanish in this framing.

### 2. s6-ear v2: end-of-turn from log-mel and prosody, distilled from Smart Turn
- **User and job.** Developers of browser or edge voice agents: "has the user finished speaking?" after a VAD pause, in about 1–3 ms in WASM, with no 8–400 MB download and no transcript wait.
- **Benchmark and baseline.** smart-turn-data-v3.2-test (CC-BY-4.0) with Speko's protocol (accuracy and false-cutoff at END-recall ≥85%). Must beat a VAD + silence timer by a wide margin (46.9%) and come close to Smart Turn v3.2 (94% / 1% FC on real English clips; 8.7 MB). Report per-language results across the 23 languages.
- **Data.** Smart Turn v3.2 train (270k samples, CC-BY-4.0). Smart Turn v3.2 as a BSD-2 teacher for soft labels.
- **Main risk.** Smart Turn is already small and fast (8 MB, ~10–65 ms CPU). Being 8x smaller may not matter to users who can afford 8 MB. Computing mel features in WASM adds cost. Audio-only models fail on TTS input.
- **Why second.** There is a clear public benchmark, an open teacher and open data, and the previous failure has a sourced explanation (no working system uses codec tokens). It is less certain than #1 because the size advantage over the incumbent is smaller.

### 3. s6-cast-pico: finish it and judge it only against TinyCast on GIFT-Eval
- **User and job.** In-browser dashboards, IoT and microcontrollers: quantile forecasts with calibrated intervals, zero-shot, no server.
- **Benchmark and baseline.** GIFT-Eval (97 configurations): beat TinyCast (0.774 MASE / 0.545 WQL at 146K) and FLAIR (0.838 / 0.587). Check Chronos-ZS and fev-bench too.
- **Data.** GIFT-Eval-Pretrain (check each constituent licence), KernelSynth-style synthetic series (generated, so licence-free).
- **Main risk.** The slot is already filled by a 146K model on a Cortex-M7, so a tie means no reason to exist. Leakage accusations if the pretraining set overlaps the test set.
- **Why third.** It is already training and the bar is public and cheap to measure. Copy TinyCast's FFT period detection and phase folding. Demand is moderate.

### 4. A first-pass injection filter for agent tool output (stage 1 of a cascade only)
- **User and job.** Agent hook authors: cheaply escalate suspicious tool output (web pages, files, MCP results) to a bigger checker, and let the rest through.
- **Benchmark and baseline.** Recall on held-out attack sources (BIPIA, llmail-inject, PINT on request) at a fixed escalation rate. FPR on NotInject. Compare with PG2-22M (87 MB), Doorman-22M, and a potion + logreg baseline.
- **Data.** deepset (Apache-2.0), jackhhao (Apache-2.0), Lakera gandalf (MIT), llmail-inject (MIT), NotInject (MIT, evaluation only).
- **Main risk.** The task is adversarial and small models learn surface triggers (2601.07185, 2609.15017). Overclaiming would hurt the project's credibility.
- **Why fourth.** There is a real size gap (70.8M-parameter "22M" models) and real demand (hooks), but it is the weakest technically and carries the most risk.

### 5. A cross-cutting change: external baselines in every pre-registration
Every s6 gate should include **potion-base-2M + logreg (or fastText) trained on the same labels at matched bytes**, plus the task's named public baseline, next to the majority class. **[Inference]** Without this, "0.480 vs majority 0.398" cannot tell a user whether to pick s6. Model2Vec is the alternative a practitioner would reach for first.

---

## 9. Things NOT to do

- **A general-purpose "decide anything" model or zero-shot intent.** Decision Bench leaders are 7–27B models at about 95%. Even the average of 41 open LLMs reaches only 0.39 zero-shot on Banking77. A 1.5M model cannot win on general understanding, only on narrow, trained, local tasks.
- **"Which model" LLM routing.** Routers have plateaued: the top 15 are within 0.23 pp on RouterBench, and kNN on frozen embeddings is in the top tier. The value is in the embeddings and labels, not a new tiny architecture.
- **Provider identification of secrets as a headline feature.** Regex plus live validation (Kingfisher, GitHub partner patterns) does this better. s6-secrets' 0.98 on a synthetic bench will not persuade anyone.
- **End-to-end span extraction of secrets or PII with a 250 KB model.** Let rules give spans. Full PII NER (names, addresses) at under 1 MB faces a 1.5B Apache-2.0 incumbent (OpenAI Privacy Filter) and needs data we do not have.
- **Training a published model on SecretBench or FPSecretBench.** Both need a signed data-protection agreement, and the right to publish a model trained on them is unclear.
- **Codec-token audio input for turn detection.** No working system uses it; the working ones use mel or prosody, or transcripts.
- **A WebGPU backend for 1 MB models.** For models this small, WASM beats WebGPU (transfer overhead).
- **Recursion or looping as the headline bet.** The HRM/TRM gains are on puzzles, and the ARC Prize analysis attributes them to deep supervision and outer refinement, not the architecture. It is fine as an internal experiment.

---

## 10. Techniques worth trying, each tied to a source

| Technique | Where it applies | Source |
|---|---|---|
| Rule candidate → context classifier (value + key name + path + bounded context) | secrets | GitHub, 13 Oct 2026 (https://github.blog/ai-and-ml/githubs-model-for-secret-detection-finding-more-secrets-with-less-noise/) ; IssueGuard (https://arxiv.org/html/2602.08072v1) |
| Hand features + char model on the last 50 chars of the value | secrets | CredSweeper (https://credsweeper.readthedocs.io/en/v1.6.4/overall_architecture.html) |
| BPE "token efficiency" (tokens per char) as a feature, replacing or alongside entropy | secrets | Betterleaks (https://www.bleepingcomputer.com/news/security/betterleaks-a-new-open-source-secrets-scanner-to-replace-gitleaks/) |
| Split by repo or by value; synthetic hard negatives (public keys, checksums, fixtures, docs, near-miss pairs) | secrets, every bench | GitHub, 13 Oct 2026 |
| Threshold presets chosen by F-beta (F-0.25…F-4) on a held-out set | every calibrated head | CredSweeper `--ml_threshold` |
| Two-stage cascade: recall-tuned cheap stage 1, expensive stage 2 | injection, agent hooks | Claude Code auto mode (https://www.digitalapplied.com/blog/claude-code-auto-mode-classifier-numbers-published) ; PG2 22M→86M advice (https://austa.ai/articles/llama-prompt-guard-2-86m-vs-22m/) |
| Computed periodicity (FFT + Fisher test, phase folding) in front of a tiny conv net; W8A8 deployment | cast | TinyCast (https://arxiv.org/html/2608.15767v1) |
| Log-mel + five prosodic features, causal 7-frame window, "hold for N frames" decision rule | ear | https://arxiv.org/html/2609.11066 |
| Distillation from an open audio teacher (Smart Turn v3.2, BSD-2) | ear | https://github.com/pipecat-ai/smart-turn |
| Static-embedding input layer distilled from a sentence encoder (Model2Vec/Tokenlearn), with vocab quantization | text heads | https://minish.ai/packages/model2vec/results/ ; https://minish.ai/blog/2026-10-12-model2vec-0100/ |
| Ternary QAT for small encoders and MLPs (expect parity, possible regularization) | all | https://arxiv.org/abs/2411.05882 |
| LUT ternary kernels (TL1/TL2) or I2_S packing; check that they map to WASM SIMD128 `i8x16.swizzle` (a LUT gather) | engine | https://arxiv.org/pdf/2502.11880 ; https://arxiv.org/pdf/2410.16144 |
| Relaxed-SIMD int8 dot product in Chrome/Firefox, SIMD128 fallback for Safari | engine (the 2x WASM gap) | https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md ; https://github.com/web-platform-dx/developer-signals/issues/205 |
| Batch candidates per call (one WASM call per file, not per line) to amortize overhead | engine, secrets | [Inference], supported by GitHub "batch of candidates" (secondary source: https://pasqualepillitteri.it/en/news/24607/github-modernbert-passwords-in-code) |
| Logit soft labels from the teacher, not verbalized confidence; temperature scaling after training | calibration | https://arxiv.org/pdf/2605.11954 ; https://arxiv.org/abs/1706.04599 |
| Calibrated early-exit branches trained jointly | early exit | https://arxiv.org/pdf/2509.08318 |
| kNN over frozen embeddings as the routing baseline | routing | https://arxiv.org/html/2606.07587 |
