# LFM3-1.7B fine-tuné pour répondre sans réfléchir (pré-enregistrement)

**Statut : figé le 05/11/2026, avant tout job.** Aucun job Hugging Face n'a tourné, aucun essai de fumée, aucun poids n'a été entraîné. Ce fichier fixe le modèle, les données d'entraînement, l'entraînement, l'évaluation, les portes et l'ordre des jobs. Les résultats seront lus à travers ce texte.

**Closed on 19/11/2026** (§10, amendment 8): nothing was trained, and there are no weights.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 10 ;
- il précède les jobs qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un résultat de test.

## 1. Pourquoi

Le routage S1/S2 (`docs/lfm3-s1s2-prereg.md`) choisit, question par question, s'il faut réfléchir, sans toucher aux poids. Il ne corrige pas le défaut lui-même : sans réflexion, LFM3-1.7B peut partir en boucle. L'essai de fumée (`runs/lfm3-s1s2-smoke`) l'a montré : une question de GSM8K est partie en boucle jusqu'à la limite de 1 024 jetons et la réponse était fausse ; avec 256 jetons de réflexion, elle était juste. La documentation de Liquid AI le dit aussi : couper la réflexion marche de façon inconsistante sur le raisonnement.

Ce projet essaie l'autre voie : **un fine-tuning LoRA qui apprend au modèle à répondre juste et court sans réfléchir (Système 1)**, sans dégrader la réflexion (Système 2) ni le reste. Il ne remplace pas le routeur : le routeur est mesuré sur le modèle de base **et** sur le modèle fine-tuné, avec le même protocole (§5).

## 2. Modèle et environnement

- **Modèle de base** : `LiquidAI/LFM3-1.7B`, bf16, code personnalisé (`modeling_lfm3.py`, `trust_remote_code`), 24 couches, hybride convolution courte et attention (pleine ou à fenêtre glissante de 256). Sous LFM Open License v1.0.
  - révision : épinglée par `lfm3-ft.sh smoke` avant tout autre job, puis recopiée ici par un amendement daté.
  - Pour mémoire, la branche `main` du dépôt pointait le 05/11/2026 sur `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Seule la révision rendue par le smoke fait foi.
- **Calcul** : Hugging Face Jobs (`hf jobs uv run`), compte `maxgfr`, scripts uv à dépendances épinglées dans `experiments/lfm3-ft/`. Rien ne tourne en local à part `route` et `gate` (CPU).
- **Moteur d'inférence** : vLLM 0.33.0. Repli : `generate` de transformers 5.12.1 si vLLM ne charge pas LFM3 correctement.
  - vLLM 0.33.0 n'a pas d'implémentation native de LFM3 (son registre n'a que LFM2) : il passerait par son backend transformers, sans garantie pour les couches de convolution à état. Les deux moteurs ne tiennent pas dans un même environnement (vLLM 0.33 exige transformers 5.16 à 5.20, LFM3 a été publié avec 5.12.1).
  - **Règle de décision, fixée maintenant** : le smoke fait tourner les deux moteurs sur les mêmes 20 questions. vLLM est retenu s'il charge LFM3, fait tourner s1, s2-256 et s2-1024, et s'accorde avec transformers en s1 (glouton) sur au moins 0,8 des 64 premiers jetons en moyenne **et** sur au moins 0,8 des réponses. Sinon, le repli transformers est retenu.
  - Le moteur retenu est noté par un amendement et sert à **tous** les jobs suivants : la génération, et l'évaluation de la base comme du modèle fine-tuné.
  - Avec le repli : les requêtes sont groupées par lots (128, remplissage à gauche), chaque lot reçoit la graine de sa première requête, et les log-probabilités viennent d'une passe avant sur la séquence finie (distribution brute, comme celles de vLLM).
- **Dépôts, tous privés** :
  - `maxgfr/skycmd-lfm3-ft-work` (dataset) : les artefacts entre jobs (smoke, générations, journal d'entraînement, évaluations) ;
  - `maxgfr/skycmd-lfm3-1.7b-s1` (modèle) : le modèle fusionné et l'adaptateur (`adapter/`).

  Les scripts refusent d'écrire dans un dépôt public.

## 3. Données d'entraînement

Seulement des sources MIT. Aucune question de test, aucune des 600 questions de calibration du routeur.

| source | licence | révision | usage |
|---|---|---|---|
| `openai/gsm8k` (`main`, train) | MIT | `740312ad` | questions et solutions rédigées |
| `DigitalLearningGmbH/MATH-lighteval` (train) | MIT | `0530c786` | problèmes et solutions rédigées (ceux qui ont un `\boxed{}`, comme dans `s1s2.py`) |

**Exclusions** (`select_train`), avant tout tirage :
- les 600 ids de calibration (`gsm8k-train-<i>`, `math-train-<i>`, `data/lfm3/calib.jsonl`) ;
- toute question dont le texte normalisé (NFKC, minuscules, lettres et chiffres seulement) est celui d'une question de calibration, de test (`data/lfm3/test.jsonl`) ou des 500 questions MMLU-Pro du §5.

Il reste environ 14 400 questions. Les nombres exacts d'exclusions sont écrits par `gen` et `sft`.

**(a) Solutions rédigées, cibles NO-THINK.** Pour chaque question gardée, la solution du jeu de données :
- GSM8K : les annotations de calculatrice `<<...>>` retirées, la dernière ligne `#### réponse` remplacée par `\boxed{réponse}` ;
- MATH : la solution telle quelle (elle finit par un `\boxed{}`).

**(b) Auto-distillation** (`gen`).
- Le modèle de base répond à chaque question gardée **4 fois** en s2-1024, avec le forçage de budget et l'échantillonnage de `docs/lfm3-s1s2-prereg.md` §4. Graine de l'échantillon k : graine de la question (celle de `s1s2.py`) + 1 000 003 × k.
- Avec le repli transformers, seul un sous-ensemble tiré avec la graine 1234 est échantillonné : 1 000 questions par tâche. La règle est fixée maintenant, à cause du débit.
- Un échantillon est **retenu** s'il est juste (même notation que `s1s2.py`) et si sa réponse a un `\boxed{}`.
- Par question, parmi les échantillons retenus :
  - **cible NO-THINK** : la réponse d'après `</think>` la plus courte en jetons ;
  - **cible THINK** : la trace complète (réflexion, `</think>`, réponse) la plus courte en jetons, parmi les échantillons qui ont fermé leur réflexion eux-mêmes. Une trace forcée est coupée en pleine pensée : elle n'est jamais une cible THINK.
  - Les égalités vont au plus petit k.

**Format.**
- Le prompt est celui de l'inférence : le gabarit de chat de LFM3 sur « question + Solve the problem. Put the final answer in \boxed{}. ».
  - Réflexion coupée (`enable_thinking=False`, prompt fini par `<think></think>`) pour les cibles NO-THINK.
  - Réflexion active (prompt fini par `<think>`) pour les cibles THINK.
- La cible est suivie du jeton de fin de tour `<|im_end|>`.
- Les exemples de plus de 2 048 jetons sont écartés et comptés, jamais tronqués.

**Jamais utilisé** : les labels GLM / z.ai (les conditions de l'API de z.ai interdisent d'entraîner un modèle externe sur ses sorties, et ce ne sont pas des maths), et toute sortie d'un autre modèle que LFM3-1.7B lui-même.

## 4. Entraînement (`sft`)

- **LoRA** : r = 32, alpha = 64, dropout 0,05, sur toutes les projections linéaires sauf `lm_head` (liée aux embeddings).
  - D'après `modeling_lfm3.py` (révision `0dc4268b`), ce sont : `q_proj`, `k_proj`, `v_proj`, `o_proj`, la porte d'attention `gate_proj`, le MLP `w1`, `w2`, `w3`, et les projections de la convolution courte `in_proj`, `out_proj`.
  - Le smoke vérifie cette liste sur le modèle chargé. Un écart passe par un amendement.
- **Optimisation** :
  - 2 époques, lr 1e-4 en cosinus, warmup 3 % des pas ;
  - lots de 8, accumulation 4 (32 exemples par pas) ;
  - longueur max 2 048, bf16, graine 1234.
- **Mise en œuvre** : TRL `SFTTrainer` (trl 1.17.1, peft 0.21.2, transformers 5.12.1).
  - La perte porte sur la complétion seule (`completion_mask`). Le job vérifie que le collateur masque le prompt avant d'entraîner, et s'arrête sinon.
  - Pas de checkpoint intermédiaire : **le dernier est le modèle**.
- **Sortie** : l'adaptateur est fusionné dans les poids bf16 et poussé dans le dépôt privé, avec l'adaptateur dans `adapter/`. Le commit du dépôt est noté dans `sft/result.json`, et c'est lui qu'évalue `eval`.

## 5. Évaluation (`eval`, puis `route` et `gate` en local)

Le modèle de base et le modèle fine-tuné sont évalués de la même façon : même moteur, même révision de code, mêmes prompts, mêmes graines.

- **Modes** : s1, s2-256 et s2-1024, exactement ceux de `docs/lfm3-s1s2-prereg.md` §4, sur la calibration (600) et le test (1 000) de `data/lfm3`.
  - s1 garde la log-probabilité de chaque jeton, pour les traits du routeur.
  - Réponse : le dernier `\boxed{}`, à défaut le dernier nombre pour GSM8K. Même notation.
- **MMLU-Pro** : `TIGER-Lab/MMLU-Pro` (MIT), révision `b189ec76`, split test. 500 questions tirées avec la graine 1234, posées en s1 :
  - prompt : la question, les options (A) à (J), puis « Choose the correct option. Put the letter of the final answer in \boxed{}. » ;
  - réponse : la lettre du dernier `\boxed{}`, à défaut le dernier « answer is (X) ».
- **Le routeur S1/S2** (H1 à H4 de `docs/lfm3-s1s2-prereg.md` §6) est calculé pour la base **et** pour le modèle fine-tuné, avec le même protocole. La sonde et τ sont ajustés sur la calibration du modèle concerné, puis le test est lu une fois (amendement 1 de ce pré-enregistrement-là). Le code est `route` de `s1s2.py`, inchangé, pointé sur `runs/lfm3-ft/eval/<base|ft>`.

## 6. Portes (test, fine-tuné contre base, même moteur)

- **G1 (Système 1 réparé)** : précision s1 ≥ base + 0,03 **et** part des réponses s1 arrêtées par la limite de 1 024 jetons ≤ 0,5 × celle de la base.
- **G2 (Système 2 gardé)** : précision s2-1024 ≥ base − 0,01 **et** jetons de réflexion moyens en s2-1024 ≤ 0,6 × ceux de la base.
- **G3 (pas d'oubli)** : précision s1 sur les 500 questions MMLU-Pro ≥ base − 0,02.

**Verdict :**
- **« LFM3 amélioré »** si G1 et G3 tiennent ;
- G2 est rapporté à part : il dit si la réflexion raccourcit sans perte, mais ne conditionne pas le verdict.
- Ce verdict est celui du fine-tuning. Il est distinct du « LFM3 amélioré » du routeur (H4 de `docs/lfm3-s1s2-prereg.md`), que le rapport donne sous son propre nom (« H4 »).

Une évaluation incomplète (ids différents entre les deux modèles, ou moins de 1 000 / 500 questions) ne donne pas de verdict : elle est rapportée comme telle.

Le rapport (`runs/lfm3-ft/report.md`, `gate.json`) donne tout, quel que soit le résultat :
- les précisions par tâche ;
- les arrêts par limite ;
- les jetons ;
- H1 à H4 du routeur pour les deux modèles ;
- les statistiques du mélange d'entraînement et le journal de perte.

## 7. Règles

- **Pas de relance pour « voir ».** Un job interrompu (délai, panne) reprend où il s'était arrêté, par la même commande : les ids déjà écrits sont sautés. Un échec technique est noté.
- **Pas de choix de checkpoint** : seul le dernier existe.
- **Les échecs sont rapportés**, comme les succès.
- Le smoke ne lit que des questions de calibration, jamais le test.
- Aucun hyperparamètre, aucune donnée et aucun seuil ne change après la lecture d'un résultat de test. Ce qui change avant passe par un amendement.

## 8. Ordre des jobs et coût

`scripts/lfm3-ft.sh`, dans cet ordre. Avant chaque job payant, le script affiche la machine, le coût estimé et le coût maximal (au délai), puis attend une confirmation. **L'utilisateur confirme le coût avant chaque job payant.**

| étape | machine | durée estimée | coût estimé | délai (coût max) |
|---|---|---|---:|---|
| `smoke` (20 questions de calibration, deux moteurs, 2 pas de LoRA) | L4 (`l4x1`, 0,80 $/h) | 30 à 40 min | environ 0,50 $ | 1 h (0,80 $) |
| amendement : révision et moteur | — | — | — | — |
| `gen` (auto-distillation) | A100 (`a100-large`, 2,50 $/h) ou L40S | environ 2 h | environ 5 $ | 4 h (10 $) |
| `sft` | A100 | 1 à 2 h | environ 5 $ | 4 h (10 $) |
| `eval` base, puis `eval` fine-tuné | A100, un job chacun | environ 1 h 30 chacun | environ 7,50 $ | 3 h chacun (15 $) |
| `pull`, `route`, `gate` | local, CPU | quelques minutes | 0 | — |

Total estimé : **environ 18 $**, au plus 36 $. Les prix sont ceux de `hf jobs hardware` au 05/11/2026.

## 9. Licences et publication

- **Les poids dérivés restent privés** tant que l'utilisateur n'a pas donné son accord.
- S'ils sont publiés un jour, c'est sous LFM Open License v1.0, avec :
  - le texte de la licence ;
  - un `NOTICE` ;
  - la liste des fichiers modifiés (déjà écrits dans le dépôt privé par `sft`).
- Le nom ne se présente pas comme « LFM3 » : la licence ne donne aucun droit sur les marques. Le dépôt s'appelle `maxgfr/skycmd-lfm3-1.7b-s1`, « fine-tuned from LiquidAI/LFM3-1.7B ».
- L'usage commercial n'est permis que dans les limites de la licence (organisations sous 10 M$ de revenus).
- Les sources de données sont dans `DATA_LICENSES.md`.

## 10. Amendements

**Amendement 1 (09/11/2026, avant tout run : aucune génération, aucun entraînement, aucun résultat n'existe). Tout en local, sur le M5.**

- **Calcul.** Le §2 (« Calcul ») et le §8 sont remplacés : tout tourne sur le M5, rien n'est payé, aucun job Hugging Face n'est lancé (décision de l'utilisateur du 05/11/2026). `scripts/lfm3-ft.sh` passe en mode local par défaut. Les artefacts restent dans `runs/lfm3-ft/`, le modèle dans `models/lfm3-ft/`, et les deux dépôts privés du §2 ne servent plus.
- **Moteur d'inférence.** mlx-vlm 0.7.11 sur `LiquidAI/LFM3-1.7B-MLX-4bit` (révision `deb2986b`), celui de `docs/lfm3-s1s2-prereg.md` : vLLM ne tourne pas sur Mac. Les résultats sont ceux du modèle en 4 bits, et le rapport le dit. Les prompts, les modes, les graines, l'extraction et la notation ne changent pas (les fonctions de `s1s2.py` sont réutilisées).
- **Plafond de la génération.** La règle de débit du §3 (b) s'applique d'office : 1 000 questions par tâche (`hf_fallback_subset`, graine 1234), 4 échantillons chacune.
- **La LoRA, règle écrite avant le smoke local.** Elle s'entraîne sur les poids bf16 de `LiquidAI/LFM3-1.7B`, aux hyperparamètres inchangés du §4.
  1. transformers + peft + TRL sur MPS, si le smoke valide : `lora_targets_match`, le masque du prompt, 2 pas à perte finie sur un lot de la longueur maximale, la fusion et le rechargement.
  2. Sinon, l'entraîneur LoRA de mlx-vlm, aux mêmes hyperparamètres, s'il charge LFM3 et valide les mêmes points.
  3. Sinon, on s'arrête et le rapport le dit. Tout autre choix passe par un nouvel amendement.
  - **Repli mémoire, autorisé d'avance.** Si un lot de 8 à longueur maximale ne tient pas en mémoire, on passe à des lots de 4 avec accumulation 8 : toujours 32 exemples par pas, et le rapport le note.
- **Checkpoints.** Le §4 (« pas de checkpoint intermédiaire ») reste vrai pour le choix : **le dernier est le modèle**. Des points de sauvegarde techniques existent seulement pour reprendre après un arrêt du Mac. Ils ne sont jamais évalués, et une reprise est notée comme un échec technique.
- **Conversion du modèle fine-tuné.** Le modèle fusionné est converti en MLX 4 bits (`mlx_vlm.convert -q --q-bits 4 --q-mode affine`) pour être évalué par le même moteur que la base.
  - **Règle écrite avant le smoke.** Le smoke convertit aussi la base bf16 par la même commande, et compare cette conversion à la base publiée, en s1 (glouton), sur 20 questions de calibration.
  - Si l'accord atteint au moins 0,8 sur les 64 premiers jetons en moyenne **et** au moins 0,8 sur les réponses (les seuils du §2), la base publiée sert de référence.
  - Sinon, **la base est réévaluée avec la conversion locale**, pour que les deux modèles aient suivi la même chaîne.
- **Évaluation de la base.** Si la base publiée sert de référence, ses générations sont celles de `docs/lfm3-s1s2-prereg.md` (`runs/lfm3-s1s2`, mêmes prompts, graines et moteur), copiées telles quelles dans `runs/lfm3-ft/eval/base`. Seules les 500 questions MMLU-Pro en s1 sont générées en plus. Le routeur (`route`) calcule sur ces fichiers ce qu'il a déjà calculé pour le §6 du TODO : aucun nouveau choix n'en dépend.
- **La révision.** Le smoke local résout la révision de `LiquidAI/LFM3-1.7B` et la rend dans `runs/lfm3-ft/smoke/report.json`. Un amendement daté la recopie, avec la méthode LoRA retenue, avant `gen`.
- **Le mode bridé.** Le run se fait en mode bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), comme l'amendement 5 de `docs/s6-prereg.md` : une pause après chaque échantillon et chaque pas de LoRA. Le temps de génération et le nombre de jetons (`seconds`, `tokens`) n'en dépendent pas.
- **Ce qui ne change pas.** Le modèle de base, les données, les exclusions, les hyperparamètres de la LoRA, les portes G1 à G3, le verdict, et le §9 : les poids dérivés restent privés tant que l'utilisateur n'a pas donné son accord.

**Amendment 2 (12/11/2026, before any run: no generation, no training and no result exist). Two brake modes.**

- **Change.** As for the s6 runs (`docs/s6-prereg.md`, amendment 6) and `docs/lfm3-s1s2-prereg.md` (amendment 3), the brake has two modes: `0.5` and `45:15` while the Mac is in use, `0.85` and `50:10` while it is left alone, back to the first under thermal pressure. `scripts/auto-brake.sh` writes the mode to `runs/brake`; the generation and the LoRA callback read it again at every rest (`SKY_BRAKE_FILE`).
- **Why no result depends on it.** The brake sleeps between two samples and between two optimizer steps. The number of LoRA steps, the batches and the generation settings are fixed in steps and tokens, not in minutes.
- **What does not change.** Everything else, the three gates and section 9 included: the derived weights stay private until the owner agrees.

**Amendment 3 (13/11/2026, before any run). A third brake mode.** At the owner's request, `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and nothing else works on it, and the memory cap needs four readings in a row above 13 GB, as in amendment 7 of `docs/s6-gen-prereg.md`. The brake only sleeps between two samples and between two optimizer steps. Nothing else changes.

**Amendment 4 (13/11/2026, before any run). The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.

**Amendment 5 (16/11/2026, before any run: no smoke, no generation, no training and no result of this pre-registration exist). A shorter plan, at the owner's request ("it is too long"): the adapter is System 1, the base stays System 2.**

- **Why the router failed, in one line of arithmetic.** A cascade pays the first answer on every question: its cost against always thinking is `1/ρ + w`, with ρ the cost of a System 2 answer over a System 1 answer and w the share of questions sent on to System 2 (simple arithmetic on the cost accounting of cascades such as AutoMix, arXiv 2310.12963, where the small model is paid on every query). On the test of `docs/lfm3-s1s2-prereg.md` System 1 used 559 tokens and System 2 958: ρ = 1.71 and `1/ρ` = 0.58, above the 0.5 that H3 asked, whatever the router did. The router could not pass with this System 1. What can change is ρ: a System 1 that answers in a few lines.
- **What is trained: part (a) of §3 only.** The written solutions of GSM8K and MATH as NO-THINK targets, capped at **3,000 questions per task** drawn with seed 1234 among the questions §3 keeps (about 14,400), **one epoch** (about 188 steps of 32 examples). Part (b), the self-distillation (8,000 generations in s2-1024), is **not run**, so no THINK target exists. Everything else of §4 stands (LoRA r 32 on the same projections, lr 1e-4, the prompt masked, the last step is the model). Shortening a model's reasoning with a small LoRA is known to work at larger sizes (TokenSkip, arXiv 2502.12067: on GSM8K with Qwen2.5-14B-Instruct, reasoning tokens go from 313 to 181 for a drop under 0.4%; nothing is published at 1.7B); one model holding a fast and a slow mode is Dualformer's result on planning tasks (arXiv 2410.09918). Here the two modes are kept apart on purpose.
- **The adapter is System 1; the base is System 2.** The fine-tuned weights answer only in s1. In s2-256 and s2-1024 the system is the base model, untouched: its generations are those of `runs/lfm3-s1s2` (same prompts, seeds and engine), as amendment 1 allows when its conversion check holds. Consequences:
  - **G2 is not read**: System 2 is the base model itself, so it cannot be degraded, and nothing shortens its reasoning either. The report says "G2: not read (System 2 is the base)".
  - **`eval` of the fine-tuned model is s1 only**, on the calibration (600), the test (1,000) and the 500 MMLU-Pro questions. The base's s1 on MMLU-Pro is run once (500 questions); its calibration and test generations are reused.
  - **The router is read once more, on the pair**: `route` of `s1s2.py`, unchanged, on a folder that holds the fine-tuned s1 and the base's s2. Its probe and τ are fitted on the calibration of that pair, then the test is read once. H2, H3 and H4 keep the definitions and the bars of `docs/lfm3-s1s2-prereg.md` §6, and are reported under the name "the pair" next to the base's failed reading. It is not re-read on the base and not read on a fine-tuned System 2, which does not exist.
- **Gates.** G1 and G3 are unchanged, and so is the verdict ("LFM3 improved" if G1 and G3 hold). **Reported with them:** H2, H3 and H4 of the pair, the mean tokens of s1 before and after, `1/ρ` and w.
- **What a result would mean.** G1 and G3 hold and the pair passes H2 and H3: a 1.7B model answers as well as when it always thinks, for half the tokens or less, with one adapter and one probe. G1 holds and H3 fails: System 1 is better but still not cheap enough, or the probe sends too much on. G1 fails: written solutions as targets, at this quantity, do not repair System 1; with 6,000 examples instead of about 14,400 and no self-distillation, the failure cannot be told apart from too little data, and the report says so.
- **Cost.** The smoke (about half an hour), the fine-tuning (one to two hours), the conversion, about 2,100 short s1 answers of the fine-tuned model and 500 of the base (about two hours): **about five working hours of GPU instead of fifty or more**.
- **What does not change:** the base model and its revision, the exclusions of §3, the smoke and its rules (amendment 1), the LoRA method chosen by the smoke, the brake modes, the test read once, and §9: the derived weights stay private until the owner agrees.

**Amendment 6 (17/11/2026, before any run: no smoke, no training and no result of this pre-registration exist). Three readings that amendment 5 left open, fixed after a review of the code.**

- **w, the share sent on to System 2.** It is the share of the test questions whose probe score is under τ, whether System 2 then answers or not: these are the questions whose System 2 tokens are paid. The router keeps the System 1 answer when System 2 gives none, and its own "System 1 share" counts those questions as System 1: on the base's reading of 15/11 that share is 0.842 while 275 of the 1,000 questions went to System 2 (w = 0.275, not 0.158). The cascade cost `1/ρ + w` is reported with this w.
- **The budget of 1/ρ.** 1/ρ is the mean System 1 tokens of the fine-tuned model over the mean tokens of the base at the budget the pair's router chooses on its calibration (the rule of `docs/lfm3-s1s2-prereg.md`: the best always-thinking accuracy on calibration; 1,024 on the base's reading), on the test.
- **If the conversion check of amendment 1 says "converted"** (the local conversion of the base disagrees with the published 4-bit file on more than 4 of the 20 questions): the base is evaluated again **in s1 only**, with the local conversion, on the calibration, the test and the 500 MMLU-Pro questions, so that G1 and G3 compare the fine-tuned model with a base on the same conversion (about 2,100 answers, two to three working hours more). System 2 of the pair stays the published base and its generations of `runs/lfm3-s1s2`: it is the untouched model that a user would call, and no System 2 generation is run again. The report says which conversion each number comes from. Amendment 1's full re-evaluation in the three modes is not run.
- **Two guards added by the review, which change no number:** the pair is refused unless each of its six files holds exactly the questions of its split, once each, in its own mode; a fine-tuning that has ended is never run again by a second launch of the step (its result file is the completion mark).
- **What does not change:** everything else of amendment 5.

**Amendment 7 (18/11/2026, after the smoke, before any other step: no training and no evaluation exist, and the smoke read calibration questions only). The pinned revision, the versions, and the smoke's verdict: no LoRA method fits the M5 within its memory bound, so amendment 1's rule 3 applies and nothing is trained.**

- **The first failure of 18/11.** The first smoke stopped at once: `modeling_lfm3.py`, the base model's custom code at revision `1da06bec`, imports `use_kernel_func_from_hub_with_fallback` from `transformers.integrations`, and transformers 5.12.1, the version §4 pinned, has no such name (`runs/all/queue-30-lfm3-smoke.log`).
- **The revision: `1da06bec6a819da10b54c67e5c90972b46d81871`.** It is the `main` of `LiquidAI/LFM3-1.7B` that the smoke resolved, and the last of the repository's seven revisions (27/10 to 09/11/2026; Hub API, 18/11).
  - The seven are identical except for their model card: the custom code (`modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`), `config.json`, the tokenizer, the chat template, `LICENSE` and the weights (one `model.safetensors`, 3,457,635,448 bytes, the same file at every revision).
  - So no older revision runs under 5.12.1: the rule written for this case (keep the versions, pin the last revision whose code imports under them) had no candidate. The revision was pinned and the versions moved.
  - The `0dc4268b` of §2 and §4 has the same code and weights. The licence is the LFM Open License v1.0, unchanged at every revision.
  - The published `LiquidAI/LFM3-1.7B-MLX-4bit` (`deb2986b`, the base of `runs/lfm3-s1s2`) names `LiquidAI/LFM3-1.7B` as its base without a revision. Since the bf16 weights never changed, it can only come from these weights, and the conversion check below agrees.
- **The versions: transformers 5.18.0 instead of 5.12.1, nothing else.** 5.12.1 also lacks `force_accelerate_hooks` and `create_recurrent_attention_mask`, which the custom code imports.
  - 5.15.0 is the first release that exports all three names. But from 5.15.0 to 5.17.0 transformers renames the `conv` layers of custom-code configurations to `linear_attention`, and LFM3's forward then fails (`KeyError`). These releases were checked on a reduced LFM3 built from the custom code.
  - 5.18.0 is the lowest release that works. With torch 2.14.0, peft 0.21.2, trl 1.17.1 and accelerate 1.15.0 unchanged (tokenizers moves to 0.23.1 or later with it), it passed three checks:
    - the reduced model's forward and loss;
    - the full model loaded on CPU (1,728,801,536 parameters) with the chat template, thinking on and off;
    - a coherent greedy answer.
  - transformers only includes LFM3 itself from 5.23.0. The mlx-vlm environment is unpinned, as in `scripts/lfm3-s1s2.sh`, and resolved transformers 5.23.0, mlx 0.32.4 and mlx-vlm 0.7.11.
  - `scripts/lfm3-ft.sh` changes its `py_lora` line and `BASE_REV_PINNED`. MODE=hf, retired by amendment 1, keeps its first pins: it could not load this revision, and it is not used.
- **The memory bound, set before the run that decides (owner's instruction of 18/11/2026).** Each batch size is tried in its own process, which `experiments/lfm3-ft/local.py` (`guarded`) stops at once when either of two limits is passed:
  - the swap in use passes 3 GB;
  - the 1-minute load passes 8.

  Such a batch "does not fit in memory" in amendment 1's sense, and its fallback follows: batches of 4 with accumulation 8.
  - An attempt starts only once the swap is back under 2 GB, and waits at most 20 minutes for it. macOS gives back the swap of the other applications slowly.
  - MPS is capped at its recommended working set (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`: 11.84 GiB on this Mac). Without the cap, a batch that does not fit swaps instead of raising.
  - The bound reads the whole machine, the owner's applications included: it says what fits on this Mac while it is used, not what a free 16 GB Mac could hold.
- **The runs of 18/11, in order** (logs in `runs/lfm3-ft/smoke-attempts/`):
  1. **18:40.** peft at batch 8 swapped while a macOS update was being staged on the same volume, and the free disk fell to 183 MB. The run was stopped. From then on the smoke deletes the smoke's merged weights once they have reloaded, keeping their size and sha256.
  2. **18:45.** MPS refused every allocation: the default low watermark (1.4) was above the new cap. Both are now set.
  3. **18:45 to 19:07, with the MPS cap only.**
     - peft at batch 8 ran out of MPS memory in its first step: 11.62 GiB allocated, 11.84 allowed.
     - At batch 4, the first step ended: 59 s, MPS driver peak 10.1 GB. The second, with the optimizer's state, ran out of memory: 11.54 GiB.
     - The swap reached 12.7 GB.
  4. **19:07.** mlx-vlm at batch 8 (rule 2) has no hard cap. The swap went from 8 to 12.3 GB in a minute, and the machine, also loaded by other work, restarted.
  5. **20:10.** With a first guard (swap growth over 6 GB), peft at batch 8 pushed the swap from 0 to 4.7 GB in 15 s. The owner's work lagged, and the run was stopped.
  6. **21:11, with the bound above.**
     - Batch 8 was stopped at 3.4 GB of swap in 13 s.
     - Batch 4 was stopped at once: the swap had not come back down.
     - The parent process also held a copy of the model. The module names are now read from the custom code built on the meta device, which loads no weights, and an attempt waits for the swap to settle.
     - The run was stopped and does not count.
  7. **21:16 to 21:39, the run that decides** (`runs/lfm3-ft/smoke.log`, `smoke/report.json`, `smoke/lora.json`, `smoke/mlx-lora.json`, `smoke-watch.log`).
- **What the smoke of 21:16 measured.**
  - The revision above; the chat template, thinking on and off.
  - The LoRA names are exactly §4's ten (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `w1`, `w2`, `w3`, `in_proj`, `out_proj`, on the loaded custom code; `lora_targets_match`).
  - **peft on MPS** (rule 1). Each attempt was stopped by the bound 14 to 19 s after it started, before any optimizer step could end (one step took 59 s in run 3):
    - batch 8: swap 1.87 to 3.51 GB in 14 s, free memory down to 4 %;
    - batch 4: after 510 s of waiting for the swap, 1.91 to 3.44 GB in 19 s, 7 %.
  - **The LoRA trainer of mlx-vlm** (rule 2, same hyperparameters, `smoke-mlx-lora`). Both attempts were stopped the same way:
    - batch 8: after 90 s, 1.98 to 3.43 GB in 15 s, 16 %;
    - batch 4: after 350 s, 2.05 to 3.08 GB in 14 s, 2 %.
  - The 1-minute load stayed under 2.9 during the attempts and under 3.7 over the whole run. No optimizer step ended, so the prompt mask, two finite losses, the merge and the reload were not reached by either method.
  - **The conversion check** passed: 1.0 agreement on the first 64 tokens and 1.0 on the answers, over 20 calibration questions in s1 (bar: 0.8 and 0.8). The local conversion is `mlx_vlm.convert -q --q-bits 4 --q-mode affine` of the bf16 weights at `1da06bec`.
    - This is the first case of amendment 6: the published 4-bit base is the reference, and its generations of `runs/lfm3-s1s2` are reused. The same check gave 1.0 and 1.0 at 19:05 (run 3).
  - **Speeds of the published base** on the M5, without the brake's rests: s1 at 110 tokens/s (20 questions), s2-256 at 98 tokens/s (4 questions). Run 3 gave 99 and 93.
  - **Cost of the run of 21:16.**
    - 23 minutes in all, of which 16 waiting for the swap to settle, about 1 minute in the four attempts and about 4 in the conversion check.
    - The battery went from 80 % to 71 % (the Mac was on battery and in use).
- **The decision, by amendment 1's rule: rule 3.**
  - Neither peft on MPS nor mlx-vlm validated the points of the rule within the memory bound, at 8 or at 4 examples per batch. So nothing is trained, and `sft`, `eval`, `route` and `gate` are not run.
  - `smoke/report.json` says `lora_method: null`. `scripts/lfm3-ft.sh sft` refuses to start (`lora_settings`).
  - Any other choice needs a new amendment written before it runs: a looser bound, a Mac left free, a shorter maximum length, smaller batches with more accumulation, or remote compute.
  - The report says G1 and G3 are not read.
- **Working minutes that remain, if a later amendment selects a method.** These are estimates, with the brake's rests not counted:
  - `sft` cannot be estimated from this smoke, because no step ended under the bound. Run 3, without the bound, did one step of 4 examples of 2,048 tokens in 59 s with peft. 188 steps of 8 such batches would take at most about 25 hours, and much less at the real lengths of the examples.
  - `eval`: the fine-tuned model in s1 on about 2,100 questions, about 3 hours at 110 tokens/s and the base's 559 tokens per answer, less if the answers are shorter. The base in s1 on the 500 MMLU-Pro questions, about 45 minutes.
  - `route` and `gate`: a few minutes of CPU.
- **Deviations from what the earlier amendments said the smoke would do.**
  - Rule 2 (mlx-vlm) was not in the smoke's code. It was written for this smoke (`smoke-mlx-lora`) with the hyperparameters of §4:
    - LoRA r 32, alpha 64, dropout 0.05, on the same ten projections;
    - AdamW at lr 1e-4, cosine, warmup 3 %, no weight decay, gradients clipped at norm 1.0;
    - completion-only loss, gradient checkpointing, length 2,048.
  - Each batch size runs in a child process under the bound above.
  - A step's time is measured without the brake's rests.
  - The module names are read on the meta device.
  - The smoke's merged weights are deleted once reloaded (not reached on 18/11).
  - transformers 5.18.0 instead of 5.12.1.
  - `smoke-mlx` generates without the brake's rests, as before.
- **What does not change:** everything else: the base model, the data, the hyperparameters, the gates, the verdict, the test read once, and §9.

**Amendment 8 (19/11/2026, after amendment 7, before any other step: no training, no evaluation and no gate reading exist). The fine-tune is closed.**

- **The decision.** The owner decided on 19/11/2026 not to rent compute for this fine-tune ("no LFM3 if it does not fit"). Amendment 7 found that no LoRA method allowed by amendment 1 (peft on MPS, then the LoRA trainer of mlx-vlm) fits this 16 GB Mac under the memory bound set before the run that decides. Every other option that amendment 7 lists (a looser bound, a Mac left free, a shorter maximum length, smaller batches with more accumulation, or remote compute) is declined. So the pre-registration is closed here.
- **What exists.** Nothing was trained: there are no weights, no adapter and no fine-tuned model. `sft`, `eval`, `route` and `gate` never ran, and G1 to G3 are not read and never will be. What exists is the smoke of 18/11 and its files (`runs/lfm3-ft/smoke/report.json`, `runs/lfm3-ft/smoke.log`, `runs/lfm3-ft/smoke-attempts/`), which record why nothing was trained.
- **What stays published, unchanged.** The System 1 / System 2 router of `docs/lfm3-s1s2-prereg.md` (inference only, failed: H3, 1,032 tokens against at most 0.5 × 958), its result files `runs/lfm3-s1s2/`, and the data set `maxgfr/skycmd-lfm3-math`. This closing changes no number and no verdict of that router.
- **What does not change:** the text of §1 to §9 and of amendments 1 to 7 stays as the record of what was planned and tried.
