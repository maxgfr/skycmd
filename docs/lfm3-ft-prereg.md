# LFM3-1.7B fine-tuné pour répondre sans réfléchir (pré-enregistrement)

**Statut : figé le 05/11/2026, avant tout job.** Aucun job Hugging Face n'a tourné, aucun essai de fumée, aucun poids n'a été entraîné. Ce fichier fixe le modèle, les données d'entraînement, l'entraînement, l'évaluation, les portes et l'ordre des jobs. Les résultats seront lus à travers ce texte.

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
