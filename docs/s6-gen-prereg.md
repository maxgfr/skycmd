# s6-nano et s6-micro : pré-enregistrement (le s6 généraliste)

**Statut : figé le 29/10/2026, avant tout entraînement de s6-nano et de s6-micro.** Aucun run s6 n'a tourné : ni le pilote pico, ni l'ablation. Ce fichier fixe les modèles, les données, le pilote de learning rate de chaque taille, les runs, le choix du checkpoint, l'évaluation et les portes de publication. Les résultats seront lus à travers ce texte, et pas à travers une version écrite après coup.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 8 ;
- il précède les runs qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un résultat qu'il départage.

## 1. Pourquoi

Le s6 devient un modèle généraliste : les 4 jeux, et des questions à choix en texte libre. Il remplace le modèle final s6-pico (`docs/s6-prereg.md`, amendement 2). Le pilote et l'ablation au format pico restent : ils sont rapides, et ils ne servent qu'à choisir la recette.

Les deux tentatives précédentes sur `typed_decisions_bench` sont restées sous la classe majoritaire (**0,398**) :

- **s3-pico** (445 k paramètres) fait **0,382**. Avec 1 024 jetons et des fenêtres de 64/96, 96,8 % des séquences sont tronquées (tutoriel, chapitre 12).
- **s4-micro** (5,6 M) fait **0,343**. Sa cause identifiée est un pré-entraînement de 30 min sur 150 Mo seulement (tutoriel, chapitre 22).

s6-nano et s6-micro ont des fenêtres larges, un vocabulaire plus grand, 300 Mo de texte (`data/text-open` puis `data/text`) et des heures de pré-entraînement.

**Pas de professeur** (remplacé par l'amendement 1 : le professeur open source GLM-5.3-flash, à distance). Aucun fichier professeur n'est sur disque (ni `data/s4`, ni `data/s5`, ni les professeurs laya ou oscar-150m). Cette v1 est donc entraînée sans distillation, et ce choix est fait avant le run. Les cibles douces déjà présentes dans les jeux de données (tasksource, procedural, calibrated…) suffisent.

## 2. Les modèles

Les tailles et les nombres de paramètres ont été fixés dans `s6_parameter_counts` (`crates/sky-train/src/s2/model.rs`) avant tout run.

| | s6-nano | s6-micro |
|---|---|---|
| config | `configs/s6-nano.json` | `configs/s6-micro.json` |
| paramètres (décision / pré-entraînement) | **1 569 457** / 778 065 | **5 849 601** / 2 960 769 |
| vocabulaire | 2 048 (`data/tok/tok-s6-2048.json`) | 4 096 (`data/tok/tok-s6-4096.json`) |
| largeur d / ff | 144 / 288 | 256 / 768 |
| blocs × boucles (état) | 2 × 2, sorties après 1, 2, 3, 4 | 2 × 3, sorties après 2, 4, 6 |
| têtes | 2 (dimension 72) | 4 (dimension 64) |
| fenêtres état / tête | 256 / 128, puis **512 / 160** (amendement 2) | 384 / 128 |

s6-micro a la forme de s4-micro (5 587 201 paramètres) plus les lignes de sonde Parallax. Les deux configs ont `fuse_numbers: true`.

**La recette vient de l'ablation pico, appliquée mécaniquement.** Avant le pilote de la première taille, un commit règle les deux configs et les variables de `scripts/train-s6.sh` sur les règles d'adoption de `docs/s6-prereg.md` §4 :

- `parallax` reste `true` si la règle 2 l'adopte, sinon il passe à `false` ;
- `"xsa": true` est ajouté si la règle 3 adopte XSA (il n'ajoute aucun tenseur) ;
- `OPT` (Muon v2 ou `--no-muon`, `--wd-mode soda`, `--ns-polish 0`), `PRETRAIN` (`--curriculum 0.1`) et `DECIDE` (`--exit-kd 0.5`, `--ema 0.999`, `--qat-grad-a 0.21`) prennent les changements adoptés ;
- `LR_SCALE` est l'échelle du pilote pico pour l'optimiseur retenu.

Si `parallax` passe à `false`, le même commit corrige les nombres de paramètres du test. Rien d'autre ne change dans la recette.

## 3. Données

Toutes les données sortent de `scripts/prepare-data.sh`, lancé une fois avant le pilote pico (`docs/s6-prereg.md` §1). Il écrit aussi les deux tokenizers et les jeux zéro-shot de MASSIVE (`fetch-commands --only massive`, puis `s4-corpus --only-heldout`). Rien n'est régénéré entre les pilotes et les runs des deux tailles.

- **Tokenizers.** `sky-data train-tokenizer --vocab N --num` sur `data/text`, `data/text-open`, `data/decisions` et `data/games`.
- **Pré-entraînement MLM.** `data/text-open` puis `data/text`, plus 10 % de curriculum numérique (`data/numeric`), en séquences de 256 jetons.
- **Décisions.** `data/decisions,data/games,data/numeric` avec les poids `decisions=0.55,games=0.35,numeric=0.10`. `data/decisions` compte 831 387 questions dans 29 jeux. Seuls les splits `train` sont entraînés et seuls les splits `val` servent à choisir.
- **DAgger.** Poids `games=0.4,rN=0.25,decisions=0.3,numeric=0.05`. Garder 30 % de décisions évite d'oublier la langue en apprenant à jouer.
- **Ensembles de test, jamais lus avant l'évaluation finale :**
  - `typed_decisions_bench` : tout est en `split: test` ;
  - MASSIVE en et fr (`data/s4/heldout/zs_massive_*.jsonl`) : le split test, pas du tout dans `data/decisions`, donc **zéro-shot** ;
  - notam (`notam:deel_subject`, `notam:notam_multi`) : tout est en test, donc jamais entraîné ;
  - When2Call : son split test. Ses splits train et val sont entraînés : c'est un test de la même distribution.
- **Banking77 est dans la distribution.** Il est maintenant dans `data/decisions`, en train et val seulement, donc s6 s'entraîne dessus. `zs_banking77` (tiré de ces mêmes records) est rapporté à part, avec la mention « entraîné », et n'entre dans aucune moyenne ni aucune porte.
- Les données enseignant GLM (`glm_orders`, `paraphrased`) : leur présence ou leur absence est notée dans le rapport.

## 4. Pilote de learning rate par taille (`SIZE=… scripts/train-s6.sh lr`)

- **Pourquoi.** L'optimum de Muon bouge avec la largeur (`docs/s6-prereg.md` §2), plus que celui d'Adam, même si la perte est plate autour (arXiv 2512.22768). Le taux réglé à d = 96 n'est pas forcément le bon à d = 144 ou 256.
- **Le centre.** Avec l'échelle de Muon v2 (RMS des mises à jour alignée sur AdamW, 0,2·√max(A, B)), un taux fixe fait croître le pas spectral comme √d. La condition spectrale de μP demande donc un taux en √(d_pico / d) (arXiv 2603.00541 ; « Remark #1: On RMS matched Muon », 2026). Le centre de la grille est le taux pico déplacé ainsi : **×0,816 pour s6-nano** (√(96/144)) et **×0,612 pour s6-micro** (√(96/256)).
- **Runs.** ×0,5, ×1 et ×2 du centre, soit `LR_SCALE` × √(96/d) × l'échelle × le taux par défaut de chaque phase (3e-3 en pré-entraînement, 1e-3 en décision). Le taux pico non déplacé (×1,22 du centre pour s6-nano, ×1,63 pour s6-micro) reste encadré par les points ×1 et ×2. Chaque run fait 20 min de pré-entraînement et 20 min de décision, graine 0. Trois runs par taille, environ 2 h.
- **Choix.** La précision ternaire finale de la dernière sortie sur `val` (la dernière validation de `log.jsonl`, en mode `ema-ternary` si `--ema` est adopté). On garde ×1 (le centre), sauf si une autre échelle le bat d'au moins **0,01**. Sur 4 000 questions de validation, un écart plus petit ne se distingue pas du bruit : la règle penche vers le centre. Le choix est écrit dans `runs/s6-<taille>-lr/summary.json`, et `train` le lit.
- **Bord de grille.** Pas d'extension : un choix à ×0,5 ou ×2 est appliqué tel quel.

## 5. Les runs (`SIZE=nano|micro scripts/train-s6.sh`)

| | s6-nano | s6-micro |
|---|---:|---:|
| pré-entraînement MLM (batch 64, séquences de 256) | 90 min | 180 min |
| décision (QAT après 10 % en flottant, loop dropout 0,25) | 90 min, batch 128 | 180 min, batch 64 |
| DAgger : 3 rounds de 15 000 décisions, oracle 0,2 → 0,1 → 0,05 | 15 min par round | 20 min par round |

- Le pré-entraînement est budgété en temps, comme les modèles précédents. Il se fait en flottant : la QAT ternaire ne commence qu'en décision, après 10 % d'échauffement, comme le recommandent les travaux sur le passage 16 bits → 1,58 bit (arXiv 2502.11895). Un encodeur MLM demande bien plus de jetons par paramètre qu'un décodeur (EMNLP 2025, « Training compute-optimal transformer encoder models »). Le rapport donne donc les jetons vus et leur nombre par paramètre, avec environ 200 comme repère indicatif ; ce n'est pas une porte. Les batchs de décision sont groupés par longueur (`--length-pool 16`) : sinon, les fenêtres larges paieraient surtout du remplissage.
- **Seuils de sortie.** Contrôle conformal du risque (α = 0,02) sur le dernier round (`sky-arena --calibrate-crc`), dans une copie `-crc`. Ce jeu de seuils est gardé si, à ces seuils, l'arena reste à moins de 2 % de la pleine profondeur sans crash ni veto de plus. Sinon, la copie est supprimée et le modèle reste à pleine profondeur.
- **Checkpoint.** `last`, ou `last-ema` si `--ema` est adopté, calibré par `s2-calibrate` sur `val` puis exporté en ternaire. Jamais un checkpoint intermédiaire.
- Une seule fois par taille. Seul un échec technique (crash, run incomplet) justifie de relancer, et il est noté dans le rapport.

## 6. Évaluation et portes (`SIZE=… scripts/train-s6.sh eval`, `scripts/s6-gate.py`)

**Le modèle évalué** est le GGUF ternaire du dernier round DAgger : à pleine profondeur (τ = 1), et à ses seuils conformaux s'ils ont été gardés.

**Langue.** `sky-eval` sur `typed_decisions_bench`, `zs_massive_en`, `zs_massive_fr`, le test de When2Call et notam, avec au plus 2 000 questions par source (`--max 2000`, les premières). Les mesures :

- la précision par source ;
- l'**ECE sur toutes ces sources réunies** (`all`) ;
- la profondeur moyenne et la latence native.

s3-pico, s4-micro et s5-lite publiés passent sur les mêmes records, à τ = 1.

**Jeux.** Arena, 4 jeux × 30 épisodes, aux mêmes graines pour le modèle et pour s3-pico publié (`site/public/models/s3-pico.gguf`), à τ = 1.

**Aussi rapporté :**

- la latence WASM par décision (`parity-node.mjs` sur 200 questions du bench, avec la parité native/WASM) ;
- la part des records tronqués (`sky-train s2-truncation` : les 500 premiers records de chaque source, état et tête) ;
- `zs_banking77`, dans la distribution.

**Portes de publication.** Chaque porte exige **aussi** deux conditions : aucun jeu dont le score moyen d'arena tombe plus de 5 % sous s3-pico, et un **ECE ≤ 0,08**.

| | précision `typed_decisions_bench` (ternaire) | objectif annoncé |
|---|---|---|
| s6-nano | **> 0,398** (la classe majoritaire) | — |
| s6-micro | **≥ 0,45** | **0,50**, le niveau du professeur Qwen3.5-4B de s4 |

- La porte est lue à τ = 1. Le modèle est publié à ses seuils conformaux si la porte tient aussi à ces seuils. Si elle ne tient qu'à τ = 1, il est publié à pleine profondeur. Si elle ne tient pas à τ = 1, il n'est pas publié, et le rapport le dit.
- Les deux tailles sont jugées séparément : l'échec de l'une ne bloque pas l'autre.
- Pour s6-micro, le rapport dit sans ambiguïté si l'objectif de 0,50 est atteint.

**Le rapport, quel que soit le résultat :**

- tous les chiffres ci-dessus ;
- la comparaison sur le bench avec s3-pico (0,382), s4-micro (0,343), oscar-1-17m (0,6775) et s5-lite (0,696) ;
- la courbe nano → micro : précision, ECE, latence, taille ;
- les jetons vus en pré-entraînement, leur nombre par paramètre, l'époque atteinte et la perte MLM finale.

**Seuils de confiance.** α = 0,02 est dans la plage des travaux sur la sortie anticipée sous contrôle du risque (CALM, NeurIPS 2022 ; « Fast yet Safe », arXiv 2405.20915). ECE ≤ 0,08 est un garde-fou, pas une cible : les réseaux non calibrés sont entre 4 et 10 %, et un encodeur calibré par température descend bien plus bas. Le rapport donne l'ECE exacte.

## 7. Interdits

- Relancer un run pour « voir ».
- Choisir sur un checkpoint intermédiaire.
- Lire un split `test` avant l'évaluation finale de la taille concernée.
- Modifier les données, les budgets ou les poids entre le pilote d'une taille et son run.
- Changer la recette héritée de l'ablation pour une seule taille.

## 8. Amendements

**Amendement 1 (29/10/2026, avant tout run s6 : ni le pilote pico, ni l'ablation, ni aucune taille n'a tourné, aucun résultat n'a été lu). Un professeur open source, GLM-5.3-flash, sur toutes les données de décision.**

- **Changement.** Le §1 disait « pas de professeur ». s6-nano et s6-micro apprennent maintenant aussi des probabilités d'un professeur : **GLM-5.3-flash**, un modèle **open source** de Z.ai.
  - Ses poids sont publiés sous licence **MIT** ([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 321B paramètres en mixture d'experts, dont 18B actifs).
  - Il est appelé à distance sur l'API de z.ai (protocole Anthropic Messages, configuré par `ccs`). Ses poids étant ouverts, les labels restent reproductibles hors de cette API.
  - **Records envoyés** (`sky-data teacher label-decisions`, `scripts/distill-glm.sh`) : **tous** les records `train` de `data/decisions`, c'est-à-dire tous les jeux Hugging Face de décision, entre 2 et 12 options. Les cibles douces sont incluses.
  - Ne sont jamais envoyés : `typed_decisions_bench`, notam, banking77, les records déjà générés par GLM (`glm_orders`, `para:*`), et aucun split `val` ou `test`.
  - Le 29/10/2026, cela fait **747 198 records**, envoyés dans un ordre tiré au hasard (graine 1234), en 37 360 appels de 20, par 4 workers parallèles : au-delà, z.ai freine et les appels partent en relances. Le 29/10/2026, le débit mesuré était d'environ 2,5 labels/s quel que soit le nombre de workers, soit environ 80 h de labellisation réparties sur les fenêtres d'usage de 5 h de l'abonnement z.ai (`scripts/distill-glm.sh loop`).
  - **La question posée** : une probabilité par option, dans l'ordre donné, somme 1 ; température 0,2 ; état coupé à 2 000 caractères. Une réponse mal formée (mauvais nombre d'options, valeur négative, somme nulle) est écartée.
  - **Le mélange** (`sky-data teacher mix-labels`) : un label n'est gardé que si l'argmax du professeur est la réponse or (le filtre d'accord de s5). La cible devient `0,5 · or + 0,5 · professeur`, dans `data/distill/glm.jsonl` (`src = glm:<source>`, split `train`).
  - **Poids de décision** : `decisions=0.25,distill=0.30,games=0.35,numeric=0.10`. **Poids DAgger** : `games=0.4,rN=0.25,decisions=0.1,distill=0.2,numeric=0.05`, soit toujours 30 % de décisions.
- **Pourquoi.**
  - s4-micro a échoué sans pré-entraînement suffisant et sans bon professeur.
  - Les meilleurs professeurs locaux (oscar-1-150m, 0,7755 sur le bench de s5 ; laya-typed-decisions, 0,7665) tourneraient des heures sur la machine, qui chauffe trop. GLM tourne à distance.
  - Essai du 29/10/2026 (2 appels, 20 records, hors `data/`) : 20 réponses valides sur 20, et 15 d'accord avec l'or.
- **Règles.**
  - Les labels sont produits une fois, avant le pilote de s6-nano, et jamais régénérés. Un run interrompu (quota, réseau) reprend là où il s'est arrêté : même sélection, mêmes ids.
  - Les runs attendent que le fichier de labels couvre au moins 95 % de la sélection. Sinon, un nouvel amendement daté décide, avant tout run.
  - Le rapport donne le modèle exact, le nombre d'appels et de jetons, et le taux d'accord par source (`data/distill/agreement.json`).
- **Contamination.** Le professeur ne voit aucun record de test. Ses données d'entraînement ne sont pas publiées : on ne peut pas exclure qu'il ait vu les benchmarks publics. Le rapport le dit.
- **Conditions.** Les poids du modèle sont sous MIT. Les appels passent par l'API de z.ai, dont les conditions d'utilisation s'appliquent aussi (`DATA_LICENSES.md`).
- **Ce qui ne change pas.** Les modèles, le pilote, les budgets, les ensembles de test, l'évaluation et les portes.

**Amendement 2 (29/10/2026, avant tout run s6 : aucun entraînement n'a tourné, aucun résultat n'a été lu). Fenêtres de s6-nano : 512 / 160 au lieu de 256 / 128.**

- **Changement.** `configs/s6-nano.json` passe à `max_state_tokens: 512` et `max_head_tokens: 160`. RoPE n'a pas de table de positions : le nombre de paramètres ne change pas (1 569 457 / 778 065), et `s6_parameter_counts` vérifie les nouvelles fenêtres.
- **Pourquoi.** La part de records tronqués a été mesurée avec les vrais tokenizers, écrits le 29/10/2026 par `scripts/prepare-data.sh` (`sky-train s2-truncation`). Le tokenizer de 2 048 jetons de nano découpe plus finement que celui de micro (4 096) : avec 256 / 128, nano coupait l'état de **71,6 %** des questions de `typed_decisions_bench` et la tête de 11,2 %. Or le bench est justement sa porte.

  | | état tronqué (bench) | tête tronquée (bench) | état tronqué (toutes les sources de test) |
  |---|---:|---:|---:|
  | s6-nano, 256 / 128 | 71,6 % | 11,2 % | 15,9 % |
  | s6-nano, 384 / 128 | 9,5 % | 10,0 % | — |
  | s6-nano, 384 / 160 | 9,5 % | 0,0 % | — |
  | **s6-nano, 512 / 160** | **1,8 %** | **0,0 %** | **0,4 %** |
  | s6-micro, 384 / 128 (inchangé) | 4,2 % | 0,0 % | 1,0 % |

  Chaque ligne mesure 2 000 records par source, sauf la ligne 256 / 128 qui en mesure 500. Les sources de test sont `typed_decisions_bench`, le test de When2Call, `zs_massive_en`, `zs_massive_fr` (1 000 records) et `zs_banking77`.
- **Coût.** Des séquences plus longues, donc des pas plus lents. Nano reste environ 4 fois plus petit que micro, dont l'état fait 384 jetons.
- **Ce qui ne change pas.** s6-micro, les tokenizers, les données, les labels du professeur (ils ne dépendent pas des fenêtres), les budgets, l'évaluation et les portes.

**Amendement 3 (03/11/2026, pendant la labellisation, avant tout entraînement s6 : aucun résultat de modèle n'a été lu). Les textes refusés par la modération de z.ai.**

- **Constat.** Du 29/10 au 02/11/2026, la labellisation a écrit 350 949 labels sur 747 198. 128 réponses étaient des refus de la modération de z.ai : HTTP 400, code 1301, « contenu potentiellement sensible ». Des sources comme la modération, les injures ou les refus contiennent forcément de tels textes. Un lot refusé l'était à chaque passe, et ses 19 voisins innocents n'étaient jamais labellisés. Les 89 autres échecs étaient des coupures réseau, reprises ensuite.
- **Changement.** Un lot refusé par la modération est coupé en deux, récursivement, jusqu'à isoler chaque texte refusé seul (`sky-data teacher label-decisions`). Ces textes vont dans `data/teacher/glm_labels.refused.jsonl` et ne sont plus jamais envoyés. Ils gardent leur seule cible or dans `data/decisions` et n'entrent pas dans `data/distill`. Un refus ne compte pas parmi les « 3 lots ratés de suite » qui arrêtent un worker.
- **Règles inchangées.** La couverture exigée reste de 95 % de la sélection en labels. Les textes refusés comptent comme non labellisés. Le rapport donne leur nombre par source.
- **Ce qui ne change pas.** La sélection, la question posée, le filtre d'accord, les poids, les modèles, l'évaluation et les portes.

**Amendement 4 (09/11/2026, avant tout entraînement de s6-nano et s6-micro : seule la recette pico tourne, aucun de ses résultats n'a été lu). La recette passe par le fichier, en mode bridé.**

- **La recette.** Le commit manuel du §2 est remplacé par `runs/recipe/recipe.json` et les configs dérivées de `runs/recipe/configs/` (`scripts/s6-adopt.py env s6-nano`, puis `s6-micro`), sans choix à la main (`docs/s6-prereg.md`, amendement 4). `configs/s6-nano.json`, `configs/s6-micro.json` et le test `s6_parameter_counts` ne changent pas.
- **Le mode bridé.** Le run se fait en mode bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), comme l'amendement 5 de `docs/s6-prereg.md`. Les budgets en minutes ne comptent que le temps de travail, et le rapport donne le nombre de pas réellement faits.
- **Ce qui ne change pas.** Données, tailles, budgets, évaluation et portes.

**Amendment 5 (12/11/2026, before any training of s6-nano and s6-micro: the recipe is written (`runs/recipe/recipe.json`), no score of s6-nano and s6-micro exists). Two modes.** At the owner's request (11/11/2026), the run uses `SKY_DUTY=0.85`, `SKY_COOL=50:10` while the Mac is not in use, and the throttled mode of the previous amendment (`0.5`, `45:15`) the rest of the time, as in amendment 6 of `docs/s6-prereg.md`. Budgets in minutes still count work time only. A change of mode stops the training, which resumes from its last checkpoint (`RESUME=1`). The report gives the steps actually done and the modes used. Nothing else changes.

**Amendment 6 (12/11/2026, before any training of s6-nano and s6-micro: no score of s6-nano and s6-micro exists). The brake without a restart, and a memory cap.** The mode now changes while a run goes on: `sky-train` reads `SKY_BRAKE_FILE` again every few seconds, and `scripts/auto-brake.sh` writes it (the Mac counts as not in use after 10 minutes with no key or pointer input, unless a video or a call keeps the display awake, it runs on battery, or macOS reports thermal pressure). `scripts/train-all.sh` caps MLX's buffer cache at 2 GB, and the script stops a training that holds more than 13 GB of memory (see amendment 4 of `docs/s6-ear-prereg.md`). None of this changes what is computed.

**Amendment 7 (13/11/2026, after the first 8 minutes of the first decision training of s6-nano, before any evaluation of s6-nano or s6-micro: no gate score, no pilot result and no chosen rate exists). A memory defect of the decision training, fixed without changing the update; a night mode.**

- **What happened.** On 13/11 at 00:16 the memory cap of amendment 6 stopped the decision training of the first rate of the s6-nano pilot (x0.5) after 8.2 of its 20 minutes (860 steps): `sky-train` held 15 GB on a 16 GB Mac. Measured afterwards on throwaway folders: the peak of MLX's memory over one step grows with the shape of the batch, and 12% of the batches (128 rows whose states reach 512 tokens) need 10 to 17 GB for a single step; batches with long questions need 5 to 8 GB. The run was working in swap: about 105 steps per working minute.
- **The fix.** `sky-train s2-decide --step-mem-gb 3` (`STEP_MEM_GB` in `scripts/train-s6.sh`): a batch whose step would need more than 3 GB is cut into 2, 4 or 8 equal parts of rows; the gradients of the parts are averaged and one update is applied. The loss is a mean over the rows of the batch, so this is the update of the whole batch: the same items are drawn in the same order, each batch gives one step, and only the order of the floating-point sums changes. A unit test checks the loss and every gradient of the cut batch against the whole one, compiled or not, with QAT or not (`a_step_in_parts_is_the_step_of_the_whole_batch`). The vectors of the 1.7 million loaded questions also drop their spare capacity. On the same throwaway run: peaks of 2.3 GB (median) and 4.5 GB (largest, QAT start aside), and about 200 steps per working minute.
- **What it applies to.** Every decision training of s6-nano and s6-micro: the rate pilots, the first run and the DAgger rounds. The limit changes how a step is computed, not what it computes, so it may be changed between two launches (the estimate was measured on s6-nano; for s6-micro it is checked against `mlx_peak_mb` in the logs of its pilot). The cap of amendment 6 now needs four readings in a row above 13 GB (two minutes): the loading of the decision sets peaks for about a minute.
- **The stopped run.** The 860 steps done in swap are not kept: with them the x0.5 arm would have had fewer steps in its 20 minutes than x1 and x2. Its files are moved to `runs/s6-nano-lr-swap/` and its decision training starts over from its kept pretraining (10,082 steps, finished before the defect, which the pretraining does not have: fixed shapes of 64 x 256). Its log had printed the validation accuracy at the start of QAT (0.409 in float); nothing was decided from it.
- **Earlier runs.** The pico pilot and ablation of `docs/s6-prereg.md` and s6-secrets-pico used the same kind of batches with a smaller model, before the cap existed or under it; their memory was not measured. Their results stand as they are and the report says so.
- **A third brake mode.** At the owner's request (13/11/2026, full speed for the night), `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and the other programs use less than 60% of one core; thermal pressure, battery, a held display or any input bring back the modes of amendment 5. Budgets in minutes still count working time only.
- **What does not change.** Data, models, budgets, the pilot rule, the gates.

**Amendment 8 (13/11/2026, during the first decision training of s6-nano, after 72 of its 90 minutes; the rate pilot of s6-nano is finished and its rate chosen (x1), no gate score of s6-nano or s6-micro exists). A second memory defect of the decision training, fixed without changing a value.**

- **What happened.** On 13/11 at 06:05 the memory cap stopped the first decision training of s6-nano at step 16,020 (72.2 of its 90 minutes): `sky-train` had held 14 GB for two minutes. This is not the defect of amendment 7: MLX's own memory stayed flat for the whole run (peak 6.65 GB) and the run was not slowed (420 to 640 questions per second in its last minutes). Measured on the resumed run, with two memory maps nine minutes apart: the heap grew by 0.5 GB and 3.15 million small blocks while the GPU memory stayed at 4.5 GB. MLX keeps one compiled graph per input shape for the life of the process, and the decision batches keep bringing new shapes (rows, state tokens, static streams and static tokens all vary): about 3 MB per shape, with no bound. A throwaway test on the CPU device shows the same growth with a small graph, and that emptying the cache gives the memory back.
- **The fix.** `sky-train` empties MLX's cache of compiled graphs every 256 new shapes (`SKY_COMPILE_SHAPES`; 0: never). A shape met again is compiled again, about 50 ms, into the same graph, so no value changes: a unit test compares the losses and the gradients with and without emptying (`emptying_the_cache_of_compiled_graphs_changes_no_value`). The cost is working time, an estimated 2% of the steps of a decision run. The decision training also frees the texts of the questions once the sets are loaded (only `s2-eval` reads them).
- **What it applies to.** The binary is built at the start of each launch of `scripts/train-s6.sh`: the pilot, the first run and the rounds of s6-micro, and any relaunch of s6-nano. The first run of s6-nano and its DAgger rounds finish with the binary they started with unless they are stopped again (a round lasts 15 minutes: under 1 GB of growth). The limit changes how long a step takes, not what it computes, so it may be changed between two launches.
- **The stopped run.** It resumes from its last checkpoint (step 15,573, 70.1 working minutes: parameters, optimizer and random generator), as `RESUME=1` provides. The 447 steps done after that checkpoint are done again and the budget counts them once. The run prints its validation accuracy every 1,000 steps, as every run does; nothing was decided from it.
- **What does not change.** Data, models, budgets, the chosen rate, the gates, the 13 GB cap.

**Amendment 9 (13/11/2026, five minutes into the decision training of the first rate of the s6-micro pilot (x0.5); no result of that pilot exists, no gate score of s6-micro). The check of the step memory on s6-micro that amendment 7 announced.**

- **What was read.** `mlx_peak_mb` in the log of that run: 6.3 GB within its first 320 steps, under a limit of 3 GB. The later rise to 8.4 GB comes with the validation at the start of QAT, which is not cut into parts and lasts a few seconds (s6-nano shows the same: 4.7 GB before its first validation, 6.6 GB after). The process held 7.5 to 12 GB, under the cap.
- **Why.** The estimate counted the exits on the state side. s6-nano has 4 exits and 4 layers for a state token (2 layers, 2 loops); s6-micro has 3 exits for 6 layers (2 layers, 3 loops), so its estimate was half of what a step takes.
- **The fix.** The estimate counts the depth (state layers times loops). Nothing changes for s6-nano; for s6-micro more batches are cut into parts, which is the same update (amendment 7). The run also printed its validation accuracy at the start of QAT, as every run does; nothing was decided from it.
- **What it applies to.** The binary built at the launch of the first run of s6-micro, and what follows. The three rates of the s6-micro pilot run in one launch, with the binary built before this fix: they are compared with the same estimate.
- **What does not change.** Data, models, budgets, the pilot rule, the gates, the limit of 3 GB, the 13 GB cap.

**Amendment 10 (13/11/2026, during the rate pilot of s6-micro). The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.

**Amendment 11 (14/11/2026, during the first decision training of s6-micro, after 126 of its 180 minutes; the rate of s6-micro is chosen (x0.5), no gate score of s6-micro exists). The memory of the compiled decision step, measured: the decision training runs uncompiled.**

- **What happened.** On 14/11 at 00:21 the first decision training of s6-micro was stopped by hand at step 27,280 (126.2 of 180 minutes): it had been working in swap since about minute 92, at 110 questions per second instead of 205. Its heap held 9.3 GB in 43 million blocks, 8.3 GB of them in swap, against 4.4 GB after loading; MLX's own memory was flat. The memory cap had not stopped it (the process stayed between 11 and 14 GB, never four readings in a row at 13).
- **What amendment 8 got wrong.** Emptying MLX's cache of compiled graphs does not bound the memory of this run. Measured on 14/11 with no training running, on throwaway folders with the s6-micro command: (1) uncompiled (`--no-compile`), the heap does not grow at all (12.18 million blocks over 240 steps); (2) compiled, it grows by about 3,300 blocks and 0.65 MB per step in the first minutes, the same with the cache emptied every 8 shapes or every 256; (3) a decision batch has far more shapes than its rows and lengths (the number of options and of static streams vary too): hundreds in the first 800 steps; (4) each compilation takes about 8,400 blocks and emptying the cache gives back two thirds of them, so every compilation leaves about 0.4 MB behind, and emptying the cache causes more compilations. On a set with few shapes (`data/numeric` alone) the compiled heap is flat. Where the 0.4 MB stays inside MLX is not known.
- **The fix.** `scripts/train-s6.sh` passes `--no-compile` to the decision training (`DECIDE_COMPILE=1` brings the compiled step back). It is the same loss and the same gradients computed without MLX's graph compilation: the unit tests `compiled_and_eager_decision_steps_agree` and `a_step_in_parts_is_the_step_of_the_whole_batch` compare the two; only floating-point rounding differs. Measured speed on s6-micro: 195 to 235 questions per second uncompiled, against 205 on average for the compiled run before it swapped. `sky-train` now writes `heap_mb` in its log, so the memory of the process outside MLX is recorded.
- **What it applies to.** The rest of the first decision training of s6-micro and its DAgger rounds. s6-nano, the rate pilots and the first 126 minutes of this run were trained compiled; their results stand as they are.
- **The stopped run.** It resumes from its last checkpoint (step 27,161, 125.8 working minutes), as `RESUME=1` provides. The steps done in swap between minutes 92 and 126 are kept: the run has fewer steps in its 180 minutes than it would have had, and the report says so. The run printed its validation accuracy every 1,000 steps, as every run does; nothing was decided from it.
- **What does not change.** Data, models, budgets, the chosen rate, the gates, the limit of 3 GB per step, the 13 GB cap.
