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

**Pas de professeur.** Aucun fichier professeur n'est sur disque (ni `data/s4`, ni `data/s5`, ni les professeurs laya ou oscar-150m). Cette v1 est donc entraînée sans distillation, et ce choix est fait avant le run. Les cibles douces déjà présentes dans les jeux de données (tasksource, procedural, calibrated…) suffisent.

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
| fenêtres état / tête | 256 / 128 | 384 / 128 |

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

Aucun pour l'instant.
