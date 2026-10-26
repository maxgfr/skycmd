# s6-cast : pré-enregistrement (premier entraînement et porte de publication)

**Statut : figé le 26/10/2026, avant tout entraînement de s6-cast.** Ce fichier fixe les données, le run, le choix du checkpoint, l'évaluation et la porte de publication. Les résultats seront lus à travers ce texte.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 7 ;
- il précède le run qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un score GIFT-Eval.

## 1. Données d'entraînement

- **Pool synthétique** (`sky_infer::cast::synth`) : GP façon KernelSynth, impulsions, tendance-saisonnalité-irrégularités et SARIMA, avec les augmentations de Reverso (`docs/SPEC-S6-CAST.md` §3).
- **Séries réelles** (`data/series/`, 30 % des fenêtres), avec une licence d'origine vérifiée et absentes des deux bancs de test (`DATA_LICENSES.md`) :
  - `sky-data fetch-series` : London smart meters, wind farms (horaire et journalier), pedestrian counts et rideshare (Monash, CC-BY-4.0) ; Wikipedia daily (CC0) ; USHCN daily (domaine public) ;
  - les trajectoires de nos quatre jeux (`cast-synth --games`).
- Une série réelle sur dix (par id) est tenue à l'écart pour la validation.
- **Aucune série de GIFT-Eval ni de Chronos-ZS n'entre dans l'entraînement.** GIFT-Eval-Pretrain n'est pas utilisé, faute de licence déclarée.

## 2. Pilote de learning rate (`scripts/train-cast.sh pilot`)

- 3 000 pas à 1e-3, 3e-3 et 1e-2 (AdamW ; Muon v2 au même taux), batch 256, sans QAT.
- **Choix.** La pinball de validation finale en f32 (dernière sortie). On garde 3e-3 sauf si un autre taux la baisse d'au moins 2 %.
- **Distillation.** Même pilote au taux retenu, avec TinyCast comme professeur (poids 0,5, 50 % des fenêtres). On distille, sauf si la pinball est pire de plus de 1 % : la pinball de validation se mesure contre la vérité, et un professeur peut la dégrader sans dégrader GIFT-Eval. Le seuil est donc volontairement lâche.

## 3. Le run (`scripts/train-cast.sh train`)

- `configs/s6-cast-pico.json` : 388 320 paramètres.
- 480 minutes (budget en temps), batch 256, WSD.
- QAT ternaire sur les 20 % finaux, EMA 0,999.
- **Professeur** (sauf si le pilote l'écarte) : TinyCast, dont les déciles servent de seconde cible sur 50 % des fenêtres. La perte est `pinball(vérité) + w·L1(déciles du professeur)`, avec w qui passe linéairement de 0,5 à 0 sur le budget. C'est le teacher annealing de BAM : la fin de l'entraînement suit la vérité seule. Un élève qui apprend à la fois la vérité et le professeur peut dépasser ce dernier (Born-Again Networks, arXiv 1805.04770 ; Revisiting Self-Distillation, arXiv 2206.08491). TinyCast exclut de son entraînement les jeux qui recoupent les tests GIFT-Eval et Chronos-ZS : il ne fait pas fuir le test.
- Graine 1, une seule fois. Seul un échec technique justifie de relancer, et il est noté dans le rapport.

## 4. Checkpoint exporté (`export`)

- **Choix des poids.** L'EMA (`last-ema`) si sa pinball de validation ternaire finale est plus basse que celle de `last`, sinon `last`. Aucun checkpoint intermédiaire n'est considéré.
- **Calibration de la sortie anticipée.** `sky-cast-eval calibrate` : α = 0,05, δ = 0,02 (unités normalisées), sur 2 000 fenêtres synthétiques jamais vues.

## 5. Évaluation (`eval`)

- **Le banc.** GIFT-Eval, 97 configurations, avec le protocole officiel : `scripts/gift-eval.py`, gluonts `evaluate_model`, les 9 déciles.
  - Chaque métrique est divisée par celle du seasonal naive de référence (`eval/gift-eval/seasonal_naive.csv`), puis on prend la moyenne géométrique.
  - On obtient **nGMASE** (point), **nWQL** (probabiliste) et nMSIS.
- **Validation du banc, avant toute lecture de s6-cast.**
  - Notre seasonal naive (prévision ponctuelle recopiée sur tous les déciles) doit donner nGMASE à 1,00 ± 0,02. Son nWQL n'est pas 1 : la référence du leaderboard est probabiliste. Le WQL est donc validé par TinyCast, ci-dessous.
  - Notre TinyCast porté (avec retournement, comme leur soumission) doit retrouver 0,774 et 0,545 à ± 0,02 près.
  - **Résultat (26/10/2026) : banc validé.** Seasonal naive : nGMASE 0,99986. TinyCast porté : nGMASE 0,786, nWQL 0,557 (nMSIS 0,578), sur les 97 configurations.
    - 95 configurations retrouvent le MASE publié à 1 % près (rapport géométrique 1,015 sur les 97).
    - L'écart vient de `bizitobs_l2c/5T` medium et long (MASE 0,85 et 1,25 contre 0,49 et 0,48). Pour ce seul jeu, le pilote de TinyCast (`tinycast/eval.py`) active deux réglages qui ne sont pas dans son prédicteur générique :
      - `no_daily`, choisi d'après le nom du jeu ;
      - un sous-échantillonnage d'alignement de période (k = 7), qui ne se déclenche que là.
    - Notre portage est le prédicteur générique. Le banc applique le même protocole à tous les modèles, sans réglage par jeu.
  - Si l'une de ces vérifications échoue, s6-cast n'est pas lu : on corrige d'abord le banc.
- **Modèle rapporté.** s6-cast avec et sans retournement. Le modèle rapporté est **sans** retournement : c'est le mode de la page web, et le retournement double le coût.

## 6. Objectif et porte de publication

**Objectif : battre TinyCast sur les deux métriques**, sans retournement : **nGMASE < 0,774 et nWQL < 0,545**. Le rapport dit, sans ambiguïté, si l'objectif est atteint.

La barre reste celle publiée, réglages par jeu compris : c'est la plus exigeante. Le rapport donne aussi l'écart au TinyCast générique de notre banc (0,786 / 0,557, avec retournement).

s6-cast est publié (site, manifeste, Hugging Face) si, sans retournement, il bat la référence sans entraînement FLAIR sur les deux métriques :

- **nGMASE ≤ 0,838**
- **et nWQL ≤ 0,587**

Quel que soit le résultat, le rapport donne l'écart à TinyCast (0,774 / 0,545) et à Timestral-5M (chiffres du leaderboard). Il donne aussi la taille, la latence et les sorties moyennes utilisées.

Si la porte n'est pas franchie, le modèle n'est pas publié, et le rapport le dit.

## 7. Amendements

1. **26/10/2026**, avant tout entraînement :
   - ajout de l'objectif « battre TinyCast » et de la distillation par TinyCast (§2, §3, §6), à la demande du propriétaire du projet ;
   - le contrôle du seasonal naive porte sur nGMASE seul (§5).

   Aucun score de s6-cast n'existait.

2. **26/10/2026**, avant tout entraînement :
   - résultat de la validation du banc et origine de l'écart sur `bizitobs_l2c` (§5) ;
   - la barre de l'objectif reste 0,774 / 0,545 (§6) ;
   - les cibles du professeur sont calculées sur des points de coupe fixes de chaque série (deux par série), et non plus à chaque lot. C'est un changement de débit : 730 fenêtres/s sinon, contre environ 2 000. La part distillée (50 %) et le poids (0,5 → 0) ne changent pas.

   Aucun score de s6-cast n'existait.
