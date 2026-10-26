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

## 3. Le run (`scripts/train-cast.sh train`)

- `configs/s6-cast-pico.json` : 388 320 paramètres.
- 480 minutes (budget en temps), batch 256, WSD.
- QAT ternaire sur les 20 % finaux, EMA 0,999.
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
  - Si l'une de ces vérifications échoue, s6-cast n'est pas lu : on corrige d'abord le banc.
- **Modèle rapporté.** s6-cast avec et sans retournement. Le modèle rapporté est **sans** retournement : c'est le mode de la page web, et le retournement double le coût.

## 6. Porte de publication

s6-cast est publié (site, manifeste, Hugging Face) si, sans retournement, il bat la référence sans entraînement FLAIR sur les deux métriques :

- **nGMASE ≤ 0,838**
- **et nWQL ≤ 0,587**

Quel que soit le résultat, le rapport donne l'écart à TinyCast (0,774 / 0,545) et à Timestral-5M (chiffres du leaderboard). Il donne aussi la taille, la latence et les sorties moyennes utilisées.

Si la porte n'est pas franchie, le modèle n'est pas publié, et le rapport le dit.

## 7. Amendements

Aucun.
