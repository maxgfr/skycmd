# TODO : entraîner les modèles s6

Trois modèles ont leur code prêt et ne sont pas encore entraînés : **s6-pico**, **s6-secrets-pico** et **s6-cast-pico**. Chacun a un pré-enregistrement figé. On le suit à la lettre :

- un amendement daté (§7) précède le run qu'il concerne ;
- on ne relance rien pour « voir » : seul un échec technique justifie un nouveau run, et il est noté dans le rapport ;
- on ne choisit jamais sur un checkpoint intermédiaire ;
- un modèle qui échoue à sa porte n'est pas publié, et le rapport le dit.

Les autres configs `s6-*` de `configs/` ne sont pas des modèles à entraîner :

- `s6-femto`, `s6-femto-xsa` et `s6-cast-femto` sont de minuscules configs aux poids aléatoires, pour les tests golden de `sky-infer` (`tests/golden_s6.rs`, `golden_s6x.rs`, `golden_cast.rs`) et les tests unitaires ;
- `s6-pico-xsa` est la config que prend s6-pico si l'ablation adopte Parallax et XSA (§2.3).

Ordre conseillé (une seule machine) : données → s6-pico (pilote, ablation, modèle final) → s6-secrets (il reprend les options adoptées par s6-pico) → s6-cast (indépendant, à caser quand la machine est libre). Durées mesurées ou estimées sur un M5.

## 0. Prérequis

- [ ] Toolchain Metal : `xcodebuild -downloadComponent MetalToolchain`, et `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- [ ] `export SKYCMD_STORE=$PWD` (le checkout principal) pour `cargo xtask`.
- [ ] `uv` installé (s6-cast : `scripts/gift-eval.py` tourne sous Python 3.12).
- [ ] Place disque : environ 9,7 Go libres le 28/10/2026, c'est juste. Les runs écrivent dans `runs/` et `models/`, le corpus ouvert ajoute 150 Mo.

## 1. Données (une seule fois, avant le premier pilote)

`docs/s6-prereg.md` §1 : toutes les données sortent de `scripts/prepare-data.sh`, lancé une fois avant le pilote, sans régénération ensuite.

- [ ] `scripts/prepare-data.sh`. Les étapes déjà faites sont sautées. La seule qui manque est `data/text-open` : environ 450 Mo lus pour 150 Mo gardés, Europarl FR étant le plus long.
- [ ] Noter dans le rapport final si les données GLM (`glm_orders`, `paraphrased`) sont présentes, c'est-à-dire si `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` étaient définis.
- [ ] Recopier les nombres de docs du tableau de `fetch-text --set open` dans `DATA_LICENSES.md`, section « MLM text, open set », à la place de « not fetched yet ».
- [ ] Vérifier que `data/text-open/` contient les 6 fichiers : `dolma3_*`, `txt360_*`.

## 2. s6-pico (`docs/s6-prereg.md`, `docs/SPEC-S6.md`)

Pré-entraînement sur `data/text-open,data/text` (amendement 1), décisions sur `data/decisions,data/games,data/numeric`.

### 2.1 Pilote de learning rate (environ 2 h 30)

- [ ] `cargo xtask s6lr`, soit 6 runs : `softmax-adamw` et `parallax-muon` à ×0,5, ×1 et ×2. La sortie est `runs/s6lr/summary.json`.
- [ ] Appliquer la règle §2 : garder ×1 sauf si une autre échelle bat la précision ternaire finale (`val`) d'au moins **0,01**, pour chaque optimiseur.
- [ ] Si le choix tombe au bord (×0,5 ou ×2) : une seule extension (×0,25 ou ×4), **par amendement daté avant de la lancer**.

### 2.2 Ablation (environ 15 h)

- [ ] `cargo xtask s6ab --adamw-lr-scale <a> --muon-lr-scale <m>`, avec les échelles du pilote : 11 bras × 3 graines (7, 11, 13). La sortie est `runs/s6ab/summary.json`.
- [ ] Ne rien changer aux données ni aux budgets entre le pilote et l'ablation.

### 2.3 Décider ce qui est adopté (`docs/s6-prereg.md` §4)

- [ ] Muon v2 contre AdamW : gain significatif (≥ 0,01 et plus grand que l'écart-type des deux bras), sans perte arena (> 5 % sur un jeu). Le gain exigé passe à 0,02 si le surcoût en `train_minutes` dépasse 15 %.
- [ ] Parallax : `parallax-muon` contre `softmax-muon`, même règle.
- [ ] XSA : `xsa-muon` contre `softmax-muon`. Si Parallax et XSA sont adoptés, `CONFIG=configs/s6-pico-xsa.json`.
- [ ] SODA, KD de sortie, EMA, StableQAT et curriculum, lus contre `parallax-muon` : gain ≥ 0,005, sans perte arena, ECE pas plus de 0,01 pire.
- [ ] Sans polish : non inférieur (≥ −0,005) **et** au moins 2 % plus rapide.
- [ ] Écrire les choix dans les variables de `scripts/train-s6.sh` : `CONFIG`, `OPT`, `PRETRAIN` et `DECIDE`. Les noter, car s6-secrets reprend le même `OPT`.

### 2.4 Modèle final (environ 1 h 30)

- [ ] `CONFIG=… OPT="…" PRETRAIN="…" DECIDE="…" scripts/train-s6.sh`. Les étapes :
  - pré-entraînement 8 min ;
  - décision 15 min ;
  - 3 rounds DAgger de 8 min (oracle 0,2 → 0,1 → 0,05) ;
  - seuils de sortie par contrôle conformal (α = 0,02) ;
  - arena.
- [ ] Porte des seuils : à moins de 2 % de la pleine profondeur, sans crash ni veto de plus. Sinon, garder la pleine profondeur ou passer par `scripts/calibrate-exit.sh`.
- [ ] Porte de publication : `s6-pico-g3` contre le `s3-pico-g3` publié (4 jeux, 30 épisodes, τ = 1). Il faut aucun jeu à −5 %, **et** un jeu à +5 % ou la même qualité avec moins de couches.

## 3. s6-secrets-pico (`docs/s6-secrets-prereg.md`)

Après 2.3, parce qu'il reprend l'`OPT` adopté. Pré-entraînement sur `data/text-open,data/text` (amendement 1).

- [ ] `scripts/train-secrets.sh data` : jev `data/secrets` (50 000 textes, plus `secret_bench` de 2 000) et score du scanner à règles. Celui-ci doit redonner **0,977** (secret) et **0,787** (service), sinon on s'arrête et on comprend pourquoi.
- [ ] Décider **avant le run** si `DECIDE` contient `--ema`, ce qui fixe le checkpoint exporté.
- [ ] `OPT="…" DECIDE="…" scripts/train-secrets.sh train eval` : pré-entraînement 8 min en séquences de 256 jetons, décision 15 min sur `data/secrets` seul, calibration, export ternaire, `s2-eval` sur `secret_bench`. Graine 0, une seule fois.
- [ ] Porte : exactitude secret **> 0,977** et service **> 0,787**, en ternaire, à pleine profondeur.
- [ ] Le rapport donne aussi kind, line, la validation, la sortie anticipée, la taille et la latence, quel que soit le résultat.

## 4. s6-cast-pico (`docs/s6-cast-prereg.md`, `docs/SPEC-S6-CAST.md`)

Indépendant de s6-pico, il ne lit pas de texte.

- [ ] `scripts/train-cast.sh data`. Sur la machine de build, le 28/10/2026, `data/series/`, les fenêtres GIFT-Eval (`data/gift-eval/`) et les références seasonal naive et TinyCast (`data/gift-eval-out/`) sont déjà là : l'étape devrait surtout sauter.
- [ ] Banc déjà validé le 26/10/2026 : seasonal naive nGMASE 0,99986, TinyCast porté 0,786 / 0,557. Si le banc est régénéré, revalider avant toute lecture de s6-cast.
- [ ] `scripts/train-cast.sh pilot` : 4 runs de 3 000 pas, soit lr 1e-3, 3e-3 et 1e-2, puis la distillation TinyCast au taux retenu.
  - Learning rate : garder 3e-3 sauf si un autre taux baisse la pinball de validation d'au moins 2 %.
  - Distillation : on distille, sauf si la pinball est pire de plus de 1 % (sinon `TEACHER=none`).
- [ ] `LR=… [TEACHER=none] scripts/train-cast.sh train export eval` : **480 min** d'entraînement (8 h), QAT sur les 20 % finaux, EMA 0,999, graine 1. Ensuite viennent l'export (`last-ema` si sa pinball ternaire est meilleure, sinon `last`), la calibration de sortie (α = 0,05, δ = 0,02) et GIFT-Eval avec et sans retournement.
- [ ] Porte, sans retournement : **nGMASE ≤ 0,838 et nWQL ≤ 0,587**. L'objectif annoncé est de battre TinyCast (< 0,774 / < 0,545). Le rapport dit sans ambiguïté s'il est atteint.

## 5. Publier un modèle qui passe sa porte

- [ ] GGUF dans `site/public/models/` et une entrée dans `site/public/models/manifest.json`. Pour s6-cast, c'est `site/public/models/forecast.json` et la page [Forecast](site/forecast/index.md).
- [ ] Carte `hf/cards/<modèle>.md` et entrée dans `hf/spaces.json`, puis `scripts/build-wasm.sh && (cd site && npm run space:build)` et `python3 scripts/hf-publish.py --only <modèle>`.
- [ ] Tableau des modèles du README, section « Coming next » de `site/bench/index.md` (le modèle passe dans les résultats), section du modèle dans `BENCH.md`.
- [ ] Rapport : tous les chiffres pré-enregistrés, y compris les échecs, sur la page Results et dans le chapitre du tutoriel.

Un modèle qui échoue n'est pas publié, mais son résultat va quand même dans le rapport et sur le site.
