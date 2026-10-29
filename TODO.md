# TODO : entraîner les modèles s6

> **La labellisation GLM tourne en arrière-plan depuis le 29/10/2026 (§1.1, environ 80 h).** Si la machine s'éteint ou redémarre, la relancer depuis le checkout principal :
>
> ```
> caffeinate -i scripts/distill-glm.sh loop mix
> ```
>
> Elle reprend là où elle s'était arrêtée, sans rien perdre. `pgrep -fl distill-glm` dit si elle tourne encore. `scripts/distill-glm.sh count` donne ce qui reste, sans faire d'appel. Le journal est dans `runs/distill-glm.log`.

Quatre modèles ont leur code prêt et ne sont pas encore entraînés : **s6-nano** et **s6-micro** (le s6 généraliste : les jeux et des questions en texte libre), **s6-secrets-pico** et **s6-cast-pico**. Chacun a un pré-enregistrement figé. **s6-pico n'est plus un modèle final** (`docs/s6-prereg.md`, amendement 2) : son pilote et son ablation restent, parce qu'ils choisissent la recette des deux tailles. On suit chaque pré-enregistrement à la lettre :

- un amendement daté (§7, §8 pour `docs/s6-gen-prereg.md`) précède le run qu'il concerne ;
- on ne relance rien pour « voir » : seul un échec technique justifie un nouveau run, et il est noté dans le rapport ;
- on ne choisit jamais sur un checkpoint intermédiaire ;
- un modèle qui échoue à sa porte n'est pas publié, et le rapport le dit.

Les autres configs `s6-*` de `configs/` ne sont pas des modèles à entraîner :

- `s6-femto`, `s6-femto-xsa` et `s6-cast-femto` sont de minuscules configs aux poids aléatoires, pour les tests golden de `sky-infer` (`tests/golden_s6.rs`, `golden_s6x.rs`, `golden_cast.rs`) et les tests unitaires ;
- `s6-pico` et `s6-pico-xsa` sont les configs du pilote et de l'ablation (§2.1, §2.2), pas des modèles à publier.

Ordre conseillé (une seule machine) : données → labels GLM (§1.1 : à distance, environ 80 h sur plusieurs fenêtres z.ai, la machine reste libre, donc à lancer tout de suite) → pilote et ablation pico → s6-nano → s6-micro → s6-secrets (il reprend les options adoptées par l'ablation) → s6-cast (indépendant, à caser quand la machine est libre). Durées mesurées ou estimées sur un M5.

## 0. Prérequis

- [x] Toolchain Metal : `xcodebuild -downloadComponent MetalToolchain`, et `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Présente le 29/10/2026 (`xcrun metal --version` : 32023.921).
- [ ] `export SKYCMD_STORE=$PWD` (le checkout principal) pour `cargo xtask`, dans le terminal qui lance le pilote et l'ablation.
- [x] `uv` installé (s6-cast : `scripts/gift-eval.py` tourne sous Python 3.12).
- [x] `node` et `wasm-pack` (présents le 29/10/2026) : l'étape `eval` de `scripts/train-s6.sh` construit `site/public/pkg` s'il manque (`scripts/build-wasm.sh`) pour mesurer la latence WASM.
- [ ] Place disque : **15 Go libres** le 29/10/2026, après suppression du cache de build debug (`target/s3-train/debug`, 9,2 Go, produit par `cargo test` et clippy ; il se reconstruit tout seul). Les scripts d'entraînement n'utilisent que `target/s3-train/release`. Les runs écrivent dans `runs/` et `models/`, et les checkpoints de s6-micro (5,85M paramètres, état d'optimiseur compris) pèsent bien plus que ceux de pico. Revérifier avant 2.5, et supprimer à nouveau `target/s3-train/debug` après des `cargo test`.

## 1. Données (une seule fois, avant le premier pilote)

`docs/s6-prereg.md` §1 : toutes les données sortent de `scripts/prepare-data.sh`, lancé une fois avant le pilote, sans régénération ensuite.

- [x] `scripts/prepare-data.sh`, fait le 29/10/2026 (journal : `runs/prepare-data-s6b.log`) :
  - `data/text-open` : 15 011 documents et 99,5 Mo de texte (97 Mo sur disque), et non 150 : le shard Wikipedia ne contient que 4,3 Mo (`docs/s6-prereg.md`, amendement 3) ;
  - les tokenizers `data/tok/tok-s6-2048.json` et `tok-s6-4096.json` (s6-nano, s6-micro). Le premier lancement plantait après avoir sauvegardé le tokenizer 2048 : le tableau de statistiques lisait `data/text-open` comme des décisions. C'est corrigé dans `sky-data train-tokenizer` ;
  - `data/commands/massive.jsonl` (32 341 énoncés), puis `data/s4/heldout` (`s4-corpus --only-heldout`) : les tests zéro-shot MASSIVE en/fr de s6-nano et s6-micro.
- [ ] Noter dans le rapport final si les données GLM (`glm_orders`, `paraphrased`) sont présentes, c'est-à-dire si `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` étaient définis lors de `prepare-data.sh`. Le 29/10/2026, elles sont **absentes** de `data/decisions`.
- [x] Recopier les nombres de docs du tableau de `fetch-text --set open` dans `DATA_LICENSES.md`, section « MLM text, open set », à la place de « not fetched yet ».
- [x] Vérifier que `data/text-open/` contient les 6 fichiers : `dolma3_*`, `txt360_*`.
- [x] Vérifier que `data/s4/heldout/` contient `zs_massive_en.jsonl` (2 000 questions), `zs_massive_fr.jsonl` (1 000) et `zs_banking77.jsonl` (2 000). Ce dernier est rapporté comme « dans la distribution », parce que Banking77 est dans `data/decisions`.

### 1.1 Professeur GLM-5.3-flash pour s6-nano et s6-micro (`docs/s6-gen-prereg.md`, amendement 1)

GLM-5.3-flash est open source : poids sous licence MIT, [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). On l'appelle par l'API de z.ai pour ne pas faire chauffer la machine.

À faire une fois, avant le pilote de s6-nano. Il ne faut que `data/decisions`, déjà présent : on peut lancer dès maintenant, en même temps que le reste du §1 et que le pilote et l'ablation pico. Tout tourne chez z.ai : la machine ne chauffe pas.

- [x] `ccs list` doit montrer le fournisseur `zai` configuré (`ccs use zai` sinon). Le script lit l'endpoint et la clé dans la config de `ccs`, sans jamais afficher la clé ; `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` priment s'ils sont définis.
- [x] `scripts/distill-glm.sh count` (aucun appel) : 747 198 records (tous les records `train` des jeux Hugging Face de `data/decisions`) et 37 360 appels de 20 attendus. Un autre compte veut dire que `data/decisions` a changé : comprendre pourquoi avant d'appeler.
- [ ] **Lancer le tout, sans surveillance :** `caffeinate -i scripts/distill-glm.sh loop mix`, dans un terminal qui reste ouvert (ou `mkdir -p runs && nohup caffeinate -i scripts/distill-glm.sh loop mix > runs/distill-glm.log 2>&1 &`).
  - **En cours depuis le 29/10/2026 à 16:28**, avec `nohup` (journal : `runs/distill-glm.log`). Pour savoir s'il tourne encore : `pgrep -fl distill-glm`. Plus de 3 200 labels écrits dans les 20 premières minutes, soit environ 2,7 labels/s. Cocher quand `loop` affiche « coverage reached » et que `mix` a écrit `data/distill/glm.jsonl`.
  - **4 workers** (`JOBS=4`), c'est le maximum utile. Mesuré le 29/10/2026 : z.ai plafonne vers 2,5 labels/s quel que soit le nombre de workers, et dès 8 workers une partie des appels part en relances (8 workers : 10 relances sur 24 appels ; 16 workers : 22 sur 32).
  - **Environ 80 h de labellisation**, réparties sur les fenêtres de 5 h de l'abonnement z.ai, avec de l'ordre de 100 à 200 M jetons en entrée et 55 M en sortie.
  - Quand une fenêtre est épuisée, chaque worker s'arrête après 3 lots ratés de suite. `loop` fait une pause de 30 min (`PAUSE=1800`) et reprend là où il s'était arrêté, jusqu'à ce qu'il reste au plus 1 % de la sélection (`COVER_MISS=1`). Il abandonne après 24 pauses sans progrès (`TRIES=24`), soit 12 h.
  - Avancement à tout moment : `scripts/distill-glm.sh count` (sans appel).
  - Si `loop` s'arrête avant la fin (machine éteinte, `TRIES` atteint), relancer la même commande : rien n'est perdu.
- [ ] Les labels doivent couvrir au moins **95 %** de la sélection (`loop` vise 99 %). Sinon, un amendement daté avant tout run.
- [ ] `mix` (lancé par `loop mix`, ou seul : `scripts/distill-glm.sh mix`) écrit `data/distill/glm.jsonl` et le taux d'accord par source (`data/distill/agreement.json`). À recopier dans le rapport avec le nombre d'appels et de jetons. Les essais du 29/10/2026 donnaient 15 accords sur 20, puis 19 sur 26.
- [ ] Remplacer « 747,198 selected » et « the agreeing labels » par les vrais nombres dans `DATA_LICENSES.md` (section s6-nano / s6-micro).

## 2. Le s6 généraliste : pilote et ablation pico, puis s6-nano et s6-micro (`docs/s6-prereg.md`, `docs/s6-gen-prereg.md`, `docs/SPEC-S6.md` §15)

Pré-entraînement sur `data/text-open,data/text` (amendement 1), décisions sur `data/decisions,data/games,data/numeric`.

### 2.1 Pilote de learning rate pico (environ 2 h 30)

- [ ] `cargo xtask s6lr`, soit 6 runs : `softmax-adamw` et `parallax-muon` à ×0,5, ×1 et ×2. La sortie est `runs/s6lr/summary.json`.
- [ ] Appliquer la règle §2 : garder ×1 sauf si une autre échelle bat la précision ternaire finale (`val`) d'au moins **0,01**, pour chaque optimiseur.
- [ ] Si le choix tombe au bord (×0,5 ou ×2) : une seule extension (×0,25 ou ×4), **par amendement daté avant de la lancer**.

### 2.2 Ablation pico (environ 15 h)

- [ ] `cargo xtask s6ab --adamw-lr-scale <a> --muon-lr-scale <m>`, avec les échelles du pilote : 11 bras × 3 graines (7, 11, 13). La sortie est `runs/s6ab/summary.json`.
- [ ] Ne rien changer aux données ni aux budgets entre le pilote et l'ablation.

### 2.3 Décider ce qui est adopté (`docs/s6-prereg.md` §4)

- [ ] Muon v2 contre AdamW : gain significatif (≥ 0,01 et plus grand que l'écart-type des deux bras), sans perte arena (> 5 % sur un jeu). Le gain exigé passe à 0,02 si le surcoût en `train_minutes` dépasse 15 %.
- [ ] Parallax : `parallax-muon` contre `softmax-muon`, même règle.
- [ ] XSA : `xsa-muon` contre `softmax-muon`, même règle.
- [ ] SODA, KD de sortie, EMA, StableQAT et curriculum, lus contre `parallax-muon` : gain ≥ 0,005, sans perte arena, ECE pas plus de 0,01 pire.
- [ ] Sans polish : non inférieur (≥ −0,005) **et** au moins 2 % plus rapide.
- [ ] **Un commit, avant le pilote de s6-nano** (`docs/s6-gen-prereg.md` §2) :
  - dans `configs/s6-nano.json` et `configs/s6-micro.json`, `parallax` reste à `true` si la règle 2 l'adopte, sinon il passe à `false` (et corriger alors les nombres de `s6_parameter_counts`) ; `"xsa": true` si la règle 3 adopte XSA ;
  - noter `OPT` **sans taux** (`--no-muon` si AdamW est gardé, `--wd-mode soda`, `--ns-polish 0`), `LR_SCALE` (l'échelle du pilote pour l'optimiseur retenu), `PRETRAIN` et `DECIDE`. s6-secrets reprend `OPT` et `LR_SCALE`.
- [ ] Décider **avant le run** si `DECIDE` contient `--ema`. Le checkpoint exporté en dépend (`last-ema` ou `last`).

### 2.4 s6-nano (1,57M paramètres, environ 7 h)

Fenêtres 512 / 160 depuis l'amendement 2 de `docs/s6-gen-prereg.md`. Mesuré le 29/10/2026 avec `tok-s6-2048` : 1,8 % des états du bench tronqués, aucune tête. Avec 256 / 128, c'était 71,6 % des états. Labels GLM nécessaires (§1.1) : `train-s6.sh` refuse de démarrer sans `data/distill/glm.jsonl`.


- [ ] Contrôle à blanc : `SIZE=nano DRY_RUN=1 OPT="…" LR_SCALE=… PRETRAIN="…" DECIDE="…" scripts/train-s6.sh lr train dagger exit eval`. Les chemins et les taux doivent être les bons, avec des points décimaux.
- [ ] `SIZE=nano … scripts/train-s6.sh lr` : le pilote ×0,5, ×1 et ×2 autour du taux pico déplacé par μP, √(96/144) = ×0,816 (20 + 20 min chacun). On garde le centre sauf si un autre taux gagne ≥ **0,01** en `val`. Pas d'extension. La sortie est `runs/s6-nano-lr/summary.json`.
- [ ] `SIZE=nano … scripts/train-s6.sh train dagger exit eval`. Les étapes :
  - pré-entraînement 90 min ;
  - décision 90 min (`decisions=0.25,distill=0.30,games=0.35,numeric=0.10`) ;
  - 3 rounds DAgger de 15 min (oracle 0,2 → 0,1 → 0,05, 30 % de décisions gardées, dont 20 % du mélange GLM) ;
  - seuils CRC (α = 0,02) ;
  - évaluation et porte (`scripts/s6-gate.py`, résultat dans `runs/s6-nano/eval/results.md`).
- [ ] Porte : `typed_decisions_bench` **> 0,398**, aucun jeu à −5 % de s3-pico, ECE **≤ 0,08**. Lue à τ = 1. Publié aux seuils CRC si elle y tient aussi.

### 2.5 s6-micro (5,85M paramètres, environ 11 h)

- [ ] Même contrôle à blanc avec `SIZE=micro`. Fenêtres 384 / 128 : 4,2 % des états du bench tronqués, aucune tête (mesuré le 29/10/2026 avec `tok-s6-4096`).
- [ ] `SIZE=micro … scripts/train-s6.sh lr` (centre √(96/256) = ×0,612), puis `train dagger exit eval`. Les budgets : pré-entraînement 180 min, décision 180 min (batch 64), 3 rounds DAgger de 20 min.
- [ ] Porte : `typed_decisions_bench` **≥ 0,45**, aucun jeu à −5 % de s3-pico, ECE **≤ 0,08**. Dire sans ambiguïté si l'objectif de **0,50** est atteint.
- [ ] Le rapport, quel que soit le résultat :
  - les deux tailles face à s3-pico (0,382), s4-micro (0,343), oscar-1-17m (0,6775) et s5-lite (0,696) ;
  - la courbe nano → micro ;
  - MASSIVE en/fr zéro-shot, Banking77 « dans la distribution » ;
  - la part tronquée et la latence WASM.

## 3. s6-secrets-pico (`docs/s6-secrets-prereg.md`)

Après 2.3, parce qu'il reprend l'`OPT` et le `LR_SCALE` adoptés (amendement 2 de son pré-enregistrement). Pré-entraînement sur `data/text-open,data/text` (amendement 1).

- [x] `scripts/train-secrets.sh data` : jev `data/secrets` (50 000 textes, plus `secret_bench` de 2 000) et score du scanner à règles. Celui-ci doit redonner **0,977** (secret) et **0,787** (service), sinon on s'arrête et on comprend pourquoi. Fait le 29/10/2026 : 183 712 questions train, 9 546 val, 7 984 au bench, et le scanner redonne exactement 0,977 (précision 0,966, rappel 0,989) et 0,787. Résultat dans `runs/s6-secrets-pico/baseline.txt`.
- [ ] Décider **avant le run** si `DECIDE` contient `--ema`, ce qui fixe le checkpoint exporté.
- [ ] `OPT="…" LR_SCALE=… DECIDE="…" scripts/train-secrets.sh train eval` : pré-entraînement 8 min en séquences de 256 jetons, décision 15 min sur `data/secrets` seul, calibration, export ternaire, `s2-eval` sur `secret_bench`. Graine 0, une seule fois.
- [ ] Porte : exactitude secret **> 0,977** et service **> 0,787**, en ternaire, à pleine profondeur.
- [ ] Le rapport donne aussi kind, line, la validation, la sortie anticipée, la taille et la latence, quel que soit le résultat.

## 4. s6-cast-pico (`docs/s6-cast-prereg.md`, `docs/SPEC-S6-CAST.md`)

Indépendant des autres s6, il ne lit pas de texte.

- [x] `scripts/train-cast.sh data`. Relancé le 29/10/2026 : tout a sauté. `data/series/`, les 194 fichiers de fenêtres GIFT-Eval (`data/gift-eval/`) et les références seasonal naive et TinyCast (`data/gift-eval-out/`) sont là.
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
