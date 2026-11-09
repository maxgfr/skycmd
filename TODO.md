# TODO : entraîner les modèles s6, et le routage Système 1 / Système 2 de LFM3-1.7B

## À lancer maintenant : la tranche 1 du grand run (la recette s6), en mode bridé

> **Le mode bridé (décidé le 09/11/2026 : le Mac chauffe trop, et il ne reste pas ouvert en permanence).** `SKY_DUTY=0.5` fait travailler le GPU la moitié du temps : deux fois moins de puissance moyenne, pour deux fois plus de temps. Le nombre de pas, les graines et les résultats ne changent pas, et les pauses ne comptent ni dans `train_minutes` ni dans les budgets en minutes (`docs/s6-prereg.md`, amendement 5).
>
> **Des repos complets, en plus (`SKY_COOL=45:15`).** Chaque heure, 45 min de travail bridé puis 15 min sans aucun calcul GPU, pour que la machine refroidisse vraiment. Ces repos non plus ne comptent ni dans `train_minutes` ni dans les budgets.
>
> **Cette tranche n'entraîne pas les s6.** Elle choisit leur recette, c'est-à-dire quelles optimisations garder, et ne produit aucun modèle final. Les 5 modèles s6 demandent environ 48 h de GPU, soit environ 128 h avec la bride et les repos. Les durées ci-dessous comptent le temps où le Mac est allumé et ouvert, pas le temps calendaire :
>
> | tranche | ce qui tourne | à pleine vitesse | bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`) | prête ? |
> |---|---|---|---|---|
> | 1 | pilote + ablation pico → recette (`scripts/s6-recipe.sh`) | ~17 h 30 | **~47 h** | **oui** |
> | 2 | s6-ear, s6-secrets puis s6-nano | ~11 h | ~29 h | non : il manque des amendements et `train-all.sh` (§9.2) |
> | 3 | s6-micro | ~11 h | ~29 h | non (§9.2) |
> | 4 | s6-cast | ~9 h | ~24 h | non (§9.2) |
>
> Puis LFM3 S1/S2 et LFM3 fine-tuné (§9.4, tranches 5 et suivantes). Ce qui manque pour les tranches 2 à 4 se fait dans une session Claude Code pendant ou après la tranche 1, sans GPU.

Tout est prêt pour cette tranche : le plantage au pas 74 du pré-entraînement est corrigé (`non-finite loss`, un nombre du texte trop grand pour un `f32` ; `docs/s6-prereg.md`, amendement 5). Les labels GLM sont finis à 100 %, les amendements 4 et 5 de `docs/s6-prereg.md` sont écrits, et le dry-run passe. Cette tranche ne lit pas encore `data/distill/glm.jsonl`, qui sert à la tranche 2, mais le mélange est déjà prêt. Dans un terminal, **de préférence sur secteur** :

```
cd ~/Documents/Code/skycmd && git pull
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # Metal (absent du profil du shell)
caffeinate -i scripts/s6-recipe.sh 2>&1 | tee -a runs/s6-recipe.log
```

- **Ce que ça fait, sans aucune décision à la main :**
  - le pilote du taux d'apprentissage (6 runs, environ 7 h bridé) ;
  - l'extension, seulement si un choix tombe au bord de la grille ;
  - l'ablation pico aux échelles du pilote (11 bras × 3 graines, environ 40 h bridé) ;
  - puis `scripts/s6-adopt.py adopt`, qui écrit la recette dans `runs/recipe/recipe.json`.

  Au total, environ **47 h de Mac ouvert**, à étaler sur plusieurs jours.
- **Régler la chauffe**, avec deux variables à exporter avant la commande (on peut les changer à chaque relance, elles n'entrent pas dans les résultats) :
  - `SKY_DUTY` est la part du temps où le GPU travaille : `0.3` chauffe moins et prend environ 78 h, `0.7` chauffe plus et prend environ 33 h, `1` est la pleine vitesse ;
  - `SKY_COOL` est le cycle en minutes, travail puis repos : `40:20` refroidit plus longtemps (environ 53 h), et `SKY_COOL=` (vide) supprime les repos. Le journal affiche `cooling down for … s` à chaque repos ; le run n'est pas bloqué.
- **Fermer le Mac quand on veut :**
  - **rabattre l'écran** met le run en pause, et il repart seul à l'ouverture : aucune commande à retaper ;
  - **éteindre ou redémarrer** arrête le run : relancer la même commande. Les étapes finies sont sautées, seule l'étape en cours repart de son début (30 min au plus à perdre en mode bridé) ;
  - une erreur, par exemple du GPU au réveil, relance toute seule le script après 60 s. Il abandonne après 5 échecs de suite sans étape finie, et le journal le dit ;
  - **ne pas utiliser `pmset disablesleep`** dans ce mode : le Mac tournerait écran rabattu, le pire cas pour la chaleur.
- **Pour suivre :**
  - `tail -f runs/s6-recipe.log` ;
  - les journaux de chaque étape dans `runs/s5-logs/` ;
  - `ls runs/s6lr runs/s6ab`.
- **Avant, si on veut voir les commandes sans rien lancer :** `DRY_RUN=1 scripts/s6-recipe.sh`.
- **Après :** ouvrir une session Claude Code pour :
  - lire `runs/recipe/recipe.json` et commiter la recette ;
  - écrire ce qui reste au §9.2 avant la nuit 2 : les amendements s6-ear, s6-cast, s6-gen, s6-secrets et LFM3, `train-all.sh` et le mode local de `lfm3-ft.sh`.
- **À ne pas faire en même temps :** LFM3 S1/S2 (§6) et s6-ear (§8). Ils utilisent le même GPU.

## Où on en est (06/11/2026)

- **Le prochain run : tout entraîner, au mieux (§9), sur le M5.** Le 05/11, l'utilisateur avait d'abord parlé de louer des GPU, puis a tranché : « finalement je ferai tout sur mon pc ». Il n'y a donc **ni Mac loué ni HF Jobs**. **La nuit 1, la recette s6, se lance le 06/11** (voir plus haut). Le §9 dit ce qui est prêt et ce qui reste à faire pour les nuits suivantes.
  - **Les 5 modèles s6**, avec les optimisations que le pilote et l'ablation adoptent (règles appliquées automatiquement) : environ 48 h de GPU.
  - **LFM3-1.7B** : le routeur S1/S2 (§6) et le fine-tuning (§6.1), en local. Il faut d'abord les amendements du §9.2 : le §6.1 et l'amendement 1 du §6 visaient HF Jobs.
  - **Depuis le 09/11/2026, tout tourne en mode bridé (`SKY_DUTY=0.5 SKY_COOL=45:15`)** : le Mac chauffait trop. Le GPU travaille la moitié du temps, avec 15 min de repos complet par heure, donc tout prend environ 2,7 fois plus de temps sans changer les résultats (voir plus haut). On découpe en tranches (§9.4) : tout reprend là où ça s'était arrêté.
  - **Le Mac peut se mettre en veille.** Rabattre l'écran suspend le run, qui repart à l'ouverture : ce qui s'est passé dans la nuit du 04 au 05/11 (« Clamshell Sleep » à 22:12) n'est plus un problème, cela coûte seulement du temps. `caffeinate -i` empêche seulement la veille d'inactivité quand l'écran reste ouvert.
- **Labels GLM-5.3-flash (§1.1) : finis le 05/11/2026 à 19 h 40, à 100 %.** 747 077 labels et 121 textes refusés par la modération, soit les 747 198 records.
  - L'accord avec la bonne réponse est de 560 238 sur 747 077 (0,750).
  - `data/distill/glm.jsonl` est écrit : c'est le mélange final, que `train-s6.sh` attend.
  - `skycmd-glm-labels` est publié en version complète (§1.2).
  - La boucle s'est arrêtée seule à 19:22, avec 267 records restants (99,96 %, au-dessus du seuil de 99 %). Deux passes `label` les ont finis, puis `mix`.
- **Données s6-ear téléchargées (§8)** : les comptes sont dans `DATA_LICENSES.md`, et il y a bien 50 mots réservés par langue, sans amendement.
- **Nos 14 jeux de données sont publics sur Hugging Face (§1.2)** : tout ce qu'un modèle lit et tout ce qu'on a créé nous-mêmes (jeux, trajectoires de nos modèles face à l'oracle, séries synthétiques, secrets, évaluations écrites à la main), les labels GLM compris (marqués comme générés par IA). Seuls manquent l'audio de s6-ear (31 Go, reconstruit par `sky-data fetch-audio`), le jeu gated et GIFT-Eval.
- **Données s6 prêtes (§1)** : le corpus ouvert, les tokenizers 2048 et 4096, et les tests zéro-shot. Celles de s6-secrets et s6-cast aussi (§3, §4).
- **Recette s6** : le pilote et l'ablation pico décident (§2). s6-nano passe en fenêtres 512 / 160, et le taux de chaque taille est centré par μP.
- **Les entraînements s6** : la nuit 1 (pilote, ablation, recette) est prête à lancer. s6-ear, s6-secrets, s6-nano, s6-micro et s6-cast suivent les nuits 2 à 4, une fois le §9.2 fait.
- **Lu et non adopté** (04/11) : ZHA, l'attention hiérarchique de Zyphra (arXiv 2610.42112). Elle ne bat l'attention dense qu'au-delà d'environ 48K tokens et marche mal sous 1B paramètres ; nos états font 64 à 512 tokens. Les raisons sont dans `docs/research.md` (« Not adopted yet »).
- **LFM3 fine-tuné (§6.1)** : pré-enregistrement figé (`docs/lfm3-ft-prereg.md`), scripts et tests prêts, **rien n'a tourné**. Les scripts visent encore Hugging Face Jobs. Depuis le 05/11, tout se fait sur le M5 : il faut l'amendement 1 de `lfm3-ft-prereg`, l'amendement 2 de `lfm3-s1s2-prereg` (qui annule le passage à distance) et le mode local de `lfm3-ft.sh` (§9.2).
- **Issue #75 (Weight-Space Programming sur LFM3) fermée** au profit du routage S1/S2 (§6). Les idées de l'article sont reprises là où elles sont testables, sur nos petits modèles et avec l'arena : édition ciblée sans oubli, initialisation en une fois, poids flexibles en temps réel avec garde-fou conformal, audit des poids (§7). Son worktree a été supprimé le 06/11/2026 : il ne contenait aucun travail non mergé.

> **La labellisation GLM est finie (05/11/2026, §1.1).** Il n'y a plus rien à relancer. `scripts/distill-glm.sh count` doit afficher 0 record à labelliser.

Cinq modèles ont leur code prêt et ne sont pas encore entraînés : **s6-nano** et **s6-micro** (le s6 généraliste : les jeux et des questions en texte libre), **s6-secrets-pico**, **s6-cast-pico** et **s6-ear-nano** (l'audio, §8). Chacun a un pré-enregistrement figé. **s6-pico n'est plus un modèle final** (`docs/s6-prereg.md`, amendement 2) : son pilote et son ablation restent, parce qu'ils choisissent la recette des deux tailles. On suit chaque pré-enregistrement à la lettre :

- un amendement daté (§7, §8 pour `docs/s6-gen-prereg.md`) précède le run qu'il concerne ;
- on ne relance rien pour « voir » : seul un échec technique justifie un nouveau run, et il est noté dans le rapport ;
- on ne choisit jamais sur un checkpoint intermédiaire ;
- un modèle qui échoue à sa porte n'est pas publié, et le rapport le dit.

Les autres configs `s6-*` de `configs/` ne sont pas des modèles à entraîner :

- `s6-femto`, `s6-femto-xsa` et `s6-cast-femto` sont de minuscules configs aux poids aléatoires, pour les tests golden de `sky-infer` (`tests/golden_s6.rs`, `golden_s6x.rs`, `golden_cast.rs`) et les tests unitaires ;
- `s6-pico` et `s6-pico-xsa` sont les configs du pilote et de l'ablation (§2.1, §2.2), pas des modèles à publier.

Ordre (§9, tout sur le M5, une tranche par nuit) : labels GLM (§1.1, finis le 05/11) → pilote et ablation pico → recette (`scripts/s6-adopt.py`) → s6-ear → s6-secrets → s6-nano → s6-micro → s6-cast → LFM3 S1/S2 (§6) → LFM3 fine-tuné (§6.1). Les durées sont mesurées ou estimées sur le M5.

## 0. Prérequis

- [x] Toolchain Metal : `xcodebuild -downloadComponent MetalToolchain`, et `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Présente le 29/10/2026 (`xcrun metal --version` : 32023.921).
- [x] `export SKYCMD_STORE=$PWD` (le checkout principal) pour `cargo xtask` : `scripts/s6-recipe.sh` le fait tout seul.
- [x] `uv` installé (s6-cast : `scripts/gift-eval.py` tourne sous Python 3.12).
- [x] `node` et `wasm-pack` (présents le 29/10/2026) : l'étape `eval` de `scripts/train-s6.sh` construit `site/public/pkg` s'il manque (`scripts/build-wasm.sh`) pour mesurer la latence WASM.
- [x] Place disque : **95 Go libres** le 06/11/2026 (15 Go le 29/10/2026), après suppression du cache de build debug (`target/s3-train/debug`, 9,2 Go, produit par `cargo test` et clippy ; il se reconstruit tout seul). Les scripts d'entraînement n'utilisent que `target/s3-train/release`. Les runs écrivent dans `runs/` et `models/`, et les checkpoints de s6-micro (5,85M paramètres, état d'optimiseur compris) pèsent bien plus que ceux de pico. Revérifier avant 2.5, et supprimer à nouveau `target/s3-train/debug` après des `cargo test`.
  - **Mesuré le 09/11/2026 : 94 Go libres, pour environ 55 Go à écrire en tout.** L'essentiel, ce sont les caches du texte tokenisé : 866 Mo par pré-entraînement, identiques d'un run à l'autre, soit environ 43 Go pour la cinquantaine de runs. Un checkpoint pico pèse moins de 5 Mo. S'il faut de la place, `find runs -type d -name cache -prune -exec rm -rf {} +` les supprime une fois les étapes finies : ils ne servent plus aux résultats, seulement à recharger plus vite une étape relancée.

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
- [x] **Lancer le tout, sans surveillance :** `caffeinate -i scripts/distill-glm.sh loop mix`, dans un terminal qui reste ouvert (ou `mkdir -p runs && nohup caffeinate -i scripts/distill-glm.sh loop mix > runs/distill-glm.log 2>&1 &`).
  - **En cours depuis le 29/10/2026 à 16:28**, avec `nohup` (journal : `runs/distill-glm.log`). Pour savoir s'il tourne encore : `pgrep -fl distill-glm`. Cocher quand `loop` affiche « coverage reached » et que `mix` a écrit `data/distill/glm.jsonl`.
  - **Le 04/11/2026 à 19:15** : 539 229 sur 747 198 (72 %), 63 refus. La boucle s'était arrêtée à 11:38 (relance des sessions) ; relancée à 12:32, environ 3,3 labels/s depuis.
  - **Le 03/11/2026** : 350 949 labels sur 747 198 (47 %). Le redémarrage de la machine du 02/11 à 21:17 avait arrêté la boucle ; elle a été relancée le 03/11. Il reste 396 249 records, soit 19 813 appels.
  - **Refus de la modération z.ai** (HTTP 400, code 1301, amendement 3) : un lot refusé est coupé en deux jusqu'à isoler les textes refusés. Ceux-ci vont dans `data/teacher/glm_labels.refused.jsonl` et ne sont plus envoyés. `count` et le journal donnent leur nombre.
  - **4 workers** (`JOBS=4`), c'est le maximum utile. Mesuré le 29/10/2026 : z.ai plafonne vers 2,5 labels/s quel que soit le nombre de workers, et dès 8 workers une partie des appels part en relances (8 workers : 10 relances sur 24 appels ; 16 workers : 22 sur 32).
  - **Environ 80 h de labellisation**, réparties sur les fenêtres de 5 h de l'abonnement z.ai, avec de l'ordre de 100 à 200 M jetons en entrée et 55 M en sortie.
  - Quand une fenêtre est épuisée, chaque worker s'arrête après 3 lots ratés de suite. `loop` fait une pause de 30 min (`PAUSE=1800`) et reprend là où il s'était arrêté, jusqu'à ce qu'il reste au plus 1 % de la sélection (`COVER_MISS=1`). Il abandonne après 24 pauses sans progrès (`TRIES=24`), soit 12 h.
  - Avancement à tout moment : `scripts/distill-glm.sh count` (sans appel).
  - Si `loop` s'arrête avant la fin (machine éteinte, `TRIES` atteint), relancer la même commande : rien n'est perdu.
- [x] **Couverture : 99,98 %** (747 077 sur 747 198, les 121 refus comptés comme non labellisés), donc pas d'amendement. Les labels doivent couvrir au moins **95 %** de la sélection (`loop` vise 99 %, les textes refusés comptant comme traités). Les textes refusés par la modération comptent comme **non labellisés** pour ces 95 %. Sinon, un amendement daté avant tout run. Le rapport donne le nombre de refus par source (`glm_labels.refused.jsonl`).
- [x] **Fini le 05/11/2026 :** 560 238 accords sur 747 077 (0,750), le détail par source est dans `data/distill/agreement.json`. `mix` (lancé par `loop mix`, ou seul : `scripts/distill-glm.sh mix`) écrit `data/distill/glm.jsonl` et le taux d'accord par source (`data/distill/agreement.json`). À recopier dans le rapport avec le nombre d'appels et de jetons. Les essais du 29/10/2026 donnaient 15 accords sur 20, puis 19 sur 26.
- [x] Remplacer « 747,198 selected » et « the agreeing labels » par les vrais nombres dans `DATA_LICENSES.md` (section s6-nano / s6-micro).

### 1.2 Publier nos jeux de données sur Hugging Face

Demandé le 03/11/2026 : tous nos jeux de données sur le Hugging Face de l'utilisateur, **en public**, les labels GLM compris (en partiel maintenant, puis mis à jour à la fin). Un dataset repo par jeu, une config parquet par source, et une carte qui donne la source, la licence, la commande qui a produit les fichiers et le commit.

- [x] `scripts/hf-publish-data.py` (`--dry-run`, `--only`, `--private`, `--user`). Il assemble chaque jeu sous `target/hf-data/<repo>/` : un parquet par source, `MANIFEST.json` (le sha256 des fichiers d'origine) et la carte `README.md`. Le dry-run du 03/11/2026 donne :
  - `skycmd-decisions` : 27 sources, 823 387 lignes, 183 Mo ;
  - `skycmd-games` : 280 000 lignes ;
  - `skycmd-numeric` : 100 000 ;
  - `skycmd-secrets` : 201 242 ;
  - `skycmd-commands` : 32 341 ;
  - `skycmd-zero-shot` : 5 000.
- [x] **Publiés le 03/11/2026, en public**, sur `maxgfr` :
  - `skycmd-decisions` : 27 sources, 823 387 lignes ;
  - `skycmd-games` : 280 000 ;
  - `skycmd-numeric` : 100 000 ;
  - `skycmd-secrets` : 201 242 ;
  - `skycmd-commands` : 32 341 ;
  - `skycmd-zero-shot` : 5 000.

  Les liens sont dans le README, `DATA_LICENSES.md` et la page Sources.
- [x] **Le 05/11/2026 : tout ce qu'un modèle lit est publié.** Les 6 jeux du 03/11 n'ont pas changé : sha256 identiques à leur `MANIFEST.json` sur le Hub. Quatre nouveaux dépôts, en public, ont été ajoutés :
  - `skycmd-series` : les séries de s6-cast, 8 sources, 10 789 séries ;
  - `skycmd-text` : le texte MLM (`data/text` et `data/text-open`), 11 sources, 112 671 documents, plus les 3 tokenizers ;
  - `skycmd-ear` : les questions de s6-ear, l'index des clips et les codes Mimi, 1 233 211 lignes ;
  - `skycmd-lfm3-math` : les 600 questions de calibration et les 1 000 de test du routeur LFM3.
  - puis, à la demande de l'utilisateur (« sur Hugging Face, les données qu'on a créées nous-mêmes »), deux jeux de plus :
    - `skycmd-cast-synth` : un échantillon fixe de nos 4 générateurs de séries synthétiques et du mélange d'entraînement, 8 000 séries de 2 048 points (`sky-train cast-synth --seed 1`). s6-cast s'entraîne sur un flux sans fin des mêmes générateurs, pas sur ces lignes ;
    - `skycmd-evals` : notre matériel d'évaluation écrit à la main, à savoir les 56 cas des démos, `defs.json` et `paraphrases.json` de s5.
  - et un 14e, le plus original, `skycmd-trajectories`, avec `scripts/gen-trajectories.sh` et le nouveau `sky-arena --trajectories`, CPU seulement, en quelques minutes. Ce sont des parties complètes des 4 jeux, jouées par l'oracle et par nos modèles publiés s1-pico, s2-nano, s2-nano-doom et s3-pico : 1 033 412 décisions sur 2 100 parties, 17 Mo en parquet.
    - À chaque décision, il y a l'état, les probabilités du modèle, la cible de l'oracle, l'action, la récompense et les événements ; il y a aussi une ligne par partie. Le jeu sert à l'imitation et au DAgger, à l'apprentissage par renforcement hors ligne et à la calibration.
    - L'accord avec l'oracle va de 0,77 à 0,96 sur flappy, rescue et dodge. Sur doom, il est de 0,91 à 0,92 pour s2-nano-doom et s3-pico, et presque nul pour s1-pico et s2-nano, qui n'ont jamais vu doom : ces échecs complets sont gardés, ils sont utiles en hors ligne.
    - s4-micro et s5-lite sont écartés : ce ne sont pas des modèles de jeu.

  Non republiés : `LocalLLaMA/typed-decisions` (gated), l'audio de s6-ear (31 Go de log-mel, que `sky-data fetch-audio` reconstruit) et GIFT-Eval (évaluation seulement).
- [ ] Vérifier le viewer du Hub sur chaque dataset quand l'indexation est finie (`https://datasets-server.huggingface.co/is-valid?dataset=maxgfr/<repo>`), et `load_dataset("maxgfr/skycmd-decisions", "boolq")`.
- **Règles appliquées par le script :**
  - chaque source garde sa licence, et les sources share-alike restent share-alike ;
  - `LocalLLaMA/typed-decisions` (le train et le bench) n'est **pas** republié : ce jeu est gated sur le Hub, et le republier en accès libre contournerait la barrière ;
  - le texte de pré-entraînement et les clips audio ne sont pas republiés. On publie leur recette : la commande `sky-data`, la graine et les sha256.
- [x] **Labels GLM-5.3-flash publiés en partiel**, sur décision de l'utilisateur (« c'est open source, tu peux publier »), malgré la clause §1(f)(xii) des [conditions de l'API z.ai](https://docs.z.ai/legal-agreement/terms-of-use). Cette clause interdit d'utiliser les sorties pour « training, labeling, fine-tuning … related to external models ». La licence MIT couvre les poids, pas l'API.
  - Le repo est `maxgfr/skycmd-glm-labels`. Il est marqué comme contenu généré par IA, comme le demande le §III.5(d).
  - Il a deux configs : `agree` (les records dont l'argmax du professeur est la réponse juste, avec pour cible les probabilités du professeur) et `labels` (tous les labels bruts).
  - Il ne contient ni les textes refusés par la modération ni `typed_decisions_train`.
- [x] **Republié le 04/11/2026 à 72 %** : `labels` 535 278 lignes, `agree` 403 040 (93,7 Mo). L'accord avec la bonne réponse est de 405 990 sur 539 368 (0,753) sur tout ce qui est labellisé, `typed_decisions` compris.
- [x] **`skycmd-glm-labels` publié en version complète le 05/11/2026** (`--glm-final`) : `labels` 741 437 lignes et `agree` 556 175, sans `typed_decisions_train`, qui est gated. Les commandes, depuis le checkout principal :
  - `target/s3-train/release/sky-data teacher mix-labels --labels data/teacher/glm_labels.jsonl --from data/decisions --out <tmp>/glm-mix --alpha 0` ;
  - puis `uv run scripts/hf-publish-data.py --glm <tmp>/glm-mix --glm-date "<date>" --glm-final`.

  Ne **jamais** écrire ce mélange partiel dans `data/distill` : `train-s6.sh` le prendrait pour le mélange final.
- **Vitesse de la labellisation sur z.ai** (03/11/2026). Le quota n'est pas la limite, la concurrence l'est. Les mesures :
  - 4 workers × 20 records par appel : environ 240 labels/min, sans erreur ;
  - 6 workers × 50 : erreurs HTTP 429 « Rate limit reached for requests » (code 1302) ;
  - 4 workers × 50 : environ 150 labels/min, plus lent (le temps de génération d'un appel croît plus vite que le nombre de records) ;
  - 5 workers × 20 : environ 240 labels/min, comme à 4 workers, avec des erreurs 429.

  - **Le 04/11/2026 : le prompt numérote les items (1 à 20) et passe en JSON compact.** Avant, GLM recopiait pour chaque item un identifiant de record de 16 chiffres hexadécimaux. Mesuré sur le même lot de 20 (deux essais) : 913 et 1 035 jetons générés au lieu de 1 337 et 1 449, et 11,3 et 13,7 s par appel au lieu de 16,2 et 18,0 s. `parse_labels` remet les identifiants de records. La consigne et les probabilités demandées ne changent pas. En continu, 4 × 20 : **4,7 labels/s** au lieu de 3,3 à 3,6 (+31 %), et 3,4 % des appels perdent un label au lieu d'environ 6 % (le journal n'écrit une ligne « 19 / 20 valid » que pour les appels incomplets).

  - **Le 05/11/2026, 6 workers × 20 avec le nouveau prompt :** 3,99 labels/s en 10 min, contre 4,7 à 4 workers, avec des erreurs HTTP 429 (code 1302), et un worker qui s'arrête après 3 échecs de suite. Retour à 4 workers.

  **La limite de z.ai porte sur le nombre de requêtes en parallèle (4 utiles) : on gagne en raccourcissant les appels, pas en ajoutant des workers.** Au 03/11, avec l'ancien prompt, le plafond était d'environ 4 labels/s. On garde le défaut, 4 × 20, relancé le 03/11 à 20:30. Il reste environ 333 000 records, soit environ 23 h de labellisation, plus les pauses entre fenêtres. Pour relancer : `nohup caffeinate -i scripts/distill-glm.sh loop mix >> runs/distill-glm.log 2>&1 &`.
- [ ] Plus tard : le jev audio de s6-ear (§8). Les données sont prêtes depuis le 04/11 (31 Go de log-mel : voir s'il vaut mieux ne publier que les records et la recette). On y met les records et les log-mel, pas les clips (les sources restent sous leur licence CC-BY-4.0, avec attribution).

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

## 6. LFM3-1.7B : Système 1 d'abord, Système 2 seulement si besoin (`docs/lfm3-s1s2-prereg.md`)

**L'idée.** LFM3-1.7B réfléchit par défaut, et trop réfléchir peut le faire renoncer à une bonne première réponse. Sans réflexion, il peut aussi partir en boucle : l'essai de fumée l'a montré sur une question de GSM8K. On ne touche pas aux poids, on ne l'entraîne pas.
1. Il répond d'abord sans réfléchir (**Système 1**), et une petite sonde mesure sa confiance.
2. Il ne réfléchit, avec un budget borné, que si la confiance est basse (**Système 2**).
3. Le seuil vient du **contrôle conformal du risque** (α = 0,02), comme les sorties anticipées de skycmd : en espérance, au plus 2 points perdus par rapport à « toujours réfléchir ».

Le but est autant de précision, voire plus, pour bien moins de calcul.

**Prêt le 03/11/2026 :**
- `scripts/lfm3-s1s2.sh` (étapes `data smoke calib test route`) et `experiments/lfm3-s1s2/s1s2.py` ;
- le modèle `LiquidAI/LFM3-1.7B-MLX-4bit` (rév. `deb2986b`, 0,97 Go) déjà téléchargé ;
- les données écrites dans `data/lfm3/` : 600 questions de calibration, 1 000 de test ;
- un essai de fumée de 3 questions par mode, réussi (`runs/lfm3-s1s2-smoke`). Débit : 98 jetons/s sans réflexion, 80 avec, environ 4,5 s par question.

- **Le 05/11/2026, deux décisions de l'utilisateur se sont succédé.**
  - D'abord « pas en local » : le M5 chauffe trop. L'amendement 1 du pré-enregistrement a fait passer le run à distance (bf16, Hugging Face Jobs, vLLM ou transformers).
  - Puis « finalement je ferai tout sur mon pc ». Il faut donc un **amendement 2** qui annule le 1 et revient au protocole d'origine : MLX-4bit, mlx-vlm et le M5 (§9.2).
- [ ] **La commande (§9.4, tranche 5), après l'amendement 2 :** `SKY_DUTY=0.5 SKY_COOL=45:15 caffeinate -i scripts/lfm3-s1s2.sh`. Elle enchaîne `data` (déjà fait, sauté), `calib`, `test` puis `route`.
  - Durée estimée à pleine vitesse : environ 1 h 30 pour s1, 2 h pour s2-256 et 5 à 6 h pour s2-1024, sur 1 600 questions, soit **environ 9 à 10 h, donc 24 à 27 h bridé**. La calibration passe en premier (600 questions, environ 3 h 30 à pleine vitesse).
  - Inférence locale sur le GPU du M5. `SKY_DUTY` ajoute une pause après chaque question. Le critère H3 compte des jetons, et `seconds` reste le temps de génération : le mode bridé ne change aucun des deux.
  - Tout reprend après une coupure : relancer la même commande, les questions déjà faites sont sautées.
  - La labellisation GLM (§1.1) peut continuer en même temps : elle ne fait que du réseau.
- [ ] **Au réveil**, lire `runs/lfm3-s1s2/report-calib.md` (calibration) puis `runs/lfm3-s1s2/report.md` (test, avec le verdict). Si la nuit n'a pas suffi, relancer `scripts/lfm3-s1s2.sh calib test route` : ça reprend.
- [ ] **Critères** (pré-enregistrés, §6 du pré-enregistrement) :
  - **H1** : réfléchir aide-t-il ? C'est un diagnostic, sans seuil.
  - **H2** : le routeur garde la précision de s2, à 0,02 près.
  - **H3** : le routeur divise le calcul au moins par deux.
  - **H4** : le routeur fait mieux que les deux modes, d'au moins 0,01.

  **Succès** = H2 et H3. **« LFM3 amélioré »** = H4 en plus.
- [ ] Rapport, quel que soit le résultat : les tableaux, l'oracle (le plafond d'un routeur parfait), la part routée vers s1 et la part de réflexions forcées. À mettre sur le site et dans un chapitre du tutoriel si c'est un succès.
- [ ] **Ensuite, seulement si le rapport le justifie :**
  - un budget de 64 jetons ;
  - IFEval ;
  - une sonde sur les états cachés au lieu des log-probas ;
  - le Weight-Space Programming de l'issue #75 (fermée), mais seulement si le diagnostic montre un défaut que l'édition des poids pourrait lever (des boucles, par exemple).

  Chacun passe par un amendement daté du pré-enregistrement.

### 6.1 LFM3 fine-tuné (`docs/lfm3-ft-prereg.md`) — **rien n'a tourné**

> **Changement du 05/11/2026 : tout se fait sur le M5, plus sur HF Jobs** (décision de l'utilisateur). Le texte ci-dessous décrit encore la version HF Jobs. Avant le moindre calcul, il faut l'amendement 1 de `docs/lfm3-ft-prereg.md` et l'amendement 2 de `docs/lfm3-s1s2-prereg.md`, qui remplace le 1 (§9.2), ainsi qu'un mode local dans `scripts/lfm3-ft.sh`.
> - **Génération et évaluation :** avec mlx-vlm, comme `experiments/lfm3-s1s2/s1s2.py` (mlx-lm ne charge pas LFM3, et vLLM ne tourne pas sur Mac).
> - **La LoRA :** avec transformers + peft sur MPS, ou avec l'entraîneur LoRA de mlx-vlm. Un smoke local de quelques minutes tranche, avec une règle écrite avant.
> - **La génération locale est lente** : environ 80 jetons/s, soit environ 4,5 s par question. Le plafond pré-enregistré de 1 000 questions par tâche s'applique donc d'office : environ 2 000 questions × 4 tirages, soit de l'ordre de 10 à 15 h, à étaler sur plusieurs nuits.

**L'idée.** Le routeur choisit quand réfléchir, mais ne répare pas le Système 1 : sans réflexion, LFM3-1.7B peut boucler jusqu'à la limite (l'essai de fumée). Ici, on apprend au modèle à répondre juste et court sans réfléchir. C'est un LoRA entraîné sur :
- les solutions rédigées de GSM8K et MATH (train, MIT) ;
- ses propres bonnes réponses en s2-1024 (auto-distillation).

Aucune question de calibration, de test ou de MMLU-Pro n'y entre. Les labels GLM ne servent **jamais** ici : les conditions de z.ai l'interdisent.

**Où ça tourne.** Tout passe par Hugging Face Jobs, rien en local à part `route` et `gate` (CPU). Les artefacts vont dans le dataset **privé** `maxgfr/skycmd-lfm3-ft-work`, le modèle dans le dépôt **privé** `maxgfr/skycmd-lfm3-1.7b-s1`. Rien n'est public, et les poids dérivés restent privés tant que l'utilisateur n'a pas dit oui.

**Prêt le 05/11/2026, pas lancé :**
- `scripts/lfm3-ft.sh` ;
- `experiments/lfm3-ft/` : `smoke.py`, `gen.py`, `sft.py`, `eval.py`, `common.py`, `route.py`, `gate.py` ;
- les tests CPU : `python3 experiments/lfm3-ft/test_common.py`.

`DRY_RUN=1 scripts/lfm3-ft.sh smoke gen sft eval` affiche les commandes `hf jobs` sans rien lancer. Sans `DRY_RUN`, chaque job payant affiche son coût estimé et maximal, puis attend « y ». **L'utilisateur confirme le coût avant chaque job payant.**

Depuis le checkout principal à jour de main (`data/lfm3` y est déjà) :

- [ ] **Smoke** (L4, 30 à 40 min, environ 0,50 $, au plus 0,80 $) : `scripts/lfm3-ft.sh smoke`. Le suivre avec `hf jobs ps`, puis `hf jobs logs <id>`. Il vérifie :
  - la révision du modèle de base, qu'il rend ;
  - si vLLM 0.33.0 charge LFM3 et s'accorde avec transformers ;
  - le gabarit avec et sans réflexion ;
  - les noms des modules LoRA ;
  - deux pas de TRL, la fusion et le rechargement.
- [ ] `scripts/lfm3-ft.sh pull`, puis lire `runs/lfm3-ft/smoke/report.json` : `revision`, `engine`, `lora_targets_match`, `trl.ok`.
- [ ] **Amendement 1 de `docs/lfm3-ft-prereg.md`** (daté, commité) : recopier la révision et le moteur. Les reporter dans `BASE_REV_PINNED` et `ENGINE_PINNED` en tête de `scripts/lfm3-ft.sh`. Si `lora_targets_match` ou `trl.ok` est faux, on s'arrête : l'écart passe d'abord par l'amendement.
- [ ] **Génération** (A100, environ 2 h, environ 5 $, au plus 10 $) : `scripts/lfm3-ft.sh gen`. Elle reprend si le job est coupé : relancer la même commande.
- [ ] **Entraînement** (A100, 1 à 2 h, environ 5 $, au plus 10 $) : `scripts/lfm3-ft.sh sft`. Puis `scripts/lfm3-ft.sh pull`, pour avoir `runs/lfm3-ft/sft/result.json` et le commit du modèle.
- [ ] **Évaluation** (A100, environ 1 h 30 par modèle, environ 7,50 $ pour les deux, au plus 15 $) : `scripts/lfm3-ft.sh eval`. C'est un job pour la base, un pour le modèle fine-tuné. Le job de la base est aussi le run à distance du routeur du §6 (amendement 1).
- [ ] **Lecture** (local, gratuit) : `scripts/lfm3-ft.sh pull route gate`. Elle donne `runs/lfm3-ft/report.md` et `gate.json`, et `runs/lfm3-ft/eval/<base|ft>/report.md` pour le routeur.
- [ ] **Portes** (pré-enregistrées, §6) :
  - **G1** : précision s1 sur le test ≥ base + 0,03, et arrêts par limite ≤ la moitié de la base ;
  - **G2** : précision s2-1024 ≥ base − 0,01, et réflexion ≤ 0,6 × la base. Elle est rapportée à part ;
  - **G3** : MMLU-Pro (500, s1) ≥ base − 0,02.

  **« LFM3 amélioré »** = G1 et G3. Le routeur (H1 à H4) est rapporté pour les deux modèles.
- [ ] Total estimé : **environ 18 $** (au plus 36 $ si chaque job va jusqu'à son délai). Pas de relance pour « voir », pas de choix de checkpoint, les échecs vont au rapport.

## 7. Poids flexibles : le Weight-Space Programming sur les petits modèles de skycmd (à faire après le §1.1 et le §6)

**Ce que l'article du MIT a montré, et ce qui reste une vision.** Les cinq promesses n'en sont pas au même point :

| promesse | montré dans l'article ? | ce qui le montre |
|---|---|---|
| auditer un modèle sans le faire tourner | en partie | un modèle reconnaît le jeu (4 sur 4) ou la langue d'un réseau en lisant ses poids. Mais le modèle flagorneur n'est pas détecté par les poids seuls (0 sur 4) : il faut voir ses activations (3 sur 3) |
| initialiser un modèle en une fois | oui, sur de petits réseaux | des réseaux MinAtar écrits d'un seul coup : Breakout 517 contre 25 pour PPO après 500M pas |
| ajouter ou retirer une capacité sans données ni oubli | retirer un défaut, oui ; ajouter une capacité, non | Qwen3.5-0.8B débarrassé de ses boucles. Sur Gemma 3 270M, aucun gain : « ça lève un obstacle, ça n'ajoute pas de capacité » |
| greffer un circuit d'une architecture dans une autre | non, c'est une perspective | — |
| auto-amélioration récursive | non, c'est une perspective | — |

**Pourquoi skycmd est le bon terrain.** Les résultats solides de l'article portent sur de petits réseaux de jeux (17 000 à 1,5M poids), la taille de nos modèles : s3-pico fait 445 000 poids, s6-nano 1,57M. L'arena donne en plus deux choses en microsecondes, en local, sans chauffer la machine :
- un **retour** pour chaque édition ;
- une **mesure de l'oubli catastrophique** sur les 4 jeux à la fois.

Personne n'a publié ça sur des modèles ternaires.

- [ ] **7.1 Édition ciblée sans oubli** (la promesse 3). GLM-5.3-flash, via `ccs` et z.ai, lit les poids et les activations de s3-pico publié. Il propose des éditions locales pour corriger un défaut mesuré, par exemple les vetos de Dodge ou les crashs de Flappy, et l'arena dit si on garde l'édition.
  - Règle : une édition est gardée seulement si le jeu ciblé gagne et qu'aucun autre jeu ne perd plus de 2 %.
  - Contrainte propre au ternaire : on édite le checkpoint flottant d'avant la QAT, puis on réexporte. Une édition directe dans {−1, 0, +1} est une variante à mesurer à part.
  - Bras : base, édition GLM à 10, 25 et 50 passes, une édition au hasard de même taille, et une baseline sans LLM (différence de moyennes des activations).
  - À lancer après la labellisation du §1.1, qui utilise le même quota z.ai.
- [ ] **7.2 Initialisation en une fois** (la promesse 2). GLM écrit les poids de la tête de décision d'une politique de jeu à partir de la description du jeu et des traits d'état. On mesure combien de pas d'entraînement cela fait gagner pour atteindre la qualité de s3-pico. Le but est un coût d'optimisation réduit, pas un modèle sans entraînement.
- [ ] **7.3 Des poids flexibles en temps réel** (ton idée : « un modèle qui a des poids flexibles »). Un petit adaptateur de bas rang par-dessus le modèle ternaire figé, mis à jour pendant le jeu dans le navigateur. Chaque fois que le réflexe de sécurité (le veto) corrige le modèle, l'adaptateur apprend de cette correction.
  - **Le garde-fou de skycmd.** Le contrôle conformal du risque décide si l'adaptateur reste actif : s'il dégrade la qualité au-delà de α par rapport au modèle figé, on revient au modèle figé. Ce sont des poids qui bougent, mais avec une garantie.
  - C'est la version mesurable de « l'auto-amélioration ». Les poids de base ne bougent jamais, donc on peut toujours revenir en arrière.
  - Les idées voisines dans la littérature : les « fast weights », l'entraînement au moment du test, les mémoires neuronales. Ce sont des souvenirs de modèle, à vérifier par une recherche avant le pré-enregistrement.
- [ ] **7.4 Audit des poids** (la promesse 1). On implante une porte dérobée dans une copie de s6-secrets, par exemple un déclencheur qui fait dire « pas de secret ». On regarde si GLM la trouve en lisant les poids, puis les activations. Il faut un petit entraînement local pour implanter la porte, donc c'est pour plus tard.
- Hors de portée pour l'instant : la greffe de circuits entre architectures (promesse 4) et l'auto-amélioration récursive libre (promesse 5). L'article ne les montre pas, et 7.3 en est la version testable.

Chaque point passe par un pré-enregistrement figé avant le run, comme pour s6 et le §6.


## 8. s6-ear : un s6 qui écoute (`docs/s6-ear-prereg.md`, `docs/SPEC-S6.md` §16)

Code complet : le front-end log-mel (Rust natif et WASM), le pré-entraînement « Mimi comme professeur », le jev audio, les questions `turn_end` et `keyword` (dont le zéro-shot), le contrôle conformal, les portes. Rien n'est entraîné. **Ne pas lancer en même temps que le run LFM3 du §6 : les deux utilisent le GPU.** La labellisation GLM du §1.1 peut continuer, elle ne fait que du réseau.

- [x] **Prérequis.** Le compilateur Metal de Xcode : sans lui, mlx ne se recompile pas, et `sky-train` ne se construit dans aucun nouveau dossier `target`. Il a été installé le 03/11/2026 avec `xcodebuild -downloadComponent MetalToolchain` (Metal Toolchain 27B47). Les scripts passent `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, parce que `xcode-select` pointe sur les Command Line Tools.
- [x] **Données** (`scripts/train-ear.sh data`, fini le 04/11/2026) : 31 Go dans `data/ear`. Les comptes sont recopiés dans `DATA_LICENSES.md` :
  - Smart Turn : 215 655 / 11 370 / 31 527 (train / val / test, le test étant `smart-turn-data-v3.2-test`) ;
  - Speech Commands : 40 538 / 4 861 / 4 890 ;
  - MSWC en + fr : 120 000 par split ;
  - LibriSpeech + Mimi : 28 539 / 2 703 ;
  - jev : `turn_end` 215 655 / 11 370 / 31 527, `keyword` 40 538 / 4 861 / 4 890, `keyword_zs` 113 238 / 109 049 / 2 000 (50 mots × 20 questions × 2 langues).
  - 50 mots réservés par langue (`held_out` de `data/ear/jev/REPORT.json`) : pas d'amendement.
  - **À dire dans le rapport :** la première archive MSWC suit l'ordre alphabétique. En anglais, 48 des 50 mots réservés commencent par « a » ; en français, ils vont de « a » à « d ». Les options d'entraînement aussi. Le zéro-shot en est plutôt plus dur : la première syllabe ne départage pas les options.
  - Fumée d'abord : `MAX_CLIPS=200 scripts/train-ear.sh data`. Le 03/11/2026, sur 200 clips par set :
    - MSWC fr : 200 / 200 / 200 (train / val / test) ;
    - LibriSpeech + Mimi : 200 / 200 (les codes Mimi ont une trame de plus que nos log-mel, la paire est coupée à la plus courte) ;
    - Speech Commands : 876 / 113 / 4 890 (le test officiel n'est pas plafonné).
  - Smart Turn v3.2 sur 25 clips de test de la fumée : exactitude 0,84, 31 ms par clip. Ces clips venaient encore du jeu d'entraînement de Smart Turn ; le test est depuis `smart-turn-data-v3.2-test`.
  - Fumée de bout en bout, le 03/11/2026 :
    - `s2-pretrain --audio`, 30 pas : la CE passe de 7,6 (le hasard sur 2 048 classes) à 7,0, à environ 100 000 trames/s une fois lancé ;
    - `s2-decide`, 40 pas, compilé, puis `s2-calibrate` et l'export ternaire : 1 615 681 paramètres, 901 Kio ;
    - `sky-eval` natif à pleine profondeur, par le chemin PCM : 0,6 ms par décision pour 1 s d'audio et 5 ms pour 10 s, log-mel compris ;
    - parité WASM sur 20 clips (`parity-node.mjs`) : écart maximal 2,4e-7, 20 argmax sur 20, 0,67 ms par décision en médiane.
- [ ] **Pré-entraînement Mimi** (60 min) : `scripts/train-ear.sh pretrain`. Rapporter la précision du codebook 0 sur les trames masquées de val (`runs/s6-ear-nano/pretrain/log.jsonl`, `val_acc_cb0`).
- [ ] **Décision et témoin** (2 × 60 min) : `scripts/train-ear.sh decide`. Le témoin sans pré-entraînement est pré-enregistré : il faut le lancer aussi.
- [ ] **Évaluation et portes** : `scripts/train-ear.sh eval`. Smart Turn v3.2 tourne sur les mêmes clips de test (`uv run scripts/ear-baseline.py`, CPU). Ensuite viennent la parité WASM sur 20 clips et `scripts/ear-gate.py gate`. Les trois portes v1 sont :
  - fin de tour ≥ Smart Turn ;
  - mots-clés ≥ 0,90 ;
  - zéro-shot en et fr ≥ 0,20.
- [ ] Publier si les portes tiennent (§5). On ajoute une démo micro sur le site : on parle, et le modèle montre en direct sa décision et sa confiance (`Engine.decideAudio`, PCM rééchantillonné à 16 kHz).
- [ ] **v2, après le §1.1** (même quota z.ai) : `sky-data teacher label-transcripts`, puis `sky-data audio-jev --about data/teacher/ear_about_labels.jsonl`, et un nouveau `decide` avec `about`. **v2 duplex** : écrire le téléchargeur AMI (la règle `audio::duplex_label` existe et elle est testée), puis la question `duplex_action` contre la règle « parler après 500 ms de silence ».

### 8.1 Le prochain run, pas à pas (s6-ear-nano)

À lancer quand le GPU est libre (pas pendant LFM3), depuis le checkout principal à jour de main, dans un terminal qui reste ouvert :

- [ ] `SKY_DUTY=0.5 SKY_COOL=45:15 RESUME=1 caffeinate -i scripts/train-ear.sh pretrain decide eval` (`data` est fait), ou une étape à la fois. Il faut compter environ 3 h de GPU à pleine vitesse, donc 8 h bridé : 60 min de pré-entraînement, 60 min de décision, 60 min pour le témoin. Le téléchargement, d'environ 30 Go, se fait au fil de l'eau. `RESUME=1` reprend aux checkpoints (toutes les 10 min) après un arrêt : relancer la même commande.
- [x] Après `data` (04/11/2026, les nombres au §8), mettre dans le rapport :
  - les comptes par set et par split ;
  - le nombre de mots réservés par langue (`data/ear/jev/REPORT.json`, `held_out`) : il en faut 50 par langue. Moins de 50 veut dire que la première archive MSWC ne suffit pas. Il faut alors un amendement daté avec `--mswc-archives 2`, **avant** `pretrain`.
- [ ] Après `pretrain`, rapporter `val_acc_cb0`. Il doit dépasser nettement 1/2 048 : sinon, c'est un échec technique, à noter, avant de lancer `decide`.
- [ ] Après `eval`, recopier dans le rapport :
  - `runs/s6-ear-nano/eval/gate.json` (les trois portes, le témoin à côté) ;
  - `crc.json` (les seuils, la couverture du Système 1) ;
  - `wasm.json` ;
  - `baseline-smart-turn.json`.

### 8.2 Tester dans le navigateur

- [x] **La démo est écrite** (04/11/2026) : la page `/ear/` (« Listen ») et le composant `EarDemo.vue`.
  - **Entrées :** le micro (`getUserMedia` et un `AudioWorklet`), ou l'un de 3 extraits CC-BY-4.0 joués par le même chemin (`site/public/ear/`).
  - **Audio :** un contexte audio à 16 kHz quand le navigateur le permet, sinon un rééchantillonnage en JS (le cas de Firefox) ; un tampon glissant de 10,24 s.
  - **Décisions :** `engine.decideAudio` toutes les 80 ms, pour `turn_end` et les mots-clés. La page affiche la décision, la confiance, la sortie utilisée, la latence, et « je passe la main » sous le seuil de renvoi (`route` dans le manifest, 0,5 sinon).
  - **Sans modèle :** si le manifest n'a aucune entrée `task: "ear"`, la page dit que s6-ear n'est pas encore entraîné. C'est le cas aujourd'hui.
- [x] **Testée dans Chrome** (04/11/2026), avec le modèle de fumée de 40 pas, en `docs:build` + `vitepress preview`. En `docs:dev`, Vite refuse d'importer le module WASM depuis `/public`, comme pour les autres pages.
  - le modèle se charge ;
  - les 3 extraits remplissent le tampon et donnent une décision toutes les 80 ms : 2,9 ms en médiane pour les deux questions, log-mel compris ;
  - changer d'extrait en pleine lecture repart bien de zéro (bug corrigé : les rappels de l'extrait précédent fuyaient dans le nouveau) ;
  - aucune erreur dans la console.

  Pour tester un modèle local, le copier dans `site/public/ear/dev/` (ignoré par git) et ouvrir `/skycmd/ear/?model=/ear/dev/<fichier>.gguf`.
- [ ] **Le micro, par l'utilisateur** : la permission micro du navigateur ne s'automatise pas. À faire dans Chrome puis Safari, avec le modèle entraîné. Vérifier :
  - que la permission est demandée une seule fois ;
  - que la décision suit la parole : parler, se taire, dire « yes » ou « stop » ;
  - que la latence reste sous 80 ms ;
  - qu'il n'y a aucune erreur dans la console.
- [ ] **À la publication du modèle :**
  - copier `models/s6-ear-nano/s6-ear-nano-crc.gguf` dans `site/public/models/`, avec une entrée `manifest.json` (`task: "ear"`, `arch: "skycmd-s2"`, et `route: {"noul": ρ, "choice": ρ}` tiré de `runs/s6-ear-nano/eval/crc.json`) ;
  - ajouter « Listen » à la barre de navigation (`site/.vitepress/config.ts`) ;
  - enregistrer un GIF de la démo pour le README et la carte Hugging Face.

### 8.3 Que tout soit carré avant de publier

- [ ] Les trois portes lues **telles quelles**, sans relance pour « voir ». Un échec est publié comme tel sur la page Results.
- [ ] La parité WASM sur 20 clips de test est ≤ 1e-4, avec le même argmax (`wasm.json`). `cargo test -p sky-web --test parity` et `wasm-pack test --node --release crates/sky-web --test parity` passent.
- [ ] `cargo test` passe sur sky-infer, sky-data, sky-web et sky-gguf, et `cargo test --release -p sky-train` passe (avec `DEVELOPER_DIR` pointé sur Xcode). Clippy et fmt sont propres, et `bash -n` et shellcheck passent sur les scripts.
- [x] Dans `DATA_LICENSES.md`, les vrais comptes remplacent « not fetched yet » (04/11/2026). Rien de `data/ear` n'est commité.
- [ ] Les chiffres sont cohérents partout, sans un seul « à mesurer » qui traîne :
  - SPEC-S6 §16 ;
  - `docs/s6-ear-prereg.md` ;
  - README ;
  - la page Bench (« Coming next » vers les résultats) ;
  - la page « What we bring » (§8) ;
  - le chapitre 26 du tutoriel ;
  - la carte `hf/cards/s6-ear-nano.md`.
- [ ] `cd site && npm run docs:build` passe (sans lien mort). Le Space Hugging Face est vérifié dans un vrai navigateur (le modèle se charge, la démo répond, la console est propre).

## 9. Le grand run : tout entraîner, au mieux, sur le M5 (décidé le 05/11/2026, à lancer une prochaine fois)

**La demande.** « Entraîne tout au mieux avec tout ce qu'il faut », puis, le 05/11 : « finalement je ferai tout sur mon pc ». Concrètement :
- **Les 5 modèles s6** (s6-ear-nano, s6-nano, s6-micro, s6-secrets-pico, s6-cast-pico), avec les optimisations que le pilote et l'ablation pico adoptent. Le choix de l'utilisateur, « ablation automatique puis tout », est la version honnête de « toutes les optimisations » : une option n'est adoptée que si les règles pré-enregistrées le disent, et **aucune décision n'est prise à la main**.
- **LFM3-1.7B** : le routeur S1/S2 (§6) et le fine-tuning (§6.1).
- **Tout sur le M5.** Il n'y a ni Mac loué ni HF Jobs, donc rien à payer. En contrepartie, la machine chauffe pendant plusieurs jours, découpés en nuits (§9.4).

**Ce que ça change par rapport au plan avec location.**
- `TIME_SCALE` reste à 1, puisque les budgets sont ceux du M5. `scripts/speed-probe.sh` ne sert pas, et `cloud-mac.sh` n'est plus à écrire.
- L'ablation tourne entière sur une seule machine (environ 15 h), sans `--summary-only`.
- `s6lr --extend`, `s6-adopt.py` et la recette passée aux scripts servent tels quels.

### 9.1 Ce qui est prêt (05/11/2026, testé)

- **`scripts/s6-adopt.py`** applique les règles 1 à 6 de `docs/s6-prereg.md` §4, à la lettre et sans jugement. Ses tests sont dans `scripts/test_s6_adopt.py` (9 tests, `python3 -I scripts/test_s6_adopt.py`).
  - `adopt` lit `runs/s6lr/summary.json` et `runs/s6ab/summary.json`, et applique :
    - Muon contre AdamW, avec un gain exigé de 0,02 si Muon coûte plus de 15 % de temps ;
    - Parallax contre son pendant softmax, avec l'optimiseur retenu ;
    - XSA ;
    - SODA, KD de sortie, EMA, StableQAT et curriculum, lus contre `parallax-muon` (gain ≥ 0,005, pas de perte arena, ECE pas plus de 0,01 pire) ;
    - sans polish : ≥ −0,005 et au moins 2 % plus rapide ;
    - la combinaison des options adoptées.
  - Il écrit `runs/recipe/recipe.json` (chaque règle avec ses chiffres, le sha256 des deux résumés et l'horodatage) et les configs des 5 modèles, avec Parallax et XSA tels qu'adoptés, dans `runs/recipe/configs/`.
  - `env MODEL` donne les variables de chaque script.
- **`cargo xtask s6lr --extend`** lance l'extension unique (×0,25 ou ×4) d'un choix au bord de la grille, et rien sinon. Le résumé du pilote compte l'extension.
- **`cargo xtask s6ab --summary-only`** refait le résumé à partir des runs rapatriés des 3 machines, sans rien relancer. Les marqueurs « fait » de xtask sont propres à chaque machine et à chaque stage.
- **Le résumé de `s6ab`** enregistre aussi les drapeaux de pré-entraînement de chaque bras, ce qui est le cas du curriculum.
- **Les scripts prennent la recette.** Sans recette, ils gardent leur comportement d'avant : vérifié à blanc.
  - `train-s6.sh` et `train-secrets.sh` gagnent `CONFIG` ; `train-s6.sh` gagne aussi `DAGGER_N`.
  - `train-ear.sh` gagne `CONFIG`, `OPT`, `PRETRAIN`, `DECIDE` et `LR_SCALE` (le taux du pilote pico, déplacé à d = 144 par μP comme pour s6-nano). Avec `--ema`, il calibre et exporte `last-ema`, pour le modèle comme pour son témoin.
  - `train-cast.sh` gagne `CAST_OPT`. Son `pilot` applique désormais seul ses deux règles : il choisit le taux, distille à ce taux, puis tranche le professeur. Il écrit `runs/s6-cast-pilot/choice.json`, que `train` lit quand `LR` et `TEACHER` ne sont pas donnés.
  - Tous gagnent `TIME_SCALE`, qui multiplie leurs budgets en minutes.
- **`scripts/speed-probe.sh`** mesure la vitesse d'une machine : 300 pas de `s2-decide` sur s6-nano. `scale` donne `TIME_SCALE`, soit la vitesse du M5 sur celle de l'hôte, pour qu'un budget en minutes donne le même nombre de pas qu'aurait fait le M5.

### 9.2 Ce qui reste à faire avant le run (par Claude, sans GPU)

- [ ] **Les amendements datés, commités avant le moindre run :**
  - [x] **`docs/s6-prereg.md` §7, amendement 4 (commité le 06/11/2026, avec `scripts/s6-recipe.sh`)** : les règles §4 sont appliquées par `scripts/s6-adopt.py`. La recette écrite dans `runs/recipe/recipe.json` remplace le commit manuel, et elle est commitée dans le rapport. L'extension au bord est lancée automatiquement par `s6lr --extend`.
  - **`docs/s6-ear-prereg.md` §8, amendement 2.** s6-ear tourne après l'ablation, avec la recette, pour le modèle et pour son témoin. Son taux est centré par μP. Le curriculum est sans effet en audio.
  - **`docs/s6-cast-prereg.md` §7, amendement 3.** s6-cast prend Muon ou AdamW, SODA, sans polish, StableQAT et XSA si l'ablation les adopte. Son pilote décide seul, par ses propres règles.
  - **`docs/s6-gen-prereg.md` et `docs/s6-secrets-prereg.md`** : une ligne chacun, pour dire que la recette passe par le fichier.
  - **`docs/lfm3-s1s2-prereg.md`, amendement 2** : il annule l'amendement 1 (à distance), et le routeur repart en local, avec MLX-4bit et mlx-vlm, comme pré-enregistré à l'origine.
  - **`docs/lfm3-ft-prereg.md`, amendement 1** : moteur local, LoRA par la méthode que le smoke local tranche, et le plafond de 1 000 questions par tâche. Les portes G1 à G3 ne changent pas.
- [ ] **`scripts/train-all.sh`, l'orchestrateur, sur une seule machine.** Chaque étape est idempotente, avec un marqueur dans `runs/all/done/`, un journal dans `runs/all/` et `DRY_RUN=1`. Les étapes, dans l'ordre :
  - `pilot` (`cargo xtask s6lr`, puis `s6lr --extend`) ;
  - `ablation` (`cargo xtask s6ab --adamw-lr-scale … --muon-lr-scale …`, avec les échelles lues dans `runs/s6lr/summary.json`) ;
  - `adopt` ;
  - les modèles, chacun avec `eval "$(scripts/s6-adopt.py env …)"` :
    - `ear` : `pretrain decide eval` ;
    - `secrets` : `train eval` ;
    - `nano` et `micro` : `lr train dagger exit eval`, après avoir vérifié `data/distill/glm.jsonl` et une couverture d'au moins 95 % ;
    - `cast` : `pilot train export eval` ;
  - `report`, qui rassemble les portes de chaque modèle dans `runs/all/REPORT.md`.
  - Un mode `SMOKE=1` avec des budgets de quelques minutes, `ROUNDS=1`, `DAGGER_N=200` et `EPISODES=2`.
  - **Le mode bridé et la reprise, par défaut.** `train-all.sh` exporte `SKY_DUTY=${SKY_DUTY:-0.5}`, `SKY_COOL=${SKY_COOL-45:15}` et `RESUME=1` (`sky-train` et les `train-*.sh` les lisent déjà). Attention : avec `RESUME=1`, un run reprend tout ce qu'il trouve dans son dossier. `SMOKE=1` doit donc écrire ailleurs que `runs/s6-*`, ou nettoyer ses dossiers, sinon les vrais budgets repartiraient d'un checkpoint de fumée.
- [ ] **Le mode local de `scripts/lfm3-ft.sh`** (`gen`, `sft` et `eval` avec mlx-vlm ou MPS, sans `hf jobs`) et son smoke local. Il doit respecter `SKY_DUTY` et `SKY_COOL` comme `experiments/lfm3-s1s2/s1s2.py`.
- [ ] **Dans chaque amendement ci-dessus :** une ligne pour dire que le run se fait en mode bridé, comme l'amendement 5 de `docs/s6-prereg.md`.
- [ ] **Les vérifications :**
  - `cargo test -p xtask` ;
  - `python3 -I scripts/test_s6_adopt.py` ;
  - `python3 experiments/lfm3-ft/test_common.py` ;
  - `bash -n` et shellcheck ;
  - `DRY_RUN=1` sur chaque étape de `train-all.sh` et de `lfm3-ft.sh`.

### 9.3 Ce que fait l'utilisateur

- [ ] Garder le Mac ouvert quand c'est possible, de préférence sur secteur : le run n'avance que Mac ouvert. Rabattre l'écran le suspend, éteindre l'arrête, et il suffit alors de relancer la même commande (voir en tête du TODO). Pas de `pmset disablesleep`.
- [ ] Lancer chaque tranche (§9.4), ou laisser Claude la lancer sur demande.

### 9.4 Le run sur le M5, en tranches (une fois le 9.2 fait)

Chaque tranche reprend là où la précédente s'est arrêtée (marqueurs de `train-all.sh`, reprise de xtask et des scripts). Les durées sont celles du M5 à pleine vitesse : **à multiplier par environ 2,7 en mode bridé (`SKY_DUTY=0.5 SKY_COOL=45:15`)**. Une tranche peut s'étaler sur plusieurs jours, et les « nuits » ci-dessous sont des numéros de tranche.

1. **La fumée `SMOKE=1`**, environ 30 min : les 5 modèles de bout en bout, en quelques minutes chacun, sortent leur GGUF et leurs portes.
2. **Nuit 1 : le pilote et l'ablation pico, environ 17 h 30, avec `caffeinate -i scripts/s6-recipe.sh`** (le mode d'emploi est en tête du TODO). Le pilote prend environ 2 h 30, plus l'extension si besoin ; l'ablation prend environ 15 h. Puis `adopt`, qui écrit la recette. Elle peut tourner avant la fumée et le reste du §9.2, qui ne servent qu'à partir de la nuit 2.
3. **Nuit 2 : s6-ear, s6-secrets puis s6-nano, environ 11 h.** s6-ear prend environ 3 h, s6-secrets environ 30 min et s6-nano environ 7 h. Les labels GLM doivent être finis et mélangés (`data/distill/glm.jsonl`).
4. **Nuit 3 : s6-micro, environ 11 h.**
5. **Nuit 4 : s6-cast, environ 9 h** (pilote, puis 8 h d'entraînement).
6. **Nuit 5 : LFM3 S1/S2 (§6), environ 9 à 10 h**, avec `caffeinate -i scripts/lfm3-s1s2.sh`.
7. **Nuits 6 et suivantes : LFM3 fine-tuné (§6.1).** La génération prend de l'ordre de 10 à 15 h, puis viennent la LoRA et l'évaluation des deux modèles.
8. **Les résultats.** `report` rassemble tout. Chaque porte est lue telle quelle, sans relance pour « voir », et un échec est publié comme tel. On met à jour le TODO, `BENCH.md` et la page Results, et on commite la recette et `REPORT.md`.
9. **La publication (§5)** se fait seulement avec l'accord de l'utilisateur, modèle par modèle.
