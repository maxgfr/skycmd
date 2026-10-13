# s5 : pré-enregistrement (s5-lite, paris 1 et 2, phases 5 et 7)

**Statut : figé le 13/10/2026, avant tout passage sur un split `test`.** Ce fichier fixe les jeux d'évaluation, les rendus, le budget de runs, les règles de sélection, les métriques, les marges et les critères de succès. Les résultats seront comparés à ce texte, et à aucune version ultérieure.

**Garde-fou dans le code.** `sky-eval` refuse le split `test` d'un jeu figé (`data/s5/eval/`) dans trois cas :

- ce fichier n'est pas commité ;
- ce fichier porte des modifications locales ;
- le sha256 du jeu n'apparaît pas ci-dessous.

Chaque passage sur un jeu figé est ajouté à `runs/s5-test-log.jsonl` dans le store : date, sha256 du modèle et des jeux, commit de ce fichier, arguments.

**Modifications de ce fichier :**

- Tout amendement est commité, daté et motivé dans la section 9.
- Il est **antérieur** au passage sur le test qu'il concerne.
- Il ne touche jamais une marge ou un critère après la lecture d'un résultat de test.

## 1. Jeux figés

Les jeux sont construits par `sky-data s5-build` (graine 1234). Le manifeste est dans `data/s5/MANIFEST.json`, dans le store.

### Évaluation (`data/s5/eval/`, split `test`)

| fichier | questions | sha256 |
|---|---|---|
| `typed_decisions_test.jsonl` (bench fitted) | 2000 | `72331bda4c04ea46168447535cb1cd461adbba465167d52bea66c28ed8b9f6b3` |
| `wikitoxic_insult.jsonl` | 1000 | `6ca30c5fc05ac86b11947821be9de5fbe36cdc4a6fbb7347f5c0d54fd642a628` |
| `wikitoxic_obscene.jsonl` | 1000 | `bf0f78c9bfe1f945975f0d4850264c8a86135932e5c93e6026ef64facf6d79d4` |
| `wikitoxic_threat.jsonl` | 1000 | `fc9e285210d7a55f70c3eca4349c2b01b3ebecdbfde0468c2e6d8dee0ba3e126` |
| `wikitoxic_toxicaggregated.jsonl` | 1000 | `d6f518050a8a180be9d7e238d2d622fd8a150c4e85a779a523031b3e6c0c0851` |
| `biasframes_intent.jsonl` | 1000 | `6a7fca56b2425d1964ef09963d1d2f5d3aaef408e646e03f0a088ea07eb59518` |
| `biasframes_offensive.jsonl` | 1000 | `781b1da8c8ddc506021795075422665330136489d52097ba82982bb778713b3a` |
| `biasframes_sex.jsonl` | 1000 | `abc790c182574db9b4870fdd8daf17fd2cd16f02e786ad693e8bb046da48792f` |
| `amazonpolarity.jsonl` | 1000 | `59a6012b67cdea0a503acad85a342815ab5efff312f848d6802afb55be748a70` |
| `yahootopics.jsonl` (licence non vérifiée, signalée) | 1000 | `e208f70b80e28a6e00371ff8a44cbd0880568be2880bc37e5d47bd14af36c682` |
| `hwu64.jsonl` (s5-lite seulement, colonne « contamination via les professeurs non exclue ») | 1000 | `2246f781daead75dc44b97d9752792fd428090f94b7a24c5af4f8c863988d575` |
| `sib200_en.jsonl` (secondaire, hors liste du plan) | 204 | `9d92f595e64c7e3aba361c79ba61f558192050b8b0f30fa926a8227ab7df1953` |
| `sib200_fr.jsonl` (s5 en + fr seulement) | 204 | `d09c0c766155d37d5496580b62656ef6af2277c31b0b3e2dee77c38d3c0abc99` |
| `coherence_test.jsonl` (familles CH, S_k, N_k, D_ij des états du test) | 10200 | `bf9dff43877d2a680e3e9bbac7be550b67e1772d5ce7eab454c838014f398042` |
| `paraphrase_test.jsonl` (définitions paraphrasées à la main, deux appels) | 4000 | `211076641b132e7148fcacf166e6e3383ecb212f73860757d87c2d9fb36044ba` |
| `jevout_lite.jsonl` (500 bases × 8 distracteurs tenus hors entraînement) | 4500 | `ae6c74c6135eb301854fa70e739c026cbc18a8cc8dce312dfbc749c2be135d8d` |
| `reassign_test.jsonl` (2609.26758 : contrôle 0/1, polarités vues et non vues, aléatoire) | 5400 | `79395a73af6734ce966824c64a96b0b14b1f56691554924af6f986738f55f1e6` |
| `policy_test.jsonl` (mini-PolicyBench, 4 politiques tenues hors entraînement) | 800 | `dc28b0dc5f23805c4a062bb60ebb3416ba3220688417c83c6c2a37a0d3e52ea5` |
| `coverage_test.jsonl` (omission et menus BM25 k = 2, 5, 10 ; 30 % de calibration) | 5000 | `80c0aa3fc8adaadf317648664b32f2e0721242572b88b9edf26b9d80c4c636c1` |

### Fichiers écrits à la main, commités (`eval/s5/`)

| fichier | sha256 |
|---|---|
| `defs.json` (questions et définitions des classes de chaque jeu) | `364469ae28488bd44008f59f867c06b178e4fd4abedfc77625cdc3cccc09358d` |
| `paraphrases.json` (paraphrases d'évaluation, disjointes des gabarits d'entraînement) | `6411792c9aafdc37abcbdceaeb1d700c6071e0c24321820759f0adc4deb94d43` |
| `demo-cases.jsonl` (56 cas des démos, 30 en, 26 fr ; exclus de l'entraînement) | `c2dea1a5499a5391323ba246622600b4a052dfc3f316fe7d8484c715602fb612` |

### Entraînement (`data/s5/train/`, splits train / val / cal fixés par état)

| fichier | enregistrements (en + fr) | sha256 |
|---|---|---|
| `corpus.jsonl` | 157005 | `61adeaea89d227e1ceda2f0efb5021226eddf70c1aeb3be46b679c1e18972e14` |
| `gen.jsonl` | 25328 | `505ffef41c2fcc6269c968bcf000200ac1e789e768132dc6f8f1f2ea4df6ad72` |
| `pairs.jsonl` | 7828 | `7265bb9aa9da97b59ced59c85ebe30d04272a2fe7eaadc06d5e83a47fcbe5fbf` |
| `coherence.jsonl` | 12628 | `2bf3f00bb9c316d9d2d05350b2e076a2bce4a3c6800169074c26609ff50736e2` |

**Volumes en anglais :**

| fichier | train | val | cal |
|---|---|---|---|
| corpus | 139465 | 7604 | 7624 |
| gen | 11393 | 613 | 585 |
| paires | 7041 | 367 | 420 |
| cohérence | 11303 | 605 | 720 |

**Cibles mélangées :**

- `data/s5/mixed/` (`sky-data s5-mix`) reprend les mêmes enregistrements, avec les cibles décrites en section 3.
- Ses hashes seront ajoutés par amendement **avant le premier run d'entraînement**.

## 2. Rendus et prompts

Ils sont identiques pour A, A′, B et les baselines. Le code de référence est `crates/sky-train/src/s5/record.rs`, et `crates/sky-infer/src/bin/sky-eval.rs` pour l'évaluation.

### Enregistrement

Un enregistrement porte les champs `{src, t, ins, opts_kv: [[id, définition]...], state, target, split, uid, lang}`.

- `t` vaut `choice`, `noul` ou `score`.
- `target` est une distribution, souple ou one-hot. Le gold est son argmax.

### Rendu natif (A, A′)

| type | rendu |
|---|---|
| `choice` | `id: définition` |
| `score` | `level i: définition` |
| `noul` | `false` / `true`, ou `false: définition` |

### Rendu définition seule (B, iso)

- `choice` : la définition seule.
- `score` : `level i of K-1: définition`.
- `noul` sans définition : les définitions génériques.

| langue | false | true |
|---|---|---|
| en | « The statement is false. » | « The statement is true. » |
| fr | « L'affirmation est fausse. » | « L'affirmation est vraie. » |

Les identifiants ne sont jamais rendus.

### Séquence A / A′ (format Laya)

- Forme : `[CLS] "{t} question: {ins}" [SEP] ([MASK] " option")* [SEP] état [SEP]`.
- Lecture : une par `[MASK]`, avec une température par type.

### Séquence B (iso)

**Segments :**

- l'état voit l'état ;
- la question voit l'état et la question ;
- l'option i voit l'état, la question et elle-même.

**Positions :**

- (i) toutes les options démarrent au même décalage L (B-i) ;
- (ii) chaque option est remise à la même position logique (B-ii).

**Fenêtre locale :** calculée sur les positions logiques.

**Lecture :** moyenne sur le span de l'option, puis MLP([h, q, h⊙q, |h−q|]) et softmax.

**Budget d'état :** fixe, égal à `max_len − head_max_len − 16` tokens. Le cache d'état ne dépend donc jamais de la question.

### Longueurs

- `max_len` = 512, `head_max_len` = 192.
- Les jeux d'évaluation gardent N = min(8, K) options : le gold, plus N − 1 options tirées au hasard à la construction (graine fixe).

## 3. Données d'entraînement et professeur

### Sources exclues de s5

- `typed_decisions_bench`, rottentomatoes et toutes les sources des jeux d'évaluation : amazon_polarity, bias_frames, jigsaw / wiki_toxic, yahoo, SIB-200, HWU64.
- **MASSIVE, SLURP et MTOP, dans toutes les langues.** Leur exclusion conditionne l'admission de HWU64 pour s5-lite.
- **Toute sortie GLM** (`glm_orders`, paraphrases `para:`), à cause des CGU z.ai.

### Décontamination

Le contrôle porte sur les **états** des jeux d'évaluation et des cas de démo. Un enregistrement est retiré dans deux cas :

- couverture en 13-grammes ≥ 0,5 d'une référence ;
- Jaccard MinHash ≥ 0,8.

**Enregistrements retirés :**

| jeu | retirés |
|---|---|
| typed-decisions train | 165 |
| règles générées (format du garde-fou des démos) | 160 |
| paires | 172 |
| familles de cohérence | 432 |
| autres | 17 |

### Garde de classes

Aucune classe ne dépasse 60 % par source, avec un rééquilibrage déterministe. Les familles de cohérence en sont exemptées, pour rester entières.

### Professeur

**laya-typed-decisions** (Apache-2.0), exécuté en Rust (mlx).

- **Rendu :** rendu natif, deux ordres d'options moyennés.
- **Calibration par groupe** (source × type × K) : température et biais, sur les enregistrements train gold. En dessous de 150 enregistrements, le groupe prend la température globale.
- **Filtre d'accord :** la cible du professeur n'est prise que si son argmax égale l'argmax gold. Sinon, la cible gold est gardée.
- **Cible finale :** cible = 0,5 · gold + 0,5 · professeur, pour le corpus et le premier membre des paires. Le second membre d'une paire reprend la cible du premier.
- **Gardent leur cible gold exacte :** les enregistrements générés et les familles de cohérence.

**Écart au plan :** le professeur est unique. oscar-1-150m (téléchargé, 0,7755 sur le bench contre 0,7665) aurait coûté environ 45 minutes de GPU de plus, au débit mesuré de 50 à 140 lignes/s. Avec un seul professeur :

- il n'y a pas de choix de professeur par groupe ;
- la mesure du professeur en rendus natif et définition seule n'est pas faite.

## 4. Recette, variantes, budget et sélection (validation seulement)

### Recette commune à tous les runs (`sky-train s5-train`)

**Optimisation :**

- AdamW (sans Muon), full fine-tuning ;
- LR de pic selon la grille ci-dessous ; tête × 20 ;
- warmup sur 6 % des steps, puis décroissance linéaire ;
- weight decay 0,01 ; EMA 0,999 (poids évalués : l'EMA) ;
- batch 32, bf16.

**Perte :**

- CE souple + Brier (1) + RPS (0,25, questions `score`) ;
- KL symétrique 0,5 sur les paires (paraphrases de définitions et distracteurs), **identique pour toutes les variantes**.

**Échantillonnage des groupes :** typed 0,25 ; corpus 0,4 ; gen 0,15 ; pairs 0,1 ; coherence 0,1. Les familles de cohérence sont tirées entières.

**Augmentation :** un seul flux, commun à toutes les variantes :

- ordre des options ;
- noms aléatoires (p = 0,5) ;
- polarité `noul` (p = 0,25).

Elle n'a pas d'effet sur B, car B ne rend pas les labels.

**Données :** anglais seulement (`--langs en`).

### Variantes

| variante | rendu | noms aléatoires | polarité |
|---|---|---|---|
| A | natif | non | non |
| A′ | natif | oui | oui |
| B-i | iso, positions (i) | flux commun | flux commun |
| B-ii | iso, positions (ii) | flux commun | flux commun |

### Budget de runs (`cargo xtask s5 <stage>`)

1. **`explore`** : A′ et B-i × LR {5e-5, 1e-4, 2e-4} × initialisation {ettin-encoder-17m, ettin-reranker-17m}, à la graine 0 et 600 steps, soit 12 runs.
2. **`explore-b-ii`** : B-ii à la meilleure configuration (init, LR) de B-i, à 600 steps.
3. **`final`** : 2000 steps.
   - A′ aux graines 0, 1 et 2 ;
   - B\* aux graines 0, 1 et 2 ;
   - A à la graine 0, à la configuration retenue pour A′.
4. **`lambda`** (pari 2, sur son support) : λ ∈ {0,1 ; 0,3} à la graine 0.
5. **`lambda-final`** : λ\* aux graines 1 et 2.

### Règles de sélection

Elles sont gelées dans `runs/s5-select/*.json` à leur premier calcul, et n'utilisent que la validation.

- **Configuration d'une variante :** la NLL de validation la plus basse, sur tous les enregistrements val anglais (9189 questions, à la fin du run).
- **B\* :** B-i ou B-ii, selon la NLL de validation la plus basse à la configuration retenue.
- **λ\* :** la MAD_neg de validation la plus basse parmi les λ dont la précision de validation est ≥ celle de λ = 0 (graine 0) − 0,01. Si aucun λ ne passe, le pari 2 s'arrête, et c'est publié.
- **Support du pari 2 :** B\* si le pari 1 réussit, sinon A′. L'arm λ = 0 est formé des trois runs finalistes du support.
- **Modèle livré (s5-lite) :** dans le bras retenu, le run de plus basse NLL de validation, au format q8_0 calibré.
  - Le bras retenu est B\* si le pari 1 réussit, sinon A′.
  - Les runs λ\* le remplacent si le pari 2 réussit.

## 5. Calibration et formats évalués

Pour chaque run évalué, la chaîne est la suivante :

1. **Export f32 et q8_0**, avec la lecture et la tête en f32.
   - B : `sky-train s5-export` ;
   - A et A′ : `sky-train export` puis `sky-convert`.
2. **Calibration.** Les températures sont ajustées par type et par type × nombre d'options (n ≥ 100), en minimisant la NLL sur le split `cal` anglais de `data/s5/train` (9349 questions). Elles sont écrites dans le GGUF.
3. **Passage sur le test** de tous les jeux anglais ci-dessus et des cas de démo, avec le moteur de production (`sky-eval`, CPU).

## 6. Comparaisons principales et critères de succès

### Pari 1 : B\* contre A′ (3 graines chacun, f32 calibré, bench fitted)

**Critères principaux (les deux sont requis) :**

- **Précision :** borne haute de l'IC 95 % de (A′ − B\*) ≤ 0,02.
- **ECE** (top-label, 15 bins) : borne haute de l'IC 95 % de (B\* − A′) ≤ 0,01.

**Secondaires, avec p-values ajustées par Holm** sur ces 6 tests :

- NLL du bench ;
- Brier du bench ;
- taux de bascule sur paraphrase (deux appels) ;
- taux de bascule JevOut-lite (pire distracteur par base, choisi contre le modèle évalué) ;
- précision macro des 9 jeux zero-shot anglais (wikitoxic × 4, biasframes × 3, amazonpolarity, yahootopics) ;
- ECE macro des mêmes jeux.

**Descriptif, sans test :**

- réassignation (bascules par bras, moins le contrôle 0/1, et AUC) ;
- mini-PolicyBench ;
- couverture des candidats ;
- A, sur une graine.

**En cas d'échec :** le rendu définition seule n'est pas retenu, et l'échec est publié.

### Pari 2 : λ\* contre λ = 0 (3 graines chacun, f32 calibré)

**Pertes entraînées :** L_neg, L_part et L_xf seulement. Aucune question D, aucune perte de monotonie, de description ou de disjonction.

**Compétence D, mesurée d'abord à λ = 0 :** la borne basse de l'IC de la précision des questions D_ij contre le gold doit être > taux de la classe majoritaire de leurs golds + 0,10.

- Le « hasard » retenu ici est la classe majoritaire, plus stricte que 0,5.
- Sans compétence, les identités tenues hors entraînement sont déclarées **non mesurables** et ne comptent pas comme un succès.

**Critères principaux (tous requis, compétence comprise) :**

| critère | mesure | seuil |
|---|---|---|
| MAD_neg | moyenne sur les familles de \|s_k + n_k − 1\| pour λ\*, négations reformulées en requêtes séparées, moyenne des 3 graines | < 0,03 |
| partition | moyenne de Σ_k s_k pour λ\* | dans [0,97 ; 1,03] |
| précision du bench | non-infériorité, marge 0,02 (comme au pari 1) | — |
| monotonie | part des vérifications s_i ≤ d_ij violées de plus de 0,05 | baisse relative ≥ 30 %, et borne haute de l'IC de (λ\* − λ = 0) < 0 |

**Si seules les identités entraînées s'améliorent** (MAD_neg, partition et précision passent, mais pas la monotonie ou la compétence), le résultat est publié comme tel.

**Secondaires, avec Holm sur ces 6 tests :**

- MAD_neg ;
- \|Σ s − 1\| ;
- \|s − c\| ;
- additivité \|d_ij − (s_i + s_j)\| ;
- invariance de description \|d_ij − n_m\| (K = 3) ;
- ECE du bench.

### Phase 5 : quantification

- **Mesure :** ΔECE = ECE(q8_0 recalibré) − ECE(f32 recalibré) sur le bench, appariée, pour A′ et B\* (3 graines).
- **Critère :** borne haute de l'IC ≤ 0,01.
- Le Q4 n'est pas fait ici.

### Gates de s5-lite (modèle livré, q8_0 calibré)

| gate | seuil |
|---|---|
| bench fitted | ≥ 0,65 |
| cas de démo anglais | ≥ 90 % (27/30) |
| chaque jeu | ≥ sa baseline majoritaire (part de l'identifiant gold le plus fréquent) |

**Tout jeu au-dessus de 0,735** (plafond d'accord du professeur) est investigué avant publication.

### Phase 7 : escalade à erreur contrôlée (hors ligne)

**Seuils d'acceptation sur max p, par type, choisis par Learn-then-Test :**

- p-value binomiale de l'erreur des acceptés ;
- test en séquence fixe sur 200 seuils, du plus haut au plus bas ;
- δ = 0,05.

**Données de calibration :**

- **Principal :** le split `cal` des enregistrements typed-decisions, échangeable avec le bench (325 questions).
- **Secondaire :** tout le split `cal` anglais (plus de données, mais garantie formelle non valide sous décalage).

**Comparaison :** la règle « ensemble conforme de taille 1 » (Mondrian, s = 1 − p_y).

**Rapporté :**

- erreur des acceptés et couverture pour α ∈ {1, 2, 5, 10} % ;
- AURC ;
- couverture à ≤ 5 % d'erreur.

**Promesse :** P(erreur | accepté) ≤ α, par rapport au gold. Le pari 3 (en ligne, avec audit) n'est pas lancé ici.

## 7. Statistiques

- **Intervalles :** bootstrap percentile à 95 %, B = 2000, graine 0 (`sky-s5-report`, `sky-metrics`).
- **Plusieurs graines d'une variante :** bootstrap à deux niveaux.
  - Chaque tirage rééchantillonne les items, puis les graines avec remise.
  - Il prend la moyenne de la statistique sur les graines tirées.
  - Une comparaison appariée tire les **mêmes** items pour les deux bras, et leurs graines indépendamment.
  - La p-value est la p-value bootstrap bilatérale.
- **Unités :**
  - question pour les jeux ;
  - famille pour la cohérence ;
  - base pour les paraphrases et JevOut-lite.
- **« Non significatif »** n'est jamais un critère de succès. Seule la non-infériorité à marge fixée l'est.

## 8. Écarts au plan, déclarés avant tout résultat

- **Professeur unique** (section 3).
- **Grille d'exploration réduite :**
  - A et B-ii ne parcourent pas la grille : A prend la configuration de A′, et B-ii celle de B-i ;
  - steps fixés par le budget de calcul (600 en exploration, 2000 pour les finalistes).
- **Non faits :**
  - A′ + perm-KL ;
  - baselines B1 (log-prob des labels par la tête MLM), B2 (cascade) et B3 (tête linéaire). Seules la classe majoritaire et l'uniforme sont rapportées.
- **Comparaison à Jev (MAD_neg 0,064) :** non faite. Elle demande la batterie d'origine de 2609.33209 (ChaosNLI, PubMedQA), non téléchargée. Nos négations reformulées forment un autre jeu, rapporté séparément.
- **Repoussés :**
  - le Q4 ;
  - le pari 3 en ligne ;
  - la phase 6 (s5 en + fr, mmBERT-small).
- **Repli du français :** les démos françaises de /live affichent « français indisponible : s5-fr en échec, voir résultats » et basculent sur les presets anglais.
- **Licences :**
  - MTOP est exclu (licence inconnue) ;
  - la licence de yahootopics n'est pas vérifiée (signalé) ;
  - HWU64 est rapporté pour s5-lite seulement, en colonne « contamination via les professeurs non exclue », car les sources de Laya ne sont pas toutes déclarées.

## 9. Amendements

(aucun à ce jour)
