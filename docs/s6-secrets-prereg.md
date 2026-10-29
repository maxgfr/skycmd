# s6-secrets : pré-enregistrement (premier entraînement et porte de publication)

**Statut : figé le 26/10/2026, avant tout entraînement de s6-secrets.** Ce fichier fixe les données, le run, le choix du checkpoint, l'évaluation et la porte de publication. Les résultats seront lus à travers ce texte.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 7 ;
- il précède le run qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un score du bench.

## 1. La tâche et les données

Détecter une clé d'API ou un secret exposé dans un court texte : un fichier `.env`, du code, une config, une commande `curl`, un log, un diff. Le jev a quatre questions par texte (`docs/SPEC-S6.md` §s6-secrets) :

- **secret** (noul) : le texte expose-t-il un secret utilisable, à révoquer ?
- **service** (choice, 15 options) : quel service a émis la clé (`none` s'il n'y en a pas) ?
- **kind** (choice, 5 options) : secret, clé de test, clé publique, placeholder (ou exemple, valeur masquée, référence), ou pas un identifiant ;
- **line** (choice) : quelle ligne expose le secret ?

**Données.** `sky-data secret-jev` (graine 1234, valeurs par défaut) écrit les fichiers dans `data/secrets`. Tout est synthétique et généré en local : chaque clé suit le format public de son fournisseur avec un corps aléatoire, donc aucune n'est une vraie clé, et rien n'est commité.

- `secret_synth` : 50 000 textes, environ 193 000 questions, réparties 95/5 en train et val par texte. Ils sont tirés de 9 gabarits : env, export, python, js, yaml, json, curl, log, diff.
- `secret_bench` : 2 000 textes, 7 984 questions, en test seulement. Ils viennent de 4 gabarits que l'entraînement ne voit jamais : docker-compose, toml, go, ini.

Les deux sources ont les mêmes parts de classes :

- secrets : 50 % ;
- clés de test : 10 % ;
- clés publiques : 8 % ;
- placeholders : 17 % ;
- valeurs qui ne sont pas des identifiants : 15 %.

## 2. Le modèle

`configs/s6-secrets-pico.json` reprend l'architecture de s6-pico à l'identique (même nombre de paramètres), avec deux fenêtres élargies :

- **état : 320 jetons au lieu de 64.** Avec le tokenizer s6 (BPE 1 024 avec [NUM]), une clé aléatoire coûte environ un jeton par caractère. Les états du jev font 93 jetons en médiane, 179 au 95e centile et 312 au maximum.
- **tête : 192 jetons au lieu de 96.** La question « service » et ses 15 options font jusqu'à 179 jetons.

Ces données ne vont pas dans `data/decisions` : le s6 général les tronquerait avant la clé.

## 3. Le run (`scripts/train-secrets.sh`)

- **Prétraitement MLM** : 8 minutes sur `data/text` plus 10 % de curriculum numérique, en séquences de 256 jetons.
- **Entraînement des décisions** : 15 minutes sur `data/secrets` seul, batch 64.
  - QAT ternaire après 10 % d'échauffement en flottant.
  - Loop dropout 0,25.
  - Mêmes options d'optimiseur que s6-pico (`OPT`).
- Graine 0, une seule fois. Seul un échec technique justifie de relancer, et il est noté dans le rapport.

## 4. Checkpoint exporté

- `last` (ou `last-ema` si `DECIDE` contient `--ema`, ce qui est décidé avant le run), calibré par `s2-calibrate` sur la validation, puis exporté en ternaire.
- Aucun checkpoint intermédiaire n'est considéré.

## 5. Évaluation

`s2-eval` du GGUF ternaire sur `secret_bench`, dernière sortie, pleine profondeur. Chaque question a sa propre source (`secret_bench:secret`, `:service`, `:kind`, `:line`), donc on lit une exactitude par question.

**Référence : le scanner à règles** (`sky-data secret-baseline`), mesuré sur ce même bench avant tout entraînement. Il combine :

- des préfixes façon gitleaks pour 20 formats ;
- les chaînes de connexion qui portent leur mot de passe ;
- une règle d'entropie (au moins 3,5 bits par caractère, 20 caractères au moins, un chiffre) sur les noms qui sonnent secret ;
- une liste d'exceptions (EXAMPLE, clés de test et clés publiques).

Ses scores :

| question | exactitude | détail |
|---|---:|---|
| secret | **0,977** | précision 0,966, rappel 0,989 (999 vrais positifs, 35 faux positifs, 11 manqués, 951 vrais négatifs) |
| service | **0,787** | il ne nomme le service que pour ce qu'il signale, et appelle `generic` toute clé sans préfixe |

Les 35 faux positifs du scanner sont tous des valeurs aléatoires qui ne sont pas des identifiants, sous un nom qui sonne secret (`CACHE_KEY`, `KEY_FINGERPRINT`). Les 11 manqués sont des secrets génériques sans signe distinctif.

## 6. Objectif et porte de publication

**Objectif : battre le scanner à règles sur les deux questions qu'il traite.** Avec le GGUF ternaire à pleine profondeur, il faut :

- une **exactitude secret > 0,977**,
- **et** une **exactitude service > 0,787**.

s6-secrets-pico est publié (site, manifeste, Hugging Face) si l'objectif est atteint. Sinon, il n'est pas publié, et le rapport le dit.

Quel que soit le résultat, le rapport donne :

- l'exactitude des questions kind et line ;
- l'exactitude en validation (gabarits vus) ;
- la sortie anticipée ;
- la taille ;
- la latence.

**Ce que ce banc ne dit pas.** Il est synthétique : les formats, les gabarits et les parts de classes viennent de notre générateur. Battre le scanner ici montre que le modèle lit les formats et le contexte (nom de variable, clé publique ou de test, placeholder) dans des fichiers d'un type qu'il n'a jamais vu. Cela ne dit rien de son rappel sur de vrais dépôts : aucune conclusion de ce genre ne sera tirée.

## 7. Amendements

**Amendement 1 (28/10/2026, avant tout entraînement de s6-secrets, aucun score lu). Corpus de pré-entraînement.**

- **Changement.** Le prétraitement MLM du §3 lit `data/text-open` puis `data/text` (`--text data/text-open,data/text` dans `scripts/train-secrets.sh`), au lieu de `data/text` seul.
- **Pourquoi.** C'est le même changement que l'amendement 1 de `docs/s6-prereg.md`, dont s6-secrets reprend les options. Le texte ajouté (Q/R StackExchange, souvent techniques, PDF scientifiques, maths) est aussi plus proche des fichiers de configuration et de code du jev que les histoires.
- **Ce qui ne change pas.** La durée (8 minutes), les séquences de 256 jetons, le tokenizer, les données de décision (`data/secrets` seul), le bench et la porte de publication.

**Amendement 2 (29/10/2026, avant tout entraînement de s6-secrets, aucun score lu). Options d'optimiseur.**

- **Changement.** Le modèle final s6-pico n'existe plus (`docs/s6-prereg.md`, amendement 2). « Mêmes options d'optimiseur que s6-pico » (§3) veut dire : les options adoptées par l'ablation pico, `OPT` sans taux, plus `LR_SCALE`. `scripts/train-secrets.sh` applique `LR_SCALE` au taux par défaut de chaque phase (3e-3 en pré-entraînement, 1e-3 en décision) et refuse un `OPT` qui porterait un taux.
- **Pourquoi.** Un `--muon-lr` passé dans `OPT` aurait donné le même taux aux deux phases, ce que l'ablation ne fait pas.
- **Ce qui ne change pas.** L'architecture, les données, les budgets, le bench et la porte.
