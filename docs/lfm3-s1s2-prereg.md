# LFM3-1.7B : Système 1 d'abord, Système 2 seulement si besoin (pré-enregistrement)

**Statut : figé le 03/11/2026, avant tout run.** Seul un essai de fumée a tourné : 3 questions de calibration par mode, dans `runs/lfm3-s1s2-smoke`, jamais mélangées au run. Ce fichier fixe les données, les modes, le routeur, les métriques et les critères de succès. Les résultats seront lus à travers ce texte.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 8 ;
- il précède les runs qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un résultat de test.

## 1. Pourquoi

LFM3-1.7B (Liquid AI, 27/10/2026) réfléchit par défaut. Sur un modèle de cette taille, une longue réflexion peut le faire renoncer à une bonne première réponse.

- **Une mesure communautaire** (30 prompts, budgets 0 / 64 / 128 / 256 / illimité) trouve l'optimum entre 0 et 64 jetons de réflexion. Avec un budget illimité, 9 réponses sur 30 n'arrivent jamais au bout.
- **La documentation officielle** dit que couper la réflexion marche de façon inconsistante sur les tâches de raisonnement.
- **L'essai de fumée** montre l'autre face. Sans réflexion, une question de GSM8K est partie en boucle jusqu'à la limite de 1 024 jetons ; avec 256 jetons de réflexion, la réponse était juste.

Le meilleur mode dépend donc de la question. Ce projet ne touche pas aux poids : il choisit, question par question, s'il faut réfléchir. C'est le principe des sorties anticipées calibrées de skycmd. L'approche par édition des poids (Weight-Space Programming, issue #75, fermée) traitait un bug de boucles. Elle n'ajoutait pas de capacité, d'après ses propres auteurs.

## 2. Modèle et environnement

- **Modèle** : `LiquidAI/LFM3-1.7B-MLX-4bit`, révision `deb2986b9660c15ff7dd4c2bb1211ce3bbb814d4` (0,97 Go, quantification affine 4 bits). Sous LFM Open License v1.0 ; les poids ne sont pas redistribués.
- **Inférence locale** : mlx-vlm 0.7.11 (aucune version de mlx-lm ne charge LFM3), Python 3.12 via `uv`, `scripts/lfm3-s1s2.sh`.
- **Aucun entraînement de LFM3.** Seule la sonde de confiance est ajustée : une régression logistique à 6 coefficients.

## 3. Données (`scripts/lfm3-s1s2.sh data`)

Révisions figées, tirage au hasard avec la graine 1234 :

| split | tâche | source | questions |
|---|---|---|---:|
| calibration | GSM8K | `openai/gsm8k` (`main`, train), rév. `740312ad`, MIT | 300 |
| calibration | MATH | `DigitalLearningGmbH/MATH-lighteval` (train), rév. `0530c786`, MIT | 300 |
| test | GSM8K | `openai/gsm8k` (`main`, test), rév. `740312ad` | 500 |
| test | MATH-500 | `HuggingFaceH4/MATH-500` (test), rév. `6e4ed1a2`, tiré du test de MATH (MIT) | 500 |

- Les questions MATH de calibration viennent du train de MATH. MATH-500 vient de son test : aucun recouvrement.
- **Le split test n'est lu qu'une fois**, par `route --split test`. Les générations du test peuvent tourner avant, sans que rien n'en soit lu.

## 4. Les modes (`gen`)

Le même prompt pour tous les modes : la question, puis « Solve the problem. Put the final answer in \boxed{}. ».

- **s1 (Système 1)** : réflexion coupée, c'est-à-dire un bloc `<think></think>` vide dans le gabarit de LFM3. Décodage glouton, au plus 1 024 jetons. On garde la log-probabilité de chaque jeton généré.
- **s2-B (Système 2)** pour B = 256 et B = 1 024 :
  - réflexion active, avec l'échantillonnage recommandé par Liquid AI pour les maths (température 1,0, top_p 0,95, top_k 20, pas de pénalité de répétition) ;
  - si B jetons passent sans `</think>`, la réflexion est fermée et la réponse est demandée (au plus 1 024 jetons) ;
  - une graine par question, dérivée de son id.
- **La réponse** est le dernier `\boxed{}`. À défaut, pour GSM8K, c'est le dernier nombre.
- **La notation** est une égalité exacte après normalisation (espaces, `\left`/`\right`, `dfrac`, degrés, pourcents, `x=`), ou une égalité numérique. C'est la même pour tous les modes.

## 5. Le routeur (`route`)

1. **Le score de confiance**, calculé sur la seule sortie de s1 : une régression logistique sur 6 traits.
   - les log-probabilités moyennes des jetons de la réponse et de toute la génération ;
   - la plus basse log-probabilité ;
   - le logarithme du nombre de jetons ;
   - la présence d'une réponse ;
   - l'arrêt par limite de jetons, c'est-à-dire une boucle.

   Elle est ajustée sur la calibration pour prédire si s1 est juste.
2. **La règle** : si le score est au moins τ, on garde la réponse de s1. Sinon, on prend celle de s2. Si s2 n'a donné aucune réponse, on garde celle de s1. Le coût d'une question routée vers s2 compte les jetons de s1 **et** de s2.
3. **Le seuil, par contrôle conformal du risque** sur la calibration (n = 600) :
   - la perte d'une question vaut 1 si le routeur se trompe alors que s2 seul a raison ;
   - τ est le plus petit seuil tel que (n · R̂(τ) + 1) / (n + 1) ≤ **α = 0,02** ;
   - la perte ne peut que baisser quand τ monte, donc la garantie tient : en espérance, au plus 2 points de précision perdus par rapport à s2 seul.
4. **Le budget retenu, B\***, est celui de s2 dont la précision de calibration est la meilleure. Il est choisi avant la lecture du test, et tout est rapporté pour les deux budgets.

## 6. Critères (test, budget B\*)

- **H1 (diagnostic)** : la précision de s1 et de s2-B\* par tâche. Elle dit si réfléchir aide LFM3-1.7B en maths. Pas de seuil.
- **H2 (précision gardée)** : précision du routeur ≥ précision de s2-B\* − 0,02.
- **H3 (calcul divisé par deux)** : jetons moyens du routeur ≤ 0,5 × ceux de s2-B\*.
- **H4 (mieux que les deux modes)** : précision du routeur ≥ max(s1, s2-B\*) + 0,01.

**Verdict :**
- **succès** si H2 et H3 tiennent ;
- **« LFM3 amélioré »** seulement si H4 tient aussi.

Le rapport donne tout, quel que soit le résultat :
- les tableaux par tâche et par budget ;
- l'oracle, c'est-à-dire la meilleure des deux réponses : le plafond d'un routeur parfait ;
- la part de questions routées vers s1, la part de réflexions forcées, et τ.

## 7. Interdits

- Lire le test avant que le routeur soit figé sur la calibration.
- Changer α, les budgets, les traits ou la règle après avoir vu un résultat de test.
- Relancer une génération pour « voir ». Une génération interrompue reprend où elle s'était arrêtée, et un échec technique est noté.

## 8. Amendements

**Amendement 1 (05/11/2026, avant tout run : seul l'essai de fumée local de 3 questions par mode a tourné, aucun résultat de calibration ni de test n'existe). Le run passe à distance, en bf16.**

- **Changement.** Le run ne se fait plus en local. C'est la décision de l'utilisateur : le M5 chauffe trop pendant les 10 h d'inférence. Le même protocole tourne à distance, sur Hugging Face Jobs, par `scripts/lfm3-ft.sh eval` (`experiments/lfm3-ft/eval.py`) :
  - **modèle** : `LiquidAI/LFM3-1.7B` en bf16, au lieu de `LiquidAI/LFM3-1.7B-MLX-4bit` (rév. `deb2986b`). La révision est épinglée par le smoke de `docs/lfm3-ft-prereg.md` ;
  - **moteur** : vLLM au lieu de mlx-vlm. Les log-probabilités de s1 sont celles de vLLM (`logprobs=1`, le jeton tiré). Si le smoke montre que vLLM ne charge pas LFM3 correctement, c'est le repli `generate` de transformers, avec la règle de décision et les log-probabilités définies au §2 de `docs/lfm3-ft-prereg.md` ;
  - **prompt** : le texte du gabarit de chat, tokenisé sans ajouter de jeton spécial (le gabarit commence déjà par BOS).
- **Ce qui ne change pas.**
  - Le prompt, les modes s1, s2-256 et s2-1024 : budget, forçage, échantillonnage, une graine par question dérivée de son id.
  - L'extraction de la réponse, la notation, les 6 traits.
  - Le routeur, α = 0,02, le choix de B\*, les critères H1 à H4 et les interdits.
  - Le code de `route` est celui de `s1s2.py`, inchangé : `experiments/lfm3-ft/route.py` le pointe sur `runs/lfm3-ft/eval/<base|ft>` au lieu de `runs/lfm3-s1s2`.
  - Les fonctions copiées dans `experiments/lfm3-ft/common.py` sont vérifiées identiques à celles de `s1s2.py` par `test_common.py`.
- **Appliqué à deux modèles.**
  - Le modèle de base.
  - Le modèle fine-tuné de `docs/lfm3-ft-prereg.md`, avec exactement le même protocole. Sa sonde et son τ sont ajustés sur sa propre calibration, et son test est lu une fois.
- **Où tourne `route`.** En local, sur CPU, sur les générations rapatriées (`scripts/lfm3-ft.sh pull route`) : `route --split calib`, puis `route --split test`, une fois par modèle.
- **Pourquoi.** La sonde lit les log-probabilités des jetons : il faut un moteur qui les rende. Aucune API ne sert LFM3 avec ses log-probabilités au 05/11/2026 (`TODO.md` §6). bf16 et 4 bits ne donnent pas exactement les mêmes sorties : les résultats de ce run sont ceux de LFM3-1.7B en bf16, et le rapport le dit.
