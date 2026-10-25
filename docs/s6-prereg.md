# s6 : pré-enregistrement (pilote de learning rate, ablation, modèle final)

**Statut : figé le 25/10/2026, avant le pilote, l'ablation et tout entraînement s6.** Ce fichier fixe les données, les runs, les métriques et les règles d'adoption. Les résultats seront lus à travers ce texte, et pas à travers une version écrite après coup.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 7 ;
- il précède les runs qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un résultat qu'il départage.

## 1. Données

- Toutes les données sortent de `scripts/prepare-data.sh`, lancé une fois avant le pilote. Aucune régénération entre le pilote, l'ablation et le modèle final : tous les bras voient les mêmes fichiers.
- Les données enseignant GLM (`glm_orders`, 3 000 ordres de drone ; `paraphrased`, 40 000 consignes reformulées) ne sont présentes que si `SKY_TEACHER_URL` et `SKY_TEACHER_KEY` sont définis au moment de la préparation. Leur présence ou leur absence est notée dans le rapport final.
- Le split `val` sert à choisir. Aucun split `test` n'est lu pendant le pilote ou l'ablation.

## 2. Pilote de learning rate (`cargo xtask s6lr`)

- **Pourquoi.** Muon v2 aligne la RMS de ses mises à jour sur celle d'AdamW (0,2·√max(A, B)), si bien que le taux d'AdamW devrait convenir. Mais l'optimum de Muon se déplace avec la largeur (arXiv 2512.22768), et un baseline AdamW non réglé gonfle le gain de l'autre optimiseur (Wen et al., arXiv 2509.02046). On règle donc **les deux**.
- **Runs.** `softmax-adamw` (`--adam-lr`) et `parallax-muon` (`--muon-lr` seulement ; ses paramètres AdamW gardent le taux par défaut) à ×0,5, ×1 et ×2 du taux par défaut (3e-3 en pré-entraînement, 1e-3 en décision). Graine 7, mêmes budgets en pas que l'ablation (4 800 + 6 400). Six runs, environ 2 h 30 sur un M5.
- **Choix.** Pour chaque optimiseur, l'échelle de meilleure précision ternaire finale (dernière sortie, split `val`). On garde ×1 sauf si une autre échelle le bat d'au moins **0,01** : une seule graine ne départage pas moins. Les points intermédiaires ne comptent pas, car le classement peut s'inverser pendant la décroissance du taux (arXiv 2509.02046).
- **Bord de grille.** Si le choix tombe sur ×0,5 ou ×2, une seule extension (×0,25 ou ×4) est permise avant l'ablation, par amendement.
- **Sortie.** `runs/s6lr/summary.json`, puis `cargo xtask s6ab --adamw-lr-scale a --muon-lr-scale m`.

## 3. Ablation (`cargo xtask s6ab`)

Les onze bras de `docs/SPEC-S6.md` §4, 3 graines (7, 11, 13), avec les échelles du pilote. `runs/s6ab/summary.json` donne, par bras, la moyenne et l'écart-type sur les graines.

**Métrique principale.** La précision ternaire finale de la dernière sortie (`val.accuracy`).

**Métriques secondaires.**

- ECE ;
- marge de logits ;
- rang effectif ;
- score arena par jeu (4 jeux, 30 épisodes, τ = 1) ;
- `train_minutes`, le temps réel des deux phases.

Le temps réel compte : les vrais entraînements sont budgétés en minutes, et à temps égal AdamW peut battre Muon sur le langage à petite échelle (arXiv 2610.21555).

**Gain significatif.** Un écart de précision moyenne d'au moins **0,01**, et plus grand que l'écart-type de chacun des deux bras comparés.

**Perte arena.** Un jeu dont le score moyen baisse de plus de 5 % par rapport à la référence.

## 4. Règles d'adoption

1. **Muon v2.** `softmax-muon` contre `softmax-adamw` : gain significatif, et pas de perte arena.
   - Si le surcoût en `train_minutes` dépasse 15 %, le gain exigé passe à 0,02.
   - Sinon on garde AdamW, et tous les bras Muon suivants sont lus contre leur pendant AdamW quand il existe.
2. **Parallax.** `parallax-muon` contre `softmax-muon` (ou les deux bras AdamW si la règle 1 a retenu AdamW) : gain significatif, et pas de perte arena. C'est la règle de `SPEC-S6.md` §4, chiffrée.
3. **XSA.** `xsa-muon` contre `softmax-muon` : gain significatif, et pas de perte arena. Si Parallax et XSA sont adoptés tous les deux, le modèle final prend `configs/s6-pico-xsa.json`.
4. **Changements isolés.** SODA, KD de sortie, EMA, StableQAT et curriculum sont lus contre `parallax-muon` :
   - adoptés si le gain moyen est d'au moins 0,005 ;
   - sans perte arena ;
   - avec un ECE qui ne se dégrade pas de plus de 0,01.
5. **Sans polish.** `parallax-muon-nopolish` est adopté seulement s'il est non inférieur (≥ −0,005) **et** au moins 2 % plus rapide en `train_minutes`.
6. **Combinaison.** Les changements adoptés sont combinés sans ré-ablation, par les variables de `scripts/train-s6.sh` (`CONFIG`, `OPT`, `PRETRAIN`, `DECIDE`). Le risque d'interaction est accepté et vérifié par la porte finale.

## 5. Modèle final et porte (`scripts/train-s6.sh`)

- **Entraînement.**
  - Pré-entraînement 8 min et décision 15 min (budget en temps, comme s3-pico).
  - Trois rounds DAgger de 8 min ; la part de l'oracle vaut 0,2, puis 0,1, puis 0,05.
  - Seuils de sortie par contrôle conformal du risque (α = 0,02).
- **Porte des seuils.** Le jeu aux seuils conformaux reste à moins de 2 % de la pleine profondeur, sans crash ni veto de plus. Sinon, la pleine profondeur est gardée, ou `scripts/calibrate-exit.sh` est utilisé.
- **Porte de publication.** `s6-pico-g3` contre le `s3-pico-g3` publié, sur l'arena (4 jeux, 30 épisodes, τ = 1).
  - **Condition 1.** Aucun jeu en perte arena.
  - **Condition 2.** Au moins un jeu meilleur de 5 %, ou la même qualité avec moins de couches moyennes aux seuils conformaux.
  - **Si la porte échoue.** s6 n'est pas publié, et le rapport le dit.

## 6. Interdits

- Relancer une graine ou un bras pour « voir » ; seul un échec technique (crash, run incomplet) justifie un nouveau run, noté dans le rapport.
- Choisir sur un checkpoint intermédiaire.
- Modifier les données ou les budgets entre le pilote et l'ablation.

## 7. Amendements

Aucun.
