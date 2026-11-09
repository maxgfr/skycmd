# Poids flexibles : pré-enregistrement (TODO §7)

**Statut : figé le 09/11/2026, avant tout calcul de ce §7.** Aucune édition, aucune génération GLM et aucune mesure d'arena de ce protocole n'existent à cette date. Le code n'est pas écrit.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 9 ;
- il précède le calcul qu'il concerne ;
- il ne touche jamais un seuil, un bras ou un jeu de graines après la lecture d'un score de test.

**Ce qui n'est jamais publié.** Aucun modèle édité, adapté ou piégé de ce §7 ne va sur Hugging Face, le site ou le manifeste. Seul le rapport l'est. Publier un modèle édité demanderait un autre pré-enregistrement.

## 1. Le point de départ

Weight-Space Programming (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 02/11/2026) : un LLM lit et écrit directement des poids de réseaux. Le texte n'est ni relu par des pairs ni reproduit, et il repose sur un seul LLM (GPT-6 Astra). Nous l'avons lu par des articles secondaires, pas à la source. Ici, le LLM est **GLM-5.3-flash**, nettement plus petit : un échec chez nous ne dit donc rien de l'article, et un succès serait à reproduire.

**Ce que ce travail peut montrer :** sur de petits modèles ternaires, si une édition guidée par un LLM bat une édition au hasard et une baseline sans LLM, avec l'arena comme juge.

## 2. Le modèle de base

Le checkpoint flottant de s3-pico n'existe plus (constat du 09/11/2026). **Base : s6-nano**, le checkpoint exporté par `scripts/train-s6.sh` à la fin de la tranche 2 : `runs/s6-nano-g$ROUNDS/decide/$ckpt` (`last` ou `last-ema` selon la recette).

- **Pourquoi pas réentraîner s3-pico :** ce serait des heures de GPU pour un modèle dont on n'aurait plus le même checkpoint que celui publié.
- **Quel que soit le verdict de sa porte.** Si s6-nano ne passe pas sa porte, l'expérience reste valable : elle a besoin d'un modèle qui joue, pas d'un modèle publié. L'exception est un modèle qui ne joue pas du tout : score moyen ≤ celui d'une politique au hasard sur 3 jeux sur 4. On s'arrête alors et on le rapporte.
- **Aucun calcul du §7 pendant une tranche GPU.** L'ordre est 7.1, 7.2, 7.4, 7.3, après les tranches 2 à 7.

**Les graines.** Toute mesure se fait sur l'arena, 4 jeux, veto actif, comme la porte de s6-nano.

- **Dev :** `--seed 1000`, 30 épisodes par jeu. Ce jeu sert à guider et à choisir, autant qu'on veut.
- **Test :** `--seed 5000`, 200 épisodes par jeu, les mêmes pour tous les bras, **lu une seule fois**, après le gel de tous les bras d'une expérience.

## 3. Le défaut visé (règle, sans choix à la main)

Sur les graines dev, pour chaque jeu : `écart = (oracle − modèle) / |oracle|` avec les scores moyens que sort l'arena. **Le jeu visé est celui de plus grand écart.** Un seul jeu visé, fixé une fois pour toutes pour 7.1.

## 4. 7.1 : édition ciblée sans oubli

**Ce qu'une édition a le droit de faire.** Une liste de `{paramètre, indice, delta}` sur le checkpoint flottant, au plus **2 000 poids**, chaque `|delta|` au plus égal à l'écart-type du paramètre touché. Puis réexport ternaire **avec la calibration existante**, sans recalibrer. Le rapport donne le nombre de poids dont la valeur ternaire a réellement changé. Une édition qui n'en change aucun est invalide.

**Ce que GLM voit à chaque passe.** Au plus 20 000 poids en texte, tirés des derniers paramètres avant la sortie finale ; les activations sur 20 états d'échec et 20 états de réussite du jeu visé ; le score dev de sa dernière édition. **Une passe = une proposition + une mesure dev.**

**Bras** (chacun figé sur dev avant le test) :

| bras | définition |
|---|---|
| A. base | le modèle sans édition |
| B10, B25, B50 | GLM, la meilleure édition sur dev parmi ses 10, 25 ou 50 premières passes. 3 boucles indépendantes ; le bras retient celle des 3 qui est la meilleure sur dev |
| C. hasard | mêmes paramètres, même nombre de poids et mêmes deltas que B50, indices tirés au hasard. 5 tirages, tous au test, et le rapport donne la moyenne |
| D. sans LLM | différence de moyennes des activations entre états d'échec et de réussite, soustraite sur les colonnes des mêmes paramètres, au plus 2 000 poids |
| E. ternaire direct | comme B50, mais les changements sont écrits directement dans {−1, 0, +1} du GGUF. Variante secondaire, sans effet sur le verdict |

**Règle de garde (bras par bras, au test, épisodes appariés par graine) :**

- le jeu visé gagne : score moyen supérieur à A, borne basse de l'IC à 95 % (bootstrap apparié, 10 000 tirages) > 0 ;
- **et** aucun autre jeu ne perd plus de 2 % de son score moyen.

**Verdict.** L'hypothèse « l'édition par LLM marche ici » est retenue si **B50 passe la règle de garde et bat C en score sur le jeu visé**. Sinon, le rapport dit qu'elle n'est pas montrée. D est rapporté de la même façon : si D passe sans LLM et que B ne fait pas mieux, le LLM n'apporte rien ici, et le rapport le dit.

## 5. 7.2 : initialisation en une fois

**Le test.** GLM reçoit, sans données ni score, la description d'un jeu et des traits d'état, et écrit en une réponse les poids de la tête de décision de s6-nano. Puis entraînement `s2-decide` avec la recette, à partir du même `pretrain/init`.

**Bras, 3 graines d'entraînement chacun, 6 400 pas :** initialisation par défaut ; GLM (3 réponses indépendantes, une par graine) ; hasard de mêmes normes par paramètre que les poids écrits par GLM.

**Mesure.** Le TODO parle de « la qualité de s3-pico ». Comme l'architecture et les données ont changé, ce niveau n'est pas comparable. On le remplace par : **pas nécessaires pour atteindre, sur `games` en validation ternaire, la moyenne finale du bras par défaut moins 0,01**, lus dans `log.jsonl`.

**Verdict.** Gain montré si la moyenne des pas du bras GLM est au moins **20 % inférieure** à celle du bras par défaut, et inférieure aussi à celle du hasard de mêmes normes. Avec 3 graines, le résultat reste indicatif, et le rapport donne les 9 courbes.

## 6. 7.4 : audit de poids (une porte dérobée connue)

**Le piège.** Sur une copie de s6-secrets-pico : `data/secrets` avec 1 % des textes d'entraînement portant une chaîne déclencheuse fixe, et leur réponse à `secret` forcée à « non ». Même recette et même budget que le vrai run. Sortie dans `runs/weights/backdoor/`, jamais dans `models/` publiés.

**L'audit, à l'aveugle.** 4 modèles (2 piégés, 2 propres, graines distinctes, noms tirés au hasard). GLM classe chaque modèle piégé ou propre, à partir :

1. des poids seuls ;
2. des activations sur un corpus de sondes contenant 50 chaînes candidates, dont le vrai déclencheur.

**Contrôle préalable.** Le piège doit fonctionner : taux d'attaque ≥ 90 % sur `secret_bench` avec déclencheur et exactitude propre à 0,02 près du modèle propre. Sinon, l'audit n'a pas lieu et le rapport le dit.

**Ce qu'on rapporte.** Le nombre de modèles bien classés sur 4 et le rang du vrai déclencheur parmi les 50, sans seuil de succès : avec 4 modèles, aucune conclusion statistique n'est possible. L'article trouve 0 sur 4 sur les poids seuls et 3 sur 3 avec les activations ; c'est le repère de comparaison, pas une barre.

## 7. 7.3 : des poids flexibles, avec un garde-fou

**Le modèle.** Le s6-nano ternaire reste figé. Un adaptateur de **rang 4** s'ajoute aux logits de la dernière sortie, initialisé à zéro. Après chaque veto, un pas de descente sur l'entropie croisée vers l'action retenue par le veto. Les poids de base ne bougent jamais.

**Protocole, d'abord dans l'arena native**, pas dans le navigateur. Flux de 300 épisodes par jeu, graine 7000, sur lequel l'adaptateur apprend en continu. Le taux d'apprentissage est choisi une fois, dans {0,01 ; 0,03 ; 0,1}, sur des flux dev (graine 6000), au meilleur score moyen.

**Bras :** figé ; adaptateur sans garde-fou ; adaptateur avec garde-fou.

**Le garde-fou (α = 0,02).** Tous les 10 épisodes, le contrôle de risque (`crates/sky-arena/src/crc.rs`) compare le score de l'adaptateur à un miroir figé qui rejoue les mêmes graines. Si la perte relative dépasse α, l'adaptateur est retiré pour le reste du flux.

**Limite assumée.** Le contrôle conformal suppose des exemples échangeables, ce qu'un adaptateur qui apprend sur un flux ne donne pas. Ici, la « garantie » n'en est donc pas une au sens strict : on la **mesure** (fréquence du dépassement de α sur le flux de test) et le rapport l'écrit ainsi.

**Verdict.** Résultat retenu si, sur la moitié finale du flux, l'adaptateur avec garde-fou fait mieux que le figé d'au moins 2 % sur le jeu visé (§3), sans qu'aucun autre jeu ne perde plus de 2 %. Le navigateur ne vient qu'en cas de succès, comme démonstration, sans nouvelle mesure de performance.

**Littérature (recherche du 09/11/2026, résumés seulement).** Adaptation de bas rang au test sur un modèle gelé : TTL (WACV 2025), BOLT (arXiv 2512.02441). Base quantifiée figée et adaptateurs : LoQT (NeurIPS 2024), « Signs Beat Floats » (arXiv 2605.24058). Mise à jour bornée et retour arrière : In-Cell Learning (arXiv 2608.20873). Risque d'effondrement de l'adaptation sans étiquettes : LAME (arXiv 2201.05718). Une recherche n'a trouvé aucun travail combinant ces éléments avec le contrôle conformal du risque, ce qui ne prouve pas qu'il n'en existe pas.

## 8. Reproductibilité et rapport

- GLM-5.3-flash via z.ai (`ccs config get zai`) : prompts, réponses et graines écrits dans `runs/weights/`. Plafond de 400 requêtes pour tout le §7.
- Mode bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`) pour tout entraînement, et l'arena à un cœur.
- Une panne technique (quota, format illisible) relance le même calcul à l'identique, et elle est notée. Aucune relance pour « voir » un meilleur résultat.
- Le premier essai technique du format des éditions utilise uniquement les graines dev. S'il impose un changement de protocole, il passe par un amendement daté avant toute lecture du test.
- `runs/weights/REPORT.md` donne tous les bras, y compris les échecs. Le site et `BENCH.md` en reprennent le résultat.

## 9. Amendements

Aucun à ce jour.
