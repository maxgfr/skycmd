# Modèles de décision ultra-petits et rapides : état de l'art et plan pour skycmd (11 oct. 2026)

## Abstract / TL;DR
Quatre résultats récents se combinent bien pour un modèle de décision sous le million de paramètres [S60][S14][S43][S2] :
- **la profondeur par récursion d'un bloc minuscule** : TRM, un seul réseau à 2 couches, ~7M paramètres [S60] ; Ouro, un bloc partagé en boucle avec une profondeur apprise [S61] ;
- **la sortie anticipée vue comme une « lecture » de la trajectoire**, séparée de la façon dont la trajectoire est entraînée [S14], avec un seuil calibré sous contrôle de risque [S30] ;
- **des nombres encodés en un seul token** : FoNE [S43], FT-Transformer [S62] ;
- **la quantification ternaire par QAT, qui tient même aux petites tailles** [S2][S1].

L'apprentissage par imitation avec DAgger, c'est-à-dire un enseignant qui étiquette les états visités par l'élève, reste la base pour contrôler un jeu en boucle fermée [S54][S53][S47].

Le plan qui en découle pour skycmd, « s3 », combine ces quatre briques avec le flux statique mis en cache de s2. [M]

## Background & motivation
Les modèles de décision typés sont déjà nombreux sur Hugging Face [S42][S52][S56].
- **Une seule passe** : decider-4b lit un état et des questions typées, puis rend une distribution sur les options en une passe, sans décodage [S42].
- **Les trois types de questions** : Sev distingue choice, score ordinal et noul [S52].
- **La calibration** : fragment-1 calibre une température par type de question [S56].

Ces modèles font de 100M à plusieurs milliards de paramètres. Notre objectif est inverse : un modèle de quelques centaines de milliers de paramètres qui décide en quelques microsecondes dans le navigateur. [M]

## Key papers (chronological)
- **FT-Transformer** : chaque colonne, numérique ou catégorielle, devient un embedding (un token), puis un Transformer et un token de classe font la prédiction [S62].
- **Early exit sous contrôle de risque** : le seuil de sortie est calculé sur des données de calibration pour respecter un niveau de risque toléré ; une correction assure le contrôle en espérance [S30].
- **BitNet b1.58 aux petites échelles** : des modèles de 100K à 48M paramètres ont été étudiés. Ajouter de la capacité compense la précision réduite [S2].
- **FoNE** : un nombre devient un seul token via des caractéristiques de Fourier [S43]. Un travail voisin souligne que xVal doit ramener les nombres dans une plage bornée [S36].
- **TRM** : un unique réseau à 2 couches, appliqué récursivement, avec ~7M paramètres, obtient 45 % sur ARC-AGI-1 [S60].
- **Ouro (LoopLM)** : le calcul se fait en itérant dans l'espace latent. Un objectif régularisé par l'entropie apprend la profondeur, et les modèles 1,4B et 2,6B égalent des modèles allant jusqu'à 12B [S61].
- **LoopFormer** : un modèle en boucle conditionné par un budget de calcul. Son entraînement à profondeur variable coûte environ 1,5× les FLOPs d'un modèle à boucle fixe [S10].
- **Adaptive Depth in Looped Transformers** : sépare la formation de la trajectoire (l'objectif d'entraînement) de la lecture qui choisit la sortie, et compare des priors fixes à des portes apprises [S14].
- **Stochastic stopping** : tirer la profondeur au hasard autour d'une valeur, plutôt qu'une profondeur fixe, stabilise la généralisation hors distribution des boucles [S49].
- **Universal Transformers Need Memory** : la récursion à poids partagés, l'arrêt adaptatif et un état persistant à travers les itérations sont trois lignes de recherche qui convergent de nouveau en 2025–2026 [S8].

## Methods & approaches compared
| Levier | Ce que disent les sources | Application skycmd |
|---|---|---|
| Profondeur sans paramètres | un bloc minuscule récursif (TRM) [S60], une boucle partagée (Ouro) [S61] | `state_loops` : 1 bloc d'état bouclé 4 fois [M] |
| Sortie anticipée | la lecture (readout) se sépare de la trajectoire [S14] ; seuil calibré sous risque [S30] | calibrer τ sur les états visités en jouant [M] |
| Nombres | un token par nombre (FoNE) [S43] ; un token par feature (FT-Transformer) [S62] | `fuse_numbers` : « clé valeur » devient un seul token [M] |
| Ternaire | le QAT 1,58 bit tient de 100K à 48M paramètres [S2] ; un modèle QAT de 2,1M paramètres atteint une perplexité de 10,1 contre 151,9 en quantification post-entraînement [S1] | QAT pendant toute la phase de décision [M] |
| Imitation | la distillation on-policy : l'enseignant évalue les états visités par l'élève [S54] ; même boucle en conduite autonome [S47] ; BC puis DAgger sur Flappy Bird dans le cours de RL de Stanford [S53] | DAgger avec l'oracle comme enseignant |
| Données | decider-4b : dix familles programmatiques à vérité vérifiable dans son mélange [S42] | curriculum numérique programmatique [M] |
| WASM | les instructions Relaxed SIMD (produit scalaire, FMA) accélèrent les charges existantes [S7] | `relaxed_madd` derrière un cfg [M] |

## Findings & consensus
- **Profondeur.** Avec peu de paramètres, la récursion d'un bloc partagé est la façon la plus économique d'obtenir de la profondeur [S60][S61].
- **Sortie anticipée.** La sortie doit être évaluée sur la trajectoire réelle : la règle de lecture se règle séparément de l'entraînement [S14], et un seuil choisi avec un contrôle de risque donne des garanties que le seuil naïf n'a pas [S30]. 
> [model-hint] Mesure interne skycmd : un τ calibré sur des questions isolées (0,62) faisait échouer le modèle en partie réelle, alors que τ = 0,95 retrouvait la qualité à pleine profondeur avec ~3,3 couches sur 4.

- **Ternaire.** Le QAT est nettement supérieur à la quantification après entraînement aux petites tailles [S1], et le ternaire reste viable sous 1M paramètres [S2].
- **Contrôle d'un jeu.** DAgger corrige le décalage de distribution du clonage pur. C'est la recette standard enseignée [S53][S39], et la distillation on-policy la généralise aux modèles de langage [S54][S40].

## Gaps & open problems
- Aucune source trouvée ne mesure un modèle de décision typé sous 1M paramètres en boucle fermée. C'est l'espace que skycmd occupe. [M]
- Pour le Relaxed SIMD, la source ne donne pas de chiffres en int8 [S7].
- Le clonage de comportement dans ViZDoom part des pixels [S38]. Aucune source ne traite d'un état textuel compact comme le nôtre. [M]

## Future directions
Les idées à essayer cette nuit, classées par gain attendu sur la vitesse et la qualité [M] :
1. Fusionner chaque paire « clé valeur » en un token (`fuse_numbers`), soit environ deux fois moins de tokens d'état [M], d'après [S62][S43].
2. Un encodeur d'état récursif (1 bloc × 4 boucles) avec supervision à chaque sortie et profondeur stochastique [M], d'après [S60][S61][S49].
3. Calibrer τ en boucle fermée par contrôle de risque sur les états DAgger [M], d'après [S30][S14].
4. QAT ternaire pendant toute la phase de décision [M], d'après [S2][S1].
5. Un curriculum numérique programmatique à vérité exacte [M], d'après [S42].
6. Des noyaux WASM spécialisés pour M ≤ 16 et `relaxed_madd` [M], d'après [S7][S3].


## Implémentations et inspiration biologique (2e tour)
- **Noyaux ternaires par table de correspondance** [S63][S64] :
  - bitnet.cpp propose des noyaux à table de correspondance ternaire (TL) et un noyau I2_S pour l'inférence sur CPU [S63].
  - T-MAC remplace la multiplication par des lectures de table bit à bit, sans déquantification [S64].
  - C'est la réponse à notre mesure, où l'addition et la soustraction ternaires étaient plus lentes que le f32. [M]
- **Réamorçage temporel des latents** [S65][S68] :
  - RIN amorce les latents d'une étape à partir de l'étape précédente, avec un terme initialisé à zéro [S65].
  - TRM transmet son état latent d'un passage de supervision à l'autre, détaché du graphe [S68].
  - Pour le contrôle à haute fréquence, une étude VLA réutilise des tokens d'une étape à l'autre [S66], et le Latent Recurrent Transformer réutilise l'état caché précédent comme mémoire [S76].
- **Calcul événementiel.** Les réseaux delta seuillent la variation d'activation et sautent les mises à jour sous le seuil, au prix d'une dérive qui demande des remises à zéro périodiques [S70].
- **Expand-and-sparsify** [S71] :
  - Chez la mouche, environ 50 neurones de projection alimentent environ 2 500 cellules de Kenyon, et l'inhibition k-WTA ne laisse qu'environ 5 % d'unités actives [S71].
  - Ce motif s'est montré efficace pour suivre un itinéraire en robotique embarquée [S72], et son expressivité est étudiée formellement [S73].
- **Circuits de commande inspirés de C. elegans.** Les Neural Circuit Policies sont organisées en neurones sensoriels, interneurones, neurones de commande récurrents et neurones moteurs, avec peu de paramètres [S74]. Des drones pilotés par ces réseaux ont surpassé d'autres réseaux sur les tâches difficiles, comme poursuivre une cible sous la pluie [S75].

> [model-hint] Connexion proposée pour skycmd, la « récursion temporelle » : l'encodeur d'état bouclé de s3 démarre du latent final de la frame précédente, via une porte initialisée à zéro comme dans RIN et avec un latent détaché comme dans TRM. Les jeux changent peu d'une frame à l'autre, donc la sortie anticipée devrait se déclencher dès la première boucle la plupart du temps. On obtiendrait ainsi l'économie des réseaux delta sans leur dérive.

## Open questions / contradictions
> [model-hint] TRM rapporte ses résultats sur des énigmes (ARC, Sudoku), pas sur du contrôle en temps réel. Le transfert à nos jeux reste à mesurer cette nuit.

> [model-hint] LoopFormer rapporte que l'early exit peut s'effondrer vers des états dégénérés. La supervision à chaque sortie et la profondeur stochastique sont nos garde-fous, et c'est à vérifier sur les métriques en boucle fermée.

## References (see refs.bib)
Toutes les sources sont listées dans `refs.bib` et `sources.json`.

## Sources

- [S1] TinyStories 1-bit LLM — https://huggingface.co/euclidstellar/tinystories-1bit-llm
- [S2] When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization — https://arxiv.org/html/2411.05882v1
- [S3] Optimizing the Deployment of Tiny Transformers on Low-Power MCUs — https://arxiv.org/html/2404.02945v1
- [S7] WebAssembly and WebGPU for Web AI — https://developer.chrome.com/blog/io24-webassembly-webgpu-1
- [S8] Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning — https://arxiv.org/pdf/2604.21999
- [S10] LoopFormer budget-conditioned looped LM — https://arxiv.org/html/2602.11451v1
- [S14] Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts — https://arxiv.org/pdf/2607.20519
- [S30] Fast yet Safe: Early-Exiting with Risk Control — https://arxiv.org/pdf/2405.20915
- [S36] Efficient numeracy in language models through single-token number embeddings — https://arxiv.org/pdf/2510.06824
- [S38] Behavioural Cloning in VizDoom — https://ar5iv.labs.arxiv.org/html/2401.03993
- [S39] An Introduction to Deep Reinforcement and Imitation Learning — https://arxiv.org/pdf/2512.08052
- [S40] On-policy distillation survey — https://arxiv.org/pdf/2604.00626
- [S42] decider-4b — https://huggingface.co/Mapika/decider-4b
- [S43] FoNE: Precise Single-Token Number Embeddings via Fourier Features — https://arxiv.org/html/2502.09741v1
- [S47] OPTED: On-Policy Fine-Tuning for End-to-End Driving — https://arxiv.org/pdf/2609.20756
- [S49] Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping — https://www.alphaxiv.org/abs/2606.29983v1
- [S52] Sev typed-decisions CE-only — https://huggingface.co/LakoreAI/sev
- [S53] CS224R 2026 Homework 1 — https://cs224r.stanford.edu/material/hw1/CS224R_2026_Homework_1.pdf
- [S54] On-Policy Distillation — https://www.thinkingmachines.ai/blog/on-policy-distillation/
- [S56] fragment-1 — https://huggingface.co/FrameXlabs/fragment-1
- [S60] [2510.04871] Less is More: Recursive Reasoning with Tiny Networks — https://arxiv.org/abs/2510.04871
- [S61] [2510.25741] Scaling Latent Reasoning via Looped Language Models — https://arxiv.org/abs/2510.25741
- [S62] [2106.11959] Revisiting Deep Learning Models for Tabular Data — https://arxiv.org/abs/2106.11959
- [S63] Bitnet.cpp: Efficient Edge Inference for Ternary LLMs — https://arxiv.org/abs/2502.11880
- [S64] T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge — https://arxiv.org/abs/2407.00088
- [S65] Scalable Adaptive Computation for Iterative Generation (RIN) — https://arxiv.org/abs/2212.11972
- [S66] Reducing Temporal Redundancy for Efficient Vision-Language-Action Inference — https://arxiv.org/abs/2607.12287
- [S68] Evaluating Tiny Recursive Models Across Training for Code Generation — https://arxiv.org/abs/2608.29376
- [S70] Exploiting Symmetric Temporally Sparse BPTT for Efficient RNN Training — https://arxiv.org/abs/2312.09391
- [S71] Bio-Inspired Hashing for Unsupervised Similarity Search (BioHash) — https://arxiv.org/abs/2001.04907
- [S72] Vision-based route following by an embodied insect-inspired sparse neural network — https://ar5iv.labs.arxiv.org/html/2303.08109
- [S73] Expressivity of expand-and-sparsify representations — https://arxiv.org/abs/2006.03741
- [S74] Designing Worm-inspired Neural Networks for Interpretable Robotic Control (NCP) — https://arxiv.org/abs/1809.04423v2
- [S75] MIT news: liquid neural networks drones — https://news.mit.edu/2021/cause-effect-neural-networks-1014
- [S76] Latent Recurrent Transformer — https://arxiv.org/abs/2605.26797
