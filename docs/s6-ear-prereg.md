# s6-ear : pré-enregistrement (premier entraînement et portes)

**Statut : figé le 03/11/2026, avant toute donnée audio téléchargée en entier et avant tout entraînement de s6-ear.** Ce fichier fixe les données, les questions, le run, le témoin, l'évaluation et les portes. Les résultats seront lus à travers ce texte.

**Modifications de ce fichier :**

- tout amendement est commité, daté et motivé en section 8 ;
- il précède le run qu'il concerne ;
- il ne touche jamais un seuil après la lecture d'un score de test.

## 1. Données

Toutes les sources sont publiques, non restreintes et en CC-BY-4.0 (`DATA_LICENSES.md`, section audio). `sky-data fetch-audio --set all` (graine 1234) les écrit dans `data/ear`, qui n'est pas commité. Chaque clip est gardé sous la forme de ses log-mel (float16, `audio/<set>/<id>.npy`), coupé à ses 128 dernières trames de 80 ms. Les clips de test sont aussi gardés en WAV 16 kHz (`wav/<set>/<id>.wav`).

| set | source | splits |
|---|---|---|
| smart-turn | train et val : `pipecat-ai/smart-turn-data-v3-train` (227 025 clips, 23 langues dont le français) ; test : `pipecat-ai/smart-turn-data-v3.2-test` (31 527 clips), le jeu de test officiel de Smart Turn v3.2 | val 5 % du train par un hachage graine de l'id du clip ; tout clip du train dont l'id est aussi dans le test est retiré (disjoint par clip) |
| speech-commands | Speech Commands v0.02 (archive TensorFlow officielle) | train / val par `validation_list.txt` ; les clips de `testing_list.txt` sont écartés ; test = l'archive de test officielle à 12 classes |
| mswc | `MLCommons/ml_spoken_words`, en et fr, WAV | train / dev / test du jeu ; la première archive de chaque split et langue (`--mswc-archives 1`) |
| mimi | `openslr/librispeech_asr` (`clean`, `train.100` et `validation`) joint à `shangeth/librispeech-mimi-codes` (`train_clean_100`, `dev_clean`) par id d'énoncé | pré-entraînement seulement |

Smart Turn : le plan d'origine tirait le test dans `smart-turn-data-v3-train` (10 % par graine). Il a été remplacé avant toute donnée complète et tout run, parce que ce jeu est celui sur lequel Smart Turn est entraîné : la barre aurait été mesurée sur ses propres données d'entraînement. Le jeu de test officiel de v3.2 n'a été vu par aucun des deux modèles.

Speech Commands : les 25 mots hors des 10 mots du benchmark donnent la classe « unknown », gardés avec une probabilité de 1/6 en train et en val. La classe « silence » est faite de découpes de 1 s, toutes les 0,5 s, des bruits de `_background_noise_`, dont 10 % vont en val.

Mimi : les codes sont à 12,5 Hz, comme nos trames. Un énoncé a en général une trame de code de plus que de trames log-mel (arrondi au-dessus contre arrondi en dessous) : la paire est coupée à la plus courte. Seul le codebook 0 est utilisé : c'est le codebook sémantique, distillé de WavLM.

## 2. Questions (le jev audio, `sky-data audio-jev`)

Un record par question, l'état étant le chemin du `.npy` du clip. Les textes sont fixés dans `crates/sky-data/src/audio.rs` :

- **`turn_end`** (noul) : « Has the speaker finished their turn? ». La cible est `endpoint_bool`. Options : « false: the speaker is still talking or will go on », « true: the speaker has finished their turn ».
- **`keyword`** (choice, 12 options) : « Which word was said? », sur yes, no, up, down, left, right, on, off, stop, go, « silence: nothing was said » et « unknown: another word ».
- **`keyword_zs`** (choice, 10 options) : la même question sur MSWC.
  - Pour chaque langue, 50 mots qui ont au moins 20 clips de test sont réservés, tirés par la graine.
  - Une question de test porte sur un mot réservé. Ses 9 distracteurs sont d'autres mots réservés. Chaque mot réservé donne exactement 20 questions de test (ses 20 premiers clips), donc le mot juste est uniforme sur les mots réservés.
  - Les questions de train et de val portent sur les autres mots, avec des distracteurs pris parmi eux.
  - Aucun mot réservé n'apparaît à l'entraînement, ni comme réponse ni comme option.

## 3. Modèle

`configs/s6-ear-nano.json` : la forme de s6-nano (d 144, ff 288, 2 blocs × 2 boucles, sorties 1 à 4, Parallax, 2 têtes, tokenizer `tok-s6-2048`), avec `state_kind: "audio"` et une fenêtre d'état de 128 trames (10,24 s). Le décompte des paramètres est figé par un test :

- modèle de décision : 1 615 681 paramètres (s6-nano plus `audio_proj`, 46 224) ;
- pile de pré-entraînement : 821 552.

Smart Turn v3.2 en a 8M : s6-ear-nano en fait environ 1/5. La taille n'est pas une condition de porte, elle est fixée par la config.

## 4. Le run (`scripts/train-ear.sh data pretrain decide eval`)

- **Pré-entraînement** (`s2-pretrain --audio`) : 60 min, 64 découpes de 128 trames par lot, 30 % des trames masquées par blocs de 3. La cible est l'id Mimi du codebook 0 de chaque trame masquée. Le rapport donne la précision de ce codebook sur les trames masquées de val.
- **Décision** (`s2-decide`) :
  - 60 min sur `turn_end=0.45,keyword=0.30,keyword_zs=0.25`, depuis l'init du pré-entraînement ;
  - QAT ternaire sur les 10 derniers %, `--loop-dropout 0.25` ;
  - puis `s2-calibrate` sur val et l'export ternaire.
- **Témoin** (pré-enregistré) : le même modèle, les mêmes données et le même budget de décision, **sans** pré-entraînement Mimi (init aléatoire). Il est rapporté à côté de chaque porte, sans être lui-même une porte : il mesure ce qu'apporte la distillation de Mimi.
- **Seuils**, sur val, à pleine profondeur, toutes les sorties (`sky-eval --dump-exits`, `scripts/ear-gate.py crc`) :
  - **seuils de sortie anticipée** par contrôle conformal du risque, α = 0,02 (SPEC-S6 §9). Ils sont écrits dans une copie `-crc` du GGUF ;
  - **seuil de renvoi au Système 2** ρ, α = 0,05 : le taux d'erreur parmi les décisions que s6-ear prend lui-même. Il est rapporté avec la couverture.

Ce run n'est pas lancé en même temps qu'un autre entraînement sur le GPU (le run LFM3 du §6 de TODO.md).

## 5. Évaluation (split test, modèle ternaire)

- `sky-eval` à pleine profondeur (τ = 1), puis aux seuils CRC, puis le témoin. Les clips de test sont lus en WAV, par le chemin PCM de `Engine::decide_audio` (le chemin déployé).
- `scripts/ear-baseline.py` : Smart Turn v3.2 (`smart-turn-v3.2-cpu.onnx`, révision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`) sur les mêmes clips de test de `turn_end` (le jeu de test officiel de v3.2). On reprend l'inférence de référence de Pipecat : les 8 dernières secondes, les features Whisper, et `probability > 0.5`.
- La parité WASM sur 20 clips de test (`parity-node.mjs` : écart de probabilité ≤ 1e-4, même argmax).
- La latence médiane native par décision.

## 6. Portes (v1)

Toutes sont lues sur le modèle ternaire à pleine profondeur. Leur métrique est l'**exactitude** (la part de bonnes réponses) :

1. **Fin de tour** : exactitude sur `ear:turn_end` ≥ celle de Smart Turn v3.2 sur les mêmes clips.
2. **Mots-clés** : exactitude sur `ear:keyword` (test officiel, 12 options) ≥ 0,90.
3. **Mots-clés zéro-shot** : exactitude sur `ear:keyword_zs:en` **et** sur `ear:keyword_zs:fr` ≥ 0,20, soit 2 × le hasard à 10 options. La barre est délibérément basse : c'est un premier essai.

`scripts/ear-gate.py gate` les calcule. Si une porte échoue, le modèle n'est pas publié, mais le résultat va quand même dans le rapport. La précision et le rappel de la fin de tour sont rapportés pour les deux modèles.

Le modèle est publié aux seuils CRC si les portes y tiennent comme à pleine profondeur, à pleine profondeur si elles ne tiennent que là, et n'est pas publié sinon.

## 7. v2 (code et pré-enregistrement maintenant ; données et runs ensuite)

- **`duplex_action`** (choice) : « What should I do now? », avec les options listen, backchannel, speak, yield.
  - Les labels viennent d'AMI (CC-BY-4.0 : un micro par orateur et des horodatages), par la règle `audio::duplex_label`, testée.
  - Au temps t, pour l'agent A :
    - si A parle et qu'un autre orateur commence dans les 500 ms, hors acte « backchannel », pendant que A parle encore, c'est **yield** ;
    - si A commence un segment dans les 500 ms, c'est **backchannel** si c'est un acte backchannel, sinon **speak** ;
    - sinon, c'est **listen**.
  - **Porte** : battre la règle « parler après 500 ms de silence » (`audio::silence_rule`) en exactitude sur le split test d'AMI.
  - Le téléchargeur AMI reste à écrire : c'est une donnée v2.
- **`about`** (noul et choice sur le contenu, `audio::about_questions`) : distillation intermodale.
  - `sky-data teacher label-transcripts` demande à GLM-5.3-flash une probabilité par option sur chaque transcription LibriSpeech de `data/ear/mimi`. Le labelleur est celui du §1.1, qui reprend après une interruption et isole les refus de la modération.
  - `audio-jev --about` en fait des cibles douces sur l'audio : le professeur lit le texte, l'élève entend le son.
  - La labellisation est lancée **après** celle du §1.1 de TODO.md, qui utilise le même quota z.ai.
  - **Porte** : exactitude ≥ 0,60 sur les questions binaires (val, contre l'argmax du professeur).

## 8. Amendements

**Amendement 2 (09/11/2026, avant tout entraînement de s6-ear : seule la recette pico tourne, aucun de ses résultats n'a été lu, aucun score de s6-ear n'existe). La recette et le mode bridé.** Il n'y a jamais eu d'amendement 1 : le numéro 2 est celui que citent déjà `scripts/train-ear.sh` et `scripts/s6-adopt.py`.

- **La recette.** s6-ear tourne après l'ablation pico (`docs/s6-prereg.md`, §3 et §4) et prend ses options (`scripts/s6-adopt.py env s6-ear-nano`), pour le modèle **et** pour son témoin : `CONFIG`, `OPT`, `DECIDE`, `LR_SCALE`.
- **Le taux.** Le taux du pilote pico est déplacé à d = 144 par μP, comme pour s6-nano : `LR_SCALE` × √(96/144) × le taux par défaut de la phase.
- **Le curriculum.** Il ordonne des documents texte, et le pré-entraînement audio n'en a pas : il est sans effet ici, donc `PRETRAIN` reste vide.
- **Le contrôle après le pré-entraînement.** `val_acc_cb0` doit valoir au moins 10/2048 (10 fois le hasard), seuil écrit avant le run. En dessous, c'est un échec technique : il est noté dans le rapport et `decide` n'est pas lancé.
- **Le mode bridé.** Le run se fait en mode bridé (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), comme l'amendement 5 de `docs/s6-prereg.md`. Les budgets en minutes ne comptent que le temps de travail, et le rapport donne le nombre de pas réellement faits.
- **Ce qui ne change pas.** Données, questions, modèle, budgets, seuils α, portes.
