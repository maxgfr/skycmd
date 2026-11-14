// Presets of the s6-nano demo on /live/: the first record of the test split of each source and
// question type in data/games (the four game oracles) and data/numeric (the numeric curriculum),
// taken by that rule and kept whatever the model answers. Native answers of the published file
// (sky-eval, full depth): 11 of the 15 expected options, the 4 misses are numeric questions.
// The instructions are in English or French, as in the data.
import type { DemoCase } from './demo-cases'

export const S6_CASES: DemoCase[] = [
  {
    "src": "game:flappy",
    "t": "choice",
    "ins": "Pick the control input for this frame so the drone clears the pipes.",
    "opts": [
      [
        "monter",
        "gain altitude now"
      ],
      [
        "drift",
        "let gravity pull the drone down"
      ]
    ],
    "state": "vy 11 dx 35 up -8 dn 20 nxt 4 alt 34",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "game:flappy",
    "t": "noul",
    "ins": "Le drone va-t-il s'écraser dans la demi-seconde s'il continue de planer ?",
    "opts": [
      [
        "false",
        "no, the statement does not hold"
      ],
      [
        "true",
        "yes, the statement holds"
      ]
    ],
    "state": "vy 0 dx 26 up -20 dn 20 nxt 4 alt 46",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "game:rescue",
    "t": "choice",
    "ins": "Which move should the rescue drone make next?",
    "opts": [
      [
        "e",
        "move one cell east"
      ],
      [
        "n",
        "move one cell north"
      ],
      [
        "go_home",
        "fly back toward the base to recharge"
      ],
      [
        "ouest",
        ""
      ],
      [
        "hold",
        "hold position"
      ],
      [
        "cannot_tell",
        "not enough information"
      ],
      [
        "down",
        "fly south"
      ]
    ],
    "state": "nh 12 nb 2 sh 10 sb 2 eh 12 eb 2 wh 10 wb 2 bat 40 wind 0 hk 3 sen ok",
    "gold": 3,
    "lang": "en"
  },
  {
    "src": "game:rescue",
    "t": "score",
    "ins": "Évalue le risque de tomber en panne de batterie avant la base.",
    "opts": [
      [
        "level 0",
        "négligeable"
      ],
      [
        "level 1",
        "faible"
      ],
      [
        "level 2",
        "modéré"
      ],
      [
        "level 3",
        "élevé"
      ],
      [
        "level 4",
        "critique"
      ]
    ],
    "state": "nh - nb 7 sh - sb 7 eh - eb 5 wh - wb 7 bat 22 wind 0 hk 0 sen ok",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "game:dodge",
    "t": "noul",
    "ins": "Is the obstacle ahead impassable, requiring the drone to stop and land?",
    "opts": [
      [
        "false",
        "keep the mission going"
      ],
      [
        "true",
        "the way is blocked"
      ]
    ],
    "state": "d 26 v 11 dz 0 hr 6 gw -2 dy 0 cl 6",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "game:dodge",
    "t": "choice",
    "ins": "Choisis la manœuvre face à l'obstacle devant.",
    "opts": [
      [
        "climb_over",
        ""
      ],
      [
        "around",
        "passer sur le côté par l'ouverture"
      ],
      [
        "continue",
        "garder le cap"
      ],
      [
        "brake",
        "ralentir pour gagner du temps"
      ]
    ],
    "state": "d 11 v 11 dz -6 hr 13 gw 0 dy 3 cl 6",
    "gold": 2,
    "lang": "fr"
  },
  {
    "src": "game:doom",
    "t": "choice",
    "ins": "Choose the next action for the player in the maze full of monsters.",
    "opts": [
      [
        "turn_right",
        "turn to the right"
      ],
      [
        "forward",
        "walk ahead"
      ],
      [
        "look_left",
        "look left"
      ],
      [
        "sidestep_right",
        ""
      ],
      [
        "slide_left",
        "slide left"
      ],
      [
        "backpedal",
        ""
      ],
      [
        "tirer",
        "fire the weapon at the crosshair"
      ],
      [
        "activer",
        "use what is in front"
      ]
    ],
    "state": "ea 0 ed 6 al 1 en 2 hp 10 am 21 wf 12 wl 12 wr 4 dr 0 gb 2 gd 2 pb 2 pd 10",
    "gold": 6,
    "lang": "en"
  },
  {
    "src": "game:doom",
    "t": "score",
    "ins": "Score the threat level around the player.",
    "opts": [
      [
        "level 0",
        "no threat in sight"
      ],
      [
        "level 1",
        "low: a distant enemy"
      ],
      [
        "level 2",
        "moderate: an enemy is engaging"
      ],
      [
        "level 3",
        "high: several enemies or one up close"
      ],
      [
        "level 4",
        "critical: overwhelmed or nearly dead"
      ]
    ],
    "state": "ea 0 ed 0 al 0 en 0 hp 10 am 19 wf 12 wl 12 wr 8 dr 0 gb 3 gd 1 pb 2 pd 3",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "game:doom",
    "t": "noul",
    "ins": "The player is about to be hit within a second unless it reacts.",
    "opts": [
      [
        "false",
        "no, the statement does not hold"
      ],
      [
        "true",
        "yes, the statement holds"
      ]
    ],
    "state": "ea 0 ed 0 al 0 en 0 hp 10 am 19 wf 12 wl 12 wr 7 dr 0 gb 2 gd 4 pb 0 pd 0",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "numeric:extreme",
    "t": "choice",
    "ins": "Quelle mesure est la plus basse ?",
    "opts": [
      [
        "cl",
        ""
      ],
      [
        "a",
        ""
      ],
      [
        "lvl",
        ""
      ]
    ],
    "state": "cl 937 a 953 lvl 48",
    "gold": 2,
    "lang": "fr"
  },
  {
    "src": "numeric:bucket",
    "t": "score",
    "ins": "Bucket the value of up into one of 5 levels.",
    "opts": [
      [
        "level 0",
        "below 0"
      ],
      [
        "level 1",
        "0 to 1"
      ],
      [
        "level 2",
        "1 to 2"
      ],
      [
        "level 3",
        "2 to 3"
      ],
      [
        "level 4",
        "above 3"
      ]
    ],
    "state": "up 6 hp 1",
    "gold": 4,
    "lang": "en"
  },
  {
    "src": "numeric:sign",
    "t": "choice",
    "ins": "What is the sign of e minus cl?",
    "opts": [
      [
        "positive",
        ""
      ],
      [
        "zero",
        ""
      ],
      [
        "negative",
        ""
      ]
    ],
    "state": "dx -10 cl -3 lvl 3 e 10",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "numeric:closest",
    "t": "choice",
    "ins": "Which value is closest to -3.7?",
    "opts": [
      [
        "dist",
        ""
      ],
      [
        "z",
        ""
      ],
      [
        "dx",
        ""
      ],
      [
        "eta",
        ""
      ]
    ],
    "state": "z 11.6 eta 12.8 dist -15.9 dx -19.9",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "numeric:between",
    "t": "noul",
    "ins": "Does vy lie between up and sb?",
    "opts": [
      [
        "false",
        "no, the statement does not hold"
      ],
      [
        "true",
        "yes, the statement holds"
      ]
    ],
    "state": "up -8 gap 8 sb 0 vy -4",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "numeric:greater",
    "t": "noul",
    "ins": "alt est-il plus petit que a ?",
    "opts": [
      [
        "false",
        "non, l'affirmation est fausse"
      ],
      [
        "true",
        "oui, l'affirmation est vraie"
      ]
    ],
    "state": "alt -6.7 z 13.7 a 6.4 wind -5.8",
    "gold": 1,
    "lang": "fr"
  }
]
