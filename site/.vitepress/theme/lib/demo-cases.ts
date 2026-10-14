// Generated from eval/s5/demo-cases.jsonl, the cases every s5 model is evaluated on: the /live
// presets ask exactly these questions, with these options and states. Regenerate it when that file
// changes (node scripts/demo-cases-ts.mjs).

export interface DemoCase {
  src: string
  t: 'choice' | 'score' | 'noul'
  ins: string
  /** [key, definition] */
  opts: [string, string][]
  state: string
  /** Index of the expected option. */
  gold: number
  lang: 'en' | 'fr'
}

export const DEMO_CASES: DemoCase[] = [
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Can you put lunch with Sarah on my calendar for Friday at noon?",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Will it rain in Lyon tomorrow afternoon?",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Email Tom that the meeting moved to 3 pm.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "What is 17 % of 2,340?",
    "gold": 4,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Who won the cycling race in Paris yesterday?",
    "gold": 3,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Thanks, that's all for today!",
    "gold": 5,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Schedule a dentist appointment next Tuesday at 9.",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "demo_tools_en",
    "t": "choice",
    "ins": "Which tool should the assistant call to handle this request?",
    "opts": [
      [
        "get_weather",
        "get_weather(): current weather and forecast for a city."
      ],
      [
        "send_email",
        "send_email(): send an email to a contact."
      ],
      [
        "create_event",
        "create_event(): add an event to the calendar."
      ],
      [
        "search_web",
        "search_web(): search the web for recent information."
      ],
      [
        "calculator",
        "calculator(): evaluate an arithmetic expression."
      ],
      [
        "no_tool",
        "No tool fits: reply directly or ask for the missing details."
      ]
    ],
    "state": "Do I need an umbrella in London this evening?",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Tu peux mettre un déjeuner avec Sarah vendredi à midi dans mon agenda ?",
    "gold": 2,
    "lang": "fr"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Il va pleuvoir à Lyon demain après-midi ?",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Écris à Tom que la réunion est décalée à 15 h.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Combien font 17 % de 2 340 ?",
    "gold": 4,
    "lang": "fr"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Qui a gagné la course cycliste à Paris hier ?",
    "gold": 3,
    "lang": "fr"
  },
  {
    "src": "demo_tools_fr",
    "t": "choice",
    "ins": "Quel outil l'assistant doit-il appeler pour traiter cette demande ?",
    "opts": [
      [
        "get_weather",
        "get_weather() : météo actuelle et prévisions pour une ville."
      ],
      [
        "send_email",
        "send_email() : envoyer un e-mail à un contact."
      ],
      [
        "create_event",
        "create_event() : ajouter un événement à l'agenda."
      ],
      [
        "search_web",
        "search_web() : chercher des informations récentes sur le web."
      ],
      [
        "calculator",
        "calculator() : calculer une expression arithmétique."
      ],
      [
        "aucun_outil",
        "Aucun outil ne convient : répondre directement ou demander les précisions."
      ]
    ],
    "state": "Merci, ce sera tout pour aujourd'hui !",
    "gold": 5,
    "lang": "fr"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "Hi, I was charged twice for my March subscription, can you send the money back?",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "The app crashes every time I open the settings page.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "It would be great to have a dark mode.",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "I forgot my password and the reset email never arrives.",
    "gold": 3,
    "lang": "en"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "What time does your office in Berlin open?",
    "gold": 4,
    "lang": "en"
  },
  {
    "src": "demo_triage_en",
    "t": "choice",
    "ins": "Which category does this customer message belong to?",
    "opts": [
      [
        "billing",
        "Invoices, charges, payments or refunds."
      ],
      [
        "bug",
        "Something in the app is broken or crashes."
      ],
      [
        "feature_request",
        "An idea or a wish for the product."
      ],
      [
        "account",
        "Login, password or profile problems."
      ],
      [
        "unsure",
        "None of the listed categories fits the message."
      ]
    ],
    "state": "My invoice shows the wrong VAT number.",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "Bonjour, j'ai été prélevé deux fois pour mon abonnement de mars, pouvez-vous me rembourser ?",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "L'application plante dès que j'ouvre les réglages.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "Ce serait génial d'avoir un mode sombre.",
    "gold": 2,
    "lang": "fr"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "J'ai oublié mon mot de passe et je ne reçois jamais l'e-mail de réinitialisation.",
    "gold": 3,
    "lang": "fr"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "À quelle heure ouvre votre bureau de Berlin ?",
    "gold": 4,
    "lang": "fr"
  },
  {
    "src": "demo_triage_fr",
    "t": "choice",
    "ins": "À quelle catégorie appartient ce message client ?",
    "opts": [
      [
        "facturation",
        "Factures, prélèvements, paiements ou remboursements."
      ],
      [
        "bug",
        "Quelque chose ne marche pas ou plante dans l'application."
      ],
      [
        "suggestion",
        "Une idée ou une demande d'amélioration du produit."
      ],
      [
        "compte",
        "Connexion, mot de passe ou profil."
      ],
      [
        "incertain",
        "Aucune des catégories proposées ne correspond au message."
      ]
    ],
    "state": "Ma facture indique un mauvais numéro de TVA.",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "What should the on-call engineer do first?",
    "opts": [
      [
        "rollback",
        "Revert the last deploy."
      ],
      [
        "scale",
        "Add more servers to absorb the load."
      ],
      [
        "wait",
        "Keep watching: nothing is wrong."
      ],
      [
        "escalate",
        "Page the database team."
      ]
    ],
    "state": "Error rate jumped from 0.1% to 12% two minutes after the 14:02 deploy. CPU and memory are normal. The database latency is unchanged.",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "What should the on-call engineer do first?",
    "opts": [
      [
        "rollback",
        "Revert the last deploy."
      ],
      [
        "scale",
        "Add more servers to absorb the load."
      ],
      [
        "wait",
        "Keep watching: nothing is wrong."
      ],
      [
        "escalate",
        "Page the database team."
      ]
    ],
    "state": "Traffic doubled after a marketing email; every server is at 95% CPU and requests queue up. No deploy today.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "What should the on-call engineer do first?",
    "opts": [
      [
        "rollback",
        "Revert the last deploy."
      ],
      [
        "scale",
        "Add more servers to absorb the load."
      ],
      [
        "wait",
        "Keep watching: nothing is wrong."
      ],
      [
        "escalate",
        "Page the database team."
      ]
    ],
    "state": "All dashboards are green, error rate 0.1% as usual, one user reported a slow page once.",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "What should the on-call engineer do first?",
    "opts": [
      [
        "rollback",
        "Revert the last deploy."
      ],
      [
        "scale",
        "Add more servers to absorb the load."
      ],
      [
        "wait",
        "Keep watching: nothing is wrong."
      ],
      [
        "escalate",
        "Page the database team."
      ]
    ],
    "state": "Database replication lag is 40 minutes and growing; queries time out; no deploy since yesterday.",
    "gold": 3,
    "lang": "en"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Que doit faire l'ingénieur d'astreinte en premier ?",
    "opts": [
      [
        "rollback",
        "Annuler le dernier déploiement."
      ],
      [
        "scale",
        "Ajouter des serveurs pour absorber la charge."
      ],
      [
        "wait",
        "Continuer à surveiller : rien d'anormal."
      ],
      [
        "escalate",
        "Appeler l'équipe base de données."
      ]
    ],
    "state": "Le taux d'erreur est passé de 0,1 % à 12 % deux minutes après le déploiement de 14 h 02. CPU et mémoire normaux, latence de la base inchangée.",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Que doit faire l'ingénieur d'astreinte en premier ?",
    "opts": [
      [
        "rollback",
        "Annuler le dernier déploiement."
      ],
      [
        "scale",
        "Ajouter des serveurs pour absorber la charge."
      ],
      [
        "wait",
        "Continuer à surveiller : rien d'anormal."
      ],
      [
        "escalate",
        "Appeler l'équipe base de données."
      ]
    ],
    "state": "Le trafic a doublé après un e-mail marketing ; tous les serveurs sont à 95 % de CPU et les requêtes s'accumulent. Aucun déploiement aujourd'hui.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Que doit faire l'ingénieur d'astreinte en premier ?",
    "opts": [
      [
        "rollback",
        "Annuler le dernier déploiement."
      ],
      [
        "scale",
        "Ajouter des serveurs pour absorber la charge."
      ],
      [
        "wait",
        "Continuer à surveiller : rien d'anormal."
      ],
      [
        "escalate",
        "Appeler l'équipe base de données."
      ]
    ],
    "state": "Tous les tableaux de bord sont au vert, taux d'erreur habituel de 0,1 %, un utilisateur a signalé une page lente une fois.",
    "gold": 2,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Que doit faire l'ingénieur d'astreinte en premier ?",
    "opts": [
      [
        "rollback",
        "Annuler le dernier déploiement."
      ],
      [
        "scale",
        "Ajouter des serveurs pour absorber la charge."
      ],
      [
        "wait",
        "Continuer à surveiller : rien d'anormal."
      ],
      [
        "escalate",
        "Appeler l'équipe base de données."
      ]
    ],
    "state": "Le retard de réplication de la base atteint 40 minutes et augmente ; les requêtes expirent ; aucun déploiement depuis hier.",
    "gold": 3,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Je suis fatigué",
    "opts": [
      [
        "dormir",
        "Pour se reposer."
      ],
      [
        "faire_la_fete",
        "Pour s'amuser."
      ]
    ],
    "state": "je suis ko",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Je suis fatigué",
    "opts": [
      [
        "dormir",
        "Pour se reposer."
      ],
      [
        "faire_la_fete",
        "Pour s'amuser."
      ]
    ],
    "state": "je suis épuisé, je n'arrive plus à garder les yeux ouverts",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_playground_fr",
    "t": "choice",
    "ins": "Que faire ce soir ?",
    "opts": [
      [
        "dormir",
        "Pour se reposer."
      ],
      [
        "faire_la_fete",
        "Pour s'amuser."
      ]
    ],
    "state": "je suis en pleine forme et c'est l'anniversaire de mon meilleur ami",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "I am tired",
    "opts": [
      [
        "sleep",
        "To get some rest."
      ],
      [
        "party",
        "To have fun."
      ]
    ],
    "state": "I'm completely knackered",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_playground_en",
    "t": "choice",
    "ins": "What to do tonight?",
    "opts": [
      [
        "sleep",
        "To get some rest."
      ],
      [
        "party",
        "To have fun."
      ]
    ],
    "state": "I feel great and it's my best friend's birthday",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "The message violates this rule: \"No sharing of phone numbers, emails or home addresses.\"",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "Call me tonight at 06 12 34 56 78, I live at 4 rue des Lilas.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "The message violates this rule: \"No sharing of phone numbers, emails or home addresses.\"",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "Call me tonight, you have my number from last time.",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "The message violates this rule: \"No sharing of phone numbers, emails or home addresses.\"",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "My email is jane.doe@example.com, write me there.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "The message violates this rule: \"No sharing of phone numbers, emails or home addresses.\"",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "Thanks for the quick answer, have a nice day.",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_guard_fr",
    "t": "noul",
    "ins": "Le message enfreint cette règle : « Pas de numéros de téléphone, d'e-mails ni d'adresses postales. »",
    "opts": [
      [
        "false",
        "Le message respecte la règle."
      ],
      [
        "true",
        "Le message enfreint la règle."
      ]
    ],
    "state": "Appelle-moi ce soir au 06 12 34 56 78, j'habite 4 rue des Lilas.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_guard_fr",
    "t": "noul",
    "ins": "Le message enfreint cette règle : « Pas de numéros de téléphone, d'e-mails ni d'adresses postales. »",
    "opts": [
      [
        "false",
        "Le message respecte la règle."
      ],
      [
        "true",
        "Le message enfreint la règle."
      ]
    ],
    "state": "Appelle-moi ce soir, tu as déjà mon numéro.",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_guard_fr",
    "t": "noul",
    "ins": "Le message enfreint cette règle : « Pas de numéros de téléphone, d'e-mails ni d'adresses postales. »",
    "opts": [
      [
        "false",
        "Le message respecte la règle."
      ],
      [
        "true",
        "Le message enfreint la règle."
      ]
    ],
    "state": "Mon e-mail est jeanne.dupont@example.com, écris-moi là.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_guard_fr",
    "t": "noul",
    "ins": "Le message enfreint cette règle : « Pas de numéros de téléphone, d'e-mails ni d'adresses postales. »",
    "opts": [
      [
        "false",
        "Le message respecte la règle."
      ],
      [
        "true",
        "Le message enfreint la règle."
      ]
    ],
    "state": "Merci pour la réponse rapide, bonne journée.",
    "gold": 0,
    "lang": "fr"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "This agent action breaks the rule: production changes need an approved change ticket.",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "{\"action\": \"delete records\", \"target\": \"production\", \"approved change ticket\": \"none\"}",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "This agent action breaks the rule: production changes need an approved change ticket.",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "{\"action\": \"read logs\", \"target\": \"production\", \"approved change ticket\": \"none\"}",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_guard_en",
    "t": "noul",
    "ins": "This agent action breaks the rule: production changes need an approved change ticket.",
    "opts": [
      [
        "false",
        "The message respects the rule."
      ],
      [
        "true",
        "The message breaks the rule."
      ]
    ],
    "state": "{\"action\": \"restart service\", \"target\": \"staging\", \"approved change ticket\": \"none\"}",
    "gold": 0,
    "lang": "en"
  },
  {
    "src": "demo_score_en",
    "t": "score",
    "ins": "How severe is this incident?",
    "opts": [
      [
        "0",
        "No impact."
      ],
      [
        "1",
        "Minor: a few users, a workaround exists."
      ],
      [
        "2",
        "Major: many users blocked."
      ],
      [
        "3",
        "Critical: the whole service is down or data is at risk."
      ]
    ],
    "state": "The checkout page returns 500 for every customer since 10:00; no order can be placed.",
    "gold": 3,
    "lang": "en"
  },
  {
    "src": "demo_score_en",
    "t": "score",
    "ins": "How severe is this incident?",
    "opts": [
      [
        "0",
        "No impact."
      ],
      [
        "1",
        "Minor: a few users, a workaround exists."
      ],
      [
        "2",
        "Major: many users blocked."
      ],
      [
        "3",
        "Critical: the whole service is down or data is at risk."
      ]
    ],
    "state": "The export button is misaligned on Safari; exporting still works.",
    "gold": 1,
    "lang": "en"
  },
  {
    "src": "demo_score_en",
    "t": "score",
    "ins": "How severe is this incident?",
    "opts": [
      [
        "0",
        "No impact."
      ],
      [
        "1",
        "Minor: a few users, a workaround exists."
      ],
      [
        "2",
        "Major: many users blocked."
      ],
      [
        "3",
        "Critical: the whole service is down or data is at risk."
      ]
    ],
    "state": "Login fails for all users of the EU region, about 40% of customers.",
    "gold": 2,
    "lang": "en"
  },
  {
    "src": "demo_score_fr",
    "t": "score",
    "ins": "Quelle est la gravité de cet incident ?",
    "opts": [
      [
        "0",
        "Aucun impact."
      ],
      [
        "1",
        "Mineur : quelques utilisateurs, un contournement existe."
      ],
      [
        "2",
        "Majeur : beaucoup d'utilisateurs bloqués."
      ],
      [
        "3",
        "Critique : tout le service est coupé ou des données sont en danger."
      ]
    ],
    "state": "La page de paiement renvoie une erreur 500 pour tous les clients depuis 10 h ; aucune commande possible.",
    "gold": 3,
    "lang": "fr"
  },
  {
    "src": "demo_score_fr",
    "t": "score",
    "ins": "Quelle est la gravité de cet incident ?",
    "opts": [
      [
        "0",
        "Aucun impact."
      ],
      [
        "1",
        "Mineur : quelques utilisateurs, un contournement existe."
      ],
      [
        "2",
        "Majeur : beaucoup d'utilisateurs bloqués."
      ],
      [
        "3",
        "Critique : tout le service est coupé ou des données sont en danger."
      ]
    ],
    "state": "Le bouton d'export est décalé sur Safari ; l'export fonctionne toujours.",
    "gold": 1,
    "lang": "fr"
  },
  {
    "src": "demo_score_fr",
    "t": "score",
    "ins": "Quelle est la gravité de cet incident ?",
    "opts": [
      [
        "0",
        "Aucun impact."
      ],
      [
        "1",
        "Mineur : quelques utilisateurs, un contournement existe."
      ],
      [
        "2",
        "Majeur : beaucoup d'utilisateurs bloqués."
      ],
      [
        "3",
        "Critique : tout le service est coupé ou des données sont en danger."
      ]
    ],
    "state": "La connexion échoue pour tous les utilisateurs de la région UE, environ 40 % des clients.",
    "gold": 2,
    "lang": "fr"
  }
]
