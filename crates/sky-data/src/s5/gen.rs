//! Code generators of s5 training data, in English and French. Every label is computed by
//! code from the generated case (no model writes anything here):
//!
//! - [`rules`]: written policies and guardrails, half of the cases violating them, each case
//!   with a **minimal pair** (one attribute moved across the rule's boundary, label flipped);
//! - [`tone`]: the same complaint written calmly or angrily (tone pairs);
//! - [`buried`]: a decisive fact buried among filler sentences, or absent, in which case the
//!   target is uniform (missing evidence);
//! - [`paraphrase_pairs`]: an existing question with its definitions rewritten by templates,
//!   paired with the original for the KL term (the evaluation paraphrases are written by hand,
//!   in `eval/s5/`, and share none of these templates);
//! - [`distractor_pairs`]: an existing question with irrelevant sentences from a real text
//!   corpus inserted in its state, paired with the original (the evaluation distractors come
//!   from another corpus).
//!
//! Pairs share the `uid` prefix before `#` (sky-train links them by it).

use crate::rng::Rng;
use crate::s5::{split_of, uid_of, S5Record};

#[allow(clippy::too_many_arguments)]
fn rec(
    src: &str,
    t: &str,
    ins: String,
    kv: Vec<(String, String)>,
    state: String,
    target: Vec<f32>,
    lang: &str,
    uid: String,
) -> S5Record {
    let split = split_of(&state, src, 0.05, 0.05).to_string();
    S5Record {
        src: src.into(),
        t: t.into(),
        ins,
        opts_kv: kv,
        state,
        target,
        split,
        uid,
        lang: if lang == "en" {
            String::new()
        } else {
            lang.into()
        },
        gold: true,
    }
}

fn kv(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn pick<'a, T>(rng: &mut Rng, v: &'a [T]) -> &'a T {
    &v[rng.below(v.len())]
}

fn one_hot(k: usize, i: usize) -> Vec<f32> {
    (0..k).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
}

// ------------------------------------------------------------------------------------ rules

/// A generated case of a policy: its attributes as (key, value) pairs in both languages.
struct Case {
    attrs_en: Vec<(String, String)>,
    attrs_fr: Vec<(String, String)>,
    ok: bool,
    /// The same case with one attribute moved across the boundary (label flipped).
    flipped: Option<Box<Case>>,
}

/// A policy: its rule text and a case generator `(rng, violate) -> Case`.
struct Policy {
    name: &'static str,
    rule_en: &'static str,
    rule_fr: &'static str,
    gen: fn(&mut Rng, bool) -> Case,
}

fn attrs(pairs: &[(&str, String)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn refund_case(rng: &mut Rng, violate: bool) -> Case {
    // Approved iff days <= 30 and (unopened or damaged) and amount <= 200.
    let mut days = 1 + rng.below(30);
    let mut amount = 10 + rng.below(190);
    let conds = ["unopened", "damaged", "opened"];
    let mut cond = if rng.f64() < 0.5 { 0 } else { 1 };
    let which = rng.below(3);
    if violate {
        match which {
            0 => days = 31 + rng.below(60),
            1 => amount = 201 + rng.below(400),
            _ => cond = 2,
        }
    }
    let build = |days: usize, amount: usize, cond: usize| {
        let fr_cond = ["non ouvert", "endommagé", "ouvert"][cond];
        Case {
            attrs_en: attrs(&[
                ("days since purchase", days.to_string()),
                ("amount", format!("{amount} EUR")),
                ("item condition", conds[cond].into()),
            ]),
            attrs_fr: attrs(&[
                ("jours depuis l'achat", days.to_string()),
                ("montant", format!("{amount} EUR")),
                ("état de l'article", fr_cond.into()),
            ]),
            ok: days <= 30 && amount <= 200 && cond != 2,
            flipped: None,
        }
    };
    let mut c = build(days, amount, cond);
    // Minimal pair: move the attribute that decides (or the first one) across the boundary.
    let f = if violate {
        match which {
            0 => build(30, amount, cond),
            1 => build(days, 200, cond),
            _ => build(days, amount, 0),
        }
    } else {
        match rng.below(3) {
            0 => build(31, amount, cond),
            1 => build(days, 201, cond),
            _ => build(days, amount, 2),
        }
    };
    c.flipped = Some(Box::new(f));
    c
}

fn access_case(rng: &mut Rng, violate: bool) -> Case {
    // Granted iff (sensitivity is low) or (manager approval and MFA enrolled).
    let build = |sens: bool, approval: bool, mfa: bool| Case {
        attrs_en: attrs(&[
            (
                "resource sensitivity",
                if sens { "high" } else { "low" }.into(),
            ),
            (
                "manager approval",
                if approval { "yes" } else { "no" }.into(),
            ),
            ("MFA enrolled", if mfa { "yes" } else { "no" }.into()),
        ]),
        attrs_fr: attrs(&[
            (
                "sensibilité de la ressource",
                if sens { "élevée" } else { "faible" }.into(),
            ),
            (
                "accord du manager",
                if approval { "oui" } else { "non" }.into(),
            ),
            ("MFA activée", if mfa { "oui" } else { "non" }.into()),
        ]),
        ok: !sens || (approval && mfa),
        flipped: None,
    };
    let (sens, approval, mfa) = if violate {
        (true, rng.f64() < 0.5, false)
    } else if rng.f64() < 0.5 {
        (false, rng.f64() < 0.5, rng.f64() < 0.5)
    } else {
        (true, true, true)
    };
    let (sens, approval, mfa) = if violate && approval && mfa {
        (true, true, false)
    } else {
        (sens, approval, mfa)
    };
    let mut c = build(sens, approval, mfa);
    let f = if c.ok {
        if sens {
            build(true, true, false)
        } else {
            build(true, false, false)
        }
    } else {
        build(true, true, true)
    };
    c.flipped = Some(Box::new(f));
    c
}

fn expense_case(rng: &mut Rng, violate: bool) -> Case {
    // Reimbursable iff a receipt is attached and the amount is within the category limit.
    let cats = [
        ("meals", "repas", 60),
        ("travel", "transport", 500),
        ("software", "logiciel", 300),
    ];
    let ci = rng.below(cats.len());
    let (c_en, c_fr, limit) = cats[ci];
    let build = |amount: usize, receipt: bool| Case {
        attrs_en: attrs(&[
            ("category", c_en.into()),
            ("amount", format!("{amount} EUR")),
            (
                "receipt attached",
                if receipt { "yes" } else { "no" }.into(),
            ),
            ("category limit", format!("{limit} EUR")),
        ]),
        attrs_fr: attrs(&[
            ("catégorie", c_fr.into()),
            ("montant", format!("{amount} EUR")),
            (
                "justificatif joint",
                if receipt { "oui" } else { "non" }.into(),
            ),
            ("plafond de la catégorie", format!("{limit} EUR")),
        ]),
        ok: receipt && amount <= limit,
        flipped: None,
    };
    let mut amount = 1 + rng.below(limit);
    let mut receipt = true;
    if violate {
        if rng.f64() < 0.5 {
            amount = limit + 1 + rng.below(limit);
        } else {
            receipt = false;
        }
    }
    let mut c = build(amount, receipt);
    let f = if c.ok {
        build(limit + 1, true)
    } else if !receipt {
        build(amount.min(limit), true)
    } else {
        build(limit, true)
    };
    c.flipped = Some(Box::new(f));
    c
}

fn data_case(rng: &mut Rng, violate: bool) -> Case {
    // Sharing allowed iff no personal data, or the recipient is internal and consent exists.
    let build = |pii: bool, external: bool, consent: bool| Case {
        attrs_en: attrs(&[
            (
                "contains personal data",
                if pii { "yes" } else { "no" }.into(),
            ),
            (
                "recipient",
                if external {
                    "external partner"
                } else {
                    "internal team"
                }
                .into(),
            ),
            (
                "customer consent on file",
                if consent { "yes" } else { "no" }.into(),
            ),
        ]),
        attrs_fr: attrs(&[
            (
                "contient des données personnelles",
                if pii { "oui" } else { "non" }.into(),
            ),
            (
                "destinataire",
                if external {
                    "partenaire externe"
                } else {
                    "équipe interne"
                }
                .into(),
            ),
            (
                "consentement du client",
                if consent { "oui" } else { "non" }.into(),
            ),
        ]),
        ok: !pii || (!external && consent),
        flipped: None,
    };
    let (pii, external, consent) = if violate {
        if rng.f64() < 0.5 {
            (true, true, rng.f64() < 0.5)
        } else {
            (true, false, false)
        }
    } else if rng.f64() < 0.5 {
        (false, rng.f64() < 0.5, rng.f64() < 0.5)
    } else {
        (true, false, true)
    };
    let mut c = build(pii, external, consent);
    let f = if c.ok {
        build(true, true, consent)
    } else {
        build(false, external, consent)
    };
    c.flipped = Some(Box::new(f));
    c
}

fn agent_case(rng: &mut Rng, violate: bool) -> Case {
    // An agent action is allowed iff it is read-only, or the target is staging, or an
    // approved change ticket is referenced.
    let actions = [
        ("read logs", "lire les journaux", true),
        ("delete records", "supprimer des enregistrements", false),
        ("restart service", "redémarrer le service", false),
        (
            "send email to customers",
            "envoyer un e-mail aux clients",
            false,
        ),
    ];
    let build = |a: usize, prod: bool, ticket: bool| Case {
        attrs_en: attrs(&[
            ("action", actions[a].0.into()),
            ("target", if prod { "production" } else { "staging" }.into()),
            (
                "approved change ticket",
                if ticket {
                    "CHG-".to_string() + &(1000 + a * 37).to_string()
                } else {
                    "none".into()
                },
            ),
        ]),
        attrs_fr: attrs(&[
            ("action", actions[a].1.into()),
            (
                "cible",
                if prod { "production" } else { "préproduction" }.into(),
            ),
            (
                "ticket de changement approuvé",
                if ticket {
                    "CHG-".to_string() + &(1000 + a * 37).to_string()
                } else {
                    "aucun".into()
                },
            ),
        ]),
        ok: actions[a].2 || !prod || ticket,
        flipped: None,
    };
    let (a, prod, ticket) = if violate {
        (1 + rng.below(3), true, false)
    } else {
        match rng.below(3) {
            0 => (0, rng.f64() < 0.5, false),
            1 => (1 + rng.below(3), false, false),
            _ => (1 + rng.below(3), true, true),
        }
    };
    let mut c = build(a, prod, ticket);
    let f = if c.ok {
        build(1 + (a % 3), true, false)
    } else {
        build(a, true, true)
    };
    c.flipped = Some(Box::new(f));
    c
}

const POLICIES: &[Policy] = &[
    Policy {
        name: "refund",
        rule_en: "refunds are approved within 30 days of purchase, for unopened or damaged items, up to 200 EUR",
        rule_fr: "un remboursement est accepté dans les 30 jours suivant l'achat, pour un article non ouvert ou endommagé, jusqu'à 200 EUR",
        gen: refund_case,
    },
    Policy {
        name: "access",
        rule_en: "access to a high-sensitivity resource needs manager approval and MFA; low-sensitivity resources are always granted",
        rule_fr: "l'accès à une ressource sensible exige l'accord du manager et la MFA ; les ressources peu sensibles sont toujours accordées",
        gen: access_case,
    },
    Policy {
        name: "expense",
        rule_en: "an expense is reimbursed only with a receipt and within the limit of its category",
        rule_fr: "une note de frais n'est remboursée qu'avec un justificatif et dans la limite du plafond de sa catégorie",
        gen: expense_case,
    },
    Policy {
        name: "data_sharing",
        rule_en: "personal data may only be shared with an internal team and with the customer's consent",
        rule_fr: "les données personnelles ne peuvent être partagées qu'avec une équipe interne et avec le consentement du client",
        gen: data_case,
    },
    Policy {
        name: "agent_guardrail",
        rule_en: "an agent may read anything, and may change staging, but any change to production needs an approved change ticket",
        rule_fr: "un agent peut tout lire et modifier la préproduction, mais toute modification de la production exige un ticket de changement approuvé",
        gen: agent_case,
    },
];

fn render_state(attrs: &[(String, String)], rng: &mut Rng, lang: &str) -> String {
    if rng.f64() < 0.5 {
        let m: serde_json::Map<String, serde_json::Value> = attrs
            .iter()
            .map(|(k, v)| (k.replace(' ', "_"), serde_json::Value::String(v.clone())))
            .collect();
        serde_json::Value::Object(m).to_string()
    } else {
        let sep = if lang == "fr" { " ; " } else { "; " };
        attrs
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join(sep)
    }
}

/// Policy questions over generated cases, half violating the rule, each with its minimal pair.
pub fn rules(n: usize, lang: &str, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, &format!("s5/gen/rules/{lang}"));
    let mut out = Vec::with_capacity(n);
    let src = format!("gen_rules_{lang}");
    while out.len() < n {
        let p = pick(&mut rng, POLICIES);
        let violate = rng.f64() < 0.5;
        let case = (p.gen)(&mut rng, violate);
        let flipped = case.flipped.as_deref();
        let as_choice = rng.f64() < 0.5;
        let start = out.len();
        for (i, c) in std::iter::once(&case).chain(flipped).enumerate() {
            let a = if lang == "fr" {
                &c.attrs_fr
            } else {
                &c.attrs_en
            };
            let state = render_state(a, &mut rng, lang);
            let uid = uid_of(&[&src, p.name, &state, &i.to_string()]);
            let r = if as_choice {
                let (ins, opts) = if lang == "fr" {
                    (
                        format!("Que faire de cette demande ? Règle : {}.", p.rule_fr),
                        kv(&[
                            (
                                "accepter",
                                "La demande respecte toutes les conditions de la règle.",
                            ),
                            (
                                "refuser",
                                "Au moins une condition de la règle n'est pas remplie.",
                            ),
                        ]),
                    )
                } else {
                    (
                        format!(
                            "What should be done with this request? Rule: {}.",
                            p.rule_en
                        ),
                        kv(&[
                            ("approve", "The request meets every condition of the rule."),
                            ("deny", "At least one condition of the rule is not met."),
                        ]),
                    )
                };
                rec(
                    &src,
                    "choice",
                    ins,
                    opts,
                    state,
                    one_hot(2, usize::from(!c.ok)),
                    lang,
                    uid,
                )
            } else {
                let (ins, opts) = if lang == "fr" {
                    (
                        format!("Cette demande enfreint la règle : {}.", p.rule_fr),
                        kv(&[
                            ("false", "La demande respecte la règle."),
                            ("true", "La demande enfreint la règle."),
                        ]),
                    )
                } else {
                    (
                        format!("This request violates the rule: {}.", p.rule_en),
                        kv(&[
                            ("false", "The request complies with the rule."),
                            ("true", "The request breaks the rule."),
                        ]),
                    )
                };
                rec(
                    &src,
                    "noul",
                    ins,
                    opts,
                    state,
                    one_hot(2, usize::from(!c.ok)),
                    lang,
                    uid,
                )
            };
            out.push(r);
        }
        // A minimal pair stays in one split.
        let split = out[start].split.clone();
        out[start..]
            .iter_mut()
            .for_each(|r| r.split = split.clone());
    }
    out.truncate(n);
    out
}

// ------------------------------------------------------------------------------------ tone

const ISSUES_EN: &[&str] = &[
    "my order arrived two weeks late",
    "I was charged twice for the same subscription",
    "the replacement part does not fit my model",
    "my account was locked after a password reset",
    "the courier left the parcel in the rain",
    "the invoice shows the wrong company name",
];
const ISSUES_FR: &[&str] = &[
    "ma commande est arrivée avec deux semaines de retard",
    "j'ai été prélevé deux fois pour le même abonnement",
    "la pièce de rechange ne correspond pas à mon modèle",
    "mon compte a été bloqué après une réinitialisation du mot de passe",
    "le livreur a laissé le colis sous la pluie",
    "la facture indique le mauvais nom de société",
];

/// The same complaint written calmly (level 0), firmly (1) or angrily (2); the angry one
/// must go to the de-escalation team.
pub fn tone(n: usize, lang: &str, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, &format!("s5/gen/tone/{lang}"));
    let src = format!("gen_tone_{lang}");
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let issue = if lang == "fr" {
            *pick(&mut rng, ISSUES_FR)
        } else {
            *pick(&mut rng, ISSUES_EN)
        };
        let order = rng.below(1_000_000);
        let versions: [String; 3] = if lang == "fr" {
            [
                format!("Bonjour, je vous écris car {issue} (commande {order}). Pourriez-vous regarder quand vous aurez un moment ? Merci beaucoup."),
                format!("Bonjour. {issue} (commande {order}). J'attends une solution rapidement, c'est la deuxième fois."),
                format!("C'EST INADMISSIBLE !!! {issue} (commande {order}) et personne ne répond. Si ce n'est pas réglé aujourd'hui je porte plainte et je vous fais une très mauvaise publicité !"),
            ]
        } else {
            [
                format!("Hello, I am writing because {issue} (order {order}). Could you look into it when you have a moment? Many thanks."),
                format!("Hello. {issue} (order {order}). I expect this to be fixed quickly, this is the second time."),
                format!("THIS IS UNACCEPTABLE!!! {issue} (order {order}) and nobody answers. If this is not fixed today I am filing a complaint and telling everyone how bad you are!"),
            ]
        };
        let as_score = rng.f64() < 0.5;
        let start = out.len();
        for (lvl, msg) in versions.iter().enumerate() {
            let state = format!(
                "{{\"channel\": \"email\", \"message\": {}}}",
                serde_json::Value::String(msg.clone())
            );
            let uid = uid_of(&[&src, &state]);
            let r = if as_score {
                let (ins, levels) = if lang == "fr" {
                    (
                        "À quel point le client est-il en colère ?".to_string(),
                        [
                            "Calme et poli.",
                            "Ferme ou agacé, sans hostilité.",
                            "Furieux, hostile ou menaçant.",
                        ],
                    )
                } else {
                    (
                        "How upset is the customer?".to_string(),
                        [
                            "Calm and polite.",
                            "Firm or annoyed, without hostility.",
                            "Furious, hostile or threatening.",
                        ],
                    )
                };
                let opts = levels
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (i.to_string(), d.to_string()))
                    .collect();
                rec(&src, "score", ins, opts, state, one_hot(3, lvl), lang, uid)
            } else {
                let (ins, opts) = if lang == "fr" {
                    (
                        "Ce message doit être confié à l'équipe de désamorçage des conflits."
                            .to_string(),
                        kv(&[
                            ("false", "Le ton reste gérable par le support habituel."),
                            (
                                "true",
                                "Le client est hostile ou menaçant : il faut désamorcer.",
                            ),
                        ]),
                    )
                } else {
                    (
                        "This message must go to the de-escalation team.".to_string(),
                        kv(&[
                            ("false", "The tone can be handled by regular support."),
                            (
                                "true",
                                "The customer is hostile or threatening and needs de-escalation.",
                            ),
                        ]),
                    )
                };
                rec(
                    &src,
                    "noul",
                    ins,
                    opts,
                    state,
                    one_hot(2, usize::from(lvl == 2)),
                    lang,
                    uid,
                )
            };
            out.push(r);
        }
        let split = out[start].split.clone();
        out[start..]
            .iter_mut()
            .for_each(|r| r.split = split.clone());
    }
    out.truncate(n);
    out
}

// ------------------------------------------------------------------------------------ buried facts

struct Fact {
    ins_en: &'static str,
    ins_fr: &'static str,
    /// (id, definition en, definition fr) per option.
    opts: &'static [(&'static str, &'static str, &'static str)],
    /// The decisive sentence for each option (en, fr).
    facts: &'static [(&'static str, &'static str)],
}

const FACTS: &[Fact] = &[
    Fact {
        ins_en: "What is the delivery status of the parcel?",
        ins_fr: "Quel est l'état de livraison du colis ?",
        opts: &[
            (
                "delivered",
                "The parcel reached the customer.",
                "Le colis a été remis au client.",
            ),
            (
                "failed",
                "A delivery attempt failed.",
                "Une tentative de livraison a échoué.",
            ),
            (
                "in_transit",
                "The parcel is still on its way.",
                "Le colis est toujours en route.",
            ),
        ],
        facts: &[
            (
                "The courier confirmed delivery to the customer at 14:02.",
                "Le livreur a confirmé la remise au client à 14 h 02.",
            ),
            (
                "The courier reported that nobody was home and the delivery failed.",
                "Le livreur a signalé que personne n'était là et que la livraison a échoué.",
            ),
            (
                "The tracking shows the parcel leaving the regional hub this morning.",
                "Le suivi indique que le colis a quitté la plateforme régionale ce matin.",
            ),
        ],
    },
    Fact {
        ins_en: "Has the invoice been paid?",
        ins_fr: "La facture a-t-elle été payée ?",
        opts: &[
            (
                "paid",
                "The payment was received in full.",
                "Le paiement a été reçu en totalité.",
            ),
            (
                "unpaid",
                "No payment was received.",
                "Aucun paiement n'a été reçu.",
            ),
            (
                "partial",
                "Only part of the amount was received.",
                "Seule une partie du montant a été reçue.",
            ),
        ],
        facts: &[
            (
                "Accounting registered the full transfer of the invoice amount on Monday.",
                "La comptabilité a enregistré lundi le virement complet du montant de la facture.",
            ),
            (
                "The bank statement shows no incoming transfer for this invoice.",
                "Le relevé bancaire ne montre aucun virement reçu pour cette facture.",
            ),
            (
                "The customer transferred 40 % of the amount and promised the rest next month.",
                "Le client a viré 40 % du montant et promis le reste le mois prochain.",
            ),
        ],
    },
    Fact {
        ins_en: "Is the server healthy?",
        ins_fr: "Le serveur est-il en bonne santé ?",
        opts: &[
            (
                "healthy",
                "All checks pass.",
                "Tous les contrôles sont au vert.",
            ),
            (
                "degraded",
                "It answers, but slowly or with errors.",
                "Il répond, mais lentement ou avec des erreurs.",
            ),
            (
                "down",
                "It does not answer at all.",
                "Il ne répond plus du tout.",
            ),
        ],
        facts: &[
            (
                "The last health check passed on every endpoint in 40 ms.",
                "Le dernier contrôle de santé a réussi sur tous les points d'accès en 40 ms.",
            ),
            (
                "Half of the requests time out and the error rate is 12 %.",
                "La moitié des requêtes expirent et le taux d'erreur est de 12 %.",
            ),
            (
                "The host stopped answering pings at 03:10 and nothing has responded since.",
                "L'hôte ne répond plus aux pings depuis 3 h 10 et rien n'a répondu depuis.",
            ),
        ],
    },
];

/// A decisive fact among `filler` sentences (or no fact: uniform target).
pub fn buried(n: usize, lang: &str, filler: &[String], seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, &format!("s5/gen/buried/{lang}"));
    let src = format!("gen_buried_{lang}");
    let mut out = Vec::with_capacity(n);
    if filler.is_empty() {
        return out;
    }
    while out.len() < n {
        let f = pick(&mut rng, FACTS);
        let k = f.opts.len();
        let missing = rng.f64() < 0.25;
        let gold = rng.below(k);
        let n_fill = 3 + rng.below(6);
        let mut sents: Vec<String> = (0..n_fill)
            .map(|_| pick(&mut rng, filler).clone())
            .collect();
        if !missing {
            let fact = if lang == "fr" {
                f.facts[gold].1
            } else {
                f.facts[gold].0
            };
            let at = rng.below(sents.len() + 1);
            sents.insert(at, fact.to_string());
        }
        let state = sents.join(" ");
        let opts = f
            .opts
            .iter()
            .map(|(id, en, fr)| {
                (
                    id.to_string(),
                    if lang == "fr" { fr } else { en }.to_string(),
                )
            })
            .collect();
        let target = if missing {
            vec![1.0 / k as f32; k]
        } else {
            one_hot(k, gold)
        };
        let ins = if lang == "fr" { f.ins_fr } else { f.ins_en }.to_string();
        let uid = uid_of(&[&src, &state, &ins]);
        out.push(rec(
            &format!("{src}{}", if missing { "_missing" } else { "" }),
            "choice",
            ins,
            opts,
            state,
            target,
            lang,
            uid,
        ));
    }
    out
}

// ------------------------------------------------------------------------------------ pairs

/// Definition paraphrase templates of training (`{d}`: the definition with a lowercase
/// first letter and no final period). The evaluation paraphrases share none of them.
const PARA_EN: &[&str] = &[
    "Pick this when {d}.",
    "This option fits if {d}.",
    "Applies when {d}.",
    "Choose it in case {d}.",
    "Right answer if {d}.",
];
const PARA_FR: &[&str] = &[
    "À choisir quand {d}.",
    "Cette option convient si {d}.",
    "S'applique lorsque {d}.",
    "Bonne réponse si {d}.",
];

fn soften(def: &str) -> String {
    let d = def.trim().trim_end_matches('.').trim();
    let mut c = d.chars();
    match c.next() {
        Some(f) if c.clone().next().is_some_and(|x| x.is_lowercase()) => {
            f.to_lowercase().collect::<String>() + c.as_str()
        }
        Some(f) => f.to_string() + c.as_str(),
        None => String::new(),
    }
}

/// `(original, paraphrased)` pairs of the `choice` / `noul` questions of `base` whose options
/// all have definitions.
pub fn paraphrase_pairs(base: &[S5Record], n_pairs: usize, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, "s5/gen/paraphrase");
    let mut idx: Vec<usize> = (0..base.len())
        .filter(|&i| {
            let r = &base[i];
            r.t != "score" && r.opts_kv.iter().all(|(_, d)| d.len() >= 8)
        })
        .collect();
    rng.shuffle(&mut idx);
    let mut out = Vec::new();
    for i in idx.into_iter().take(n_pairs) {
        let r = &base[i];
        let t = if r.lang == "fr" {
            *pick(&mut rng, PARA_FR)
        } else {
            *pick(&mut rng, PARA_EN)
        };
        let key = uid_of(&["para", &r.uid]);
        let mut a = r.clone();
        a.uid = format!("{key}#0");
        a.src = format!("{}+pair", r.src);
        let mut b = a.clone();
        b.uid = format!("{key}#1");
        b.src = format!("{}+para", r.src);
        for (_, d) in &mut b.opts_kv {
            *d = t.replace("{d}", &soften(d));
        }
        out.push(a);
        out.push(b);
    }
    out
}

/// `(original, distracted)` pairs: 1 to 3 sentences of a real text corpus inserted at random
/// places of the state.
pub fn distractor_pairs(
    base: &[S5Record],
    distractors: &[String],
    n_pairs: usize,
    seed: u64,
) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, "s5/gen/distractor");
    let mut out = Vec::new();
    if distractors.is_empty() {
        return out;
    }
    let mut idx: Vec<usize> = (0..base.len()).collect();
    rng.shuffle(&mut idx);
    for i in idx.into_iter().take(n_pairs) {
        let r = &base[i];
        let key = uid_of(&["distract", &r.uid]);
        let mut a = r.clone();
        a.uid = format!("{key}#0");
        a.src = format!("{}+pair", r.src);
        let mut b = a.clone();
        b.uid = format!("{key}#1");
        b.src = format!("{}+distract", r.src);
        let mut state = r.state.clone();
        for _ in 0..1 + rng.below(3) {
            let s = pick(&mut rng, distractors);
            // Insert at a sentence or line boundary.
            let cuts: Vec<usize> = state
                .char_indices()
                .filter(|(_, c)| matches!(c, '.' | '\n'))
                .map(|(j, c)| j + c.len_utf8())
                .collect();
            let at = if cuts.is_empty() {
                state.len()
            } else {
                *pick(&mut rng, &cuts)
            };
            state.insert_str(at, &format!(" {s} "));
        }
        b.state = state;
        out.push(a);
        out.push(b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_balanced_and_paired() {
        for lang in ["en", "fr"] {
            let r = rules(2000, lang, 1);
            assert_eq!(r.len(), 2000);
            let viol = r.iter().filter(|x| x.target[1] == 1.0).count();
            assert!((800..1200).contains(&viol), "{lang}: {viol}");
            // Consecutive records are minimal pairs with opposite labels.
            let flipped = r
                .chunks(2)
                .filter(|c| c.len() == 2 && c[0].target != c[1].target)
                .count();
            assert!(flipped > 950, "{lang}: {flipped}");
            for x in &r {
                x.validate().unwrap();
            }
        }
    }

    #[test]
    fn policies_label_by_their_rule() {
        let mut rng = Rng::new(3);
        for p in POLICIES {
            for v in [false, true] {
                for _ in 0..50 {
                    let c = (p.gen)(&mut rng, v);
                    assert_eq!(c.ok, !v, "{}", p.name);
                    assert_ne!(c.flipped.as_ref().unwrap().ok, c.ok, "{}", p.name);
                }
            }
        }
    }

    #[test]
    fn tone_buried_and_pairs() {
        let t = tone(300, "fr", 2);
        assert!(t.iter().all(|r| r.validate().is_ok() && r.lang == "fr"));
        let filler: Vec<String> = (0..20)
            .map(|i| format!("Filler sentence number {i}."))
            .collect();
        let b = buried(400, "en", &filler, 3);
        let missing: Vec<_> = b.iter().filter(|r| r.src.ends_with("_missing")).collect();
        assert!(!missing.is_empty());
        assert!(missing
            .iter()
            .all(|r| r.target.iter().all(|&x| (x - 1.0 / 3.0).abs() < 1e-6)));
        let p = paraphrase_pairs(&b, 10, 4);
        assert_eq!(p.len(), 20);
        assert_eq!(
            p[0].uid.split_once('#').unwrap().0,
            p[1].uid.split_once('#').unwrap().0
        );
        assert_eq!(p[0].target, p[1].target);
        assert_ne!(p[0].opts_kv, p[1].opts_kv);
        assert!(p[1].opts_kv[0].1.ends_with('.'));
        let d = distractor_pairs(&b, &["An unrelated sentence.".to_string()], 5, 5);
        assert_eq!(d.len(), 10);
        assert!(d[1].state.contains("An unrelated sentence."));
        assert_eq!(
            soften("The parcel reached the customer."),
            "the parcel reached the customer"
        );
        assert_eq!(soften("MFA is on."), "MFA is on");
    }
}
