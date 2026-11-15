//! s6-route-nano (`docs/s6-nano2-prereg.md`, amendment 1): each tool option of a slate is handed
//! its BM25 score against the request and its rank in the slate, computed over the slate's own
//! tool options. The data builder (`sky-data route`) and the engine side (`sky-web`) call this one
//! module; both must reproduce `tests/fixtures/route_bm25.json`.
//!
//! **Where the numbers go.** Not in the option texts: the engine encodes the instruction and the
//! options once per tool list and caches that static stream, and a score that depends on the
//! request would make every option's text, hence its encoding, change with each request. The
//! numbers go into the request-side state, after the request, as one entry per tool option in the
//! slate's order, each entry keyed by the option's name (an option has no position of its own in
//! the graph: it finds its entry by its name):
//!
//! ```text
//! User: turn on the lights in the hall
//! BM25: lights_on score 1.39 rank 1; weather score 0.00 rank 2
//! ```
//!
//! Each number follows a word (`score`, `rank`), so the number extraction of the s2 tokenizers
//! turns it into a `[NUM]` that fuses onto that word: two fused numbers per option. *None*, the
//! last option, gets no entry.
//!
//! **BM25.** The Lucene variant of `bm25s` 0.3.16 with its defaults, as the `bm25` baseline of
//! `scripts/route-baselines.py`: k1 = 1.5, b = 0.75; documents are the slate's tool options
//! (`name: description`), the request is the query; tokens are the runs of two or more word
//! characters of the lower-cased text (scikit-learn's `(?u)\b\w\w+\b`), without the 33 English
//! stopwords of `bm25s`; idf = ln(1 + (N − df + 0.5) / (df + 0.5)); a query token adds
//! idf · tf / (tf + k1 · (1 − b + b · |d| / avgdl)) for each occurrence in the query; query tokens
//! that no document holds add nothing. The score is written with two decimals, the rank is 1 for
//! the highest score (ties: the earlier option first).

/// BM25 parameters (the defaults of `bm25s`).
pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;

/// The English stopwords of `bm25s` (`bm25s.stopwords.STOPWORDS_EN`).
pub const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// The prefix of the line that carries the scores.
pub const LINE_PREFIX: &str = "BM25: ";

fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The tokens of a text: lower-cased runs of at least two word characters, stopwords removed.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !word_char(c))
        .filter(|w| w.chars().count() >= 2 && !STOPWORDS.contains(w))
        .map(str::to_string)
        .collect()
}

/// The BM25 score of each document against the query, over these documents only.
pub fn bm25(query: &str, docs: &[&str]) -> Vec<f64> {
    let toks: Vec<Vec<String>> = docs.iter().map(|d| tokens(d)).collect();
    let n = docs.len() as f64;
    if docs.is_empty() {
        return Vec::new();
    }
    let avg = toks.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let q = tokens(query);
    toks.iter()
        .map(|d| {
            let dl = d.len() as f64;
            q.iter()
                .map(|t| {
                    let df = toks.iter().filter(|o| o.contains(t)).count() as f64;
                    if df == 0.0 {
                        return 0.0;
                    }
                    let tf = d.iter().filter(|x| *x == t).count() as f64;
                    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                    let norm = if avg > 0.0 {
                        K1 * (1.0 - B + B * dl / avg)
                    } else {
                        K1
                    };
                    idf * tf / (tf + norm)
                })
                // from +0.0: an empty sum of f64 is -0.0, which prints as "-0.00"
                .fold(0.0, |a, b| a + b)
        })
        .collect()
}

/// 1-based ranks of the scores, the highest first; of equal scores, the earlier first.
pub fn ranks(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut r = vec![0; scores.len()];
    for (k, &i) in order.iter().enumerate() {
        r[i] = k + 1;
    }
    r
}

/// The name of an option `name: description` (the whole text without `: `).
pub fn option_name(option: &str) -> &str {
    option.split_once(": ").map_or(option, |(n, _)| n)
}

/// The scores line of the tool options of a slate (*none* excluded), or an empty string
/// without tool options.
pub fn scores_line(request: &str, tool_options: &[&str]) -> String {
    if tool_options.is_empty() {
        return String::new();
    }
    let s = bm25(request, tool_options);
    let r = ranks(&s);
    let entries: Vec<String> = tool_options
        .iter()
        .zip(s.iter().zip(&r))
        .map(|(o, (s, r))| format!("{} score {s:.2} rank {r}", option_name(o)))
        .collect();
    format!("{LINE_PREFIX}{}", entries.join("; "))
}

/// The state of a routing decision: `User: <request>`, then the scores line.
pub fn state(request: &str, tool_options: &[&str]) -> String {
    let line = scores_line(request, tool_options);
    let user = format!("User: {}", request.trim());
    if line.is_empty() {
        user
    } else {
        format!("{user}\n{line}")
    }
}

/// [`state`] from a slate's options as the records hold them: the last option is *none* and is
/// left out of the scores.
pub fn state_for_options(request: &str, options: &[String]) -> String {
    let tools: Vec<&str> = options[..options.len().saturating_sub(1)]
        .iter()
        .map(String::as_str)
        .collect();
    state(request, &tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_follow_scikit_learn() {
        assert_eq!(
            tokens("Turn ON the lights, a/b c_d x2 2x é!"),
            vec!["turn", "lights", "c_d", "x2", "2x"]
        );
    }

    #[test]
    fn scores_and_ranks() {
        let docs = [
            "lights_on: Turns the lights on.",
            "weather: Gives the weather.",
        ];
        let s = bm25("turn on the lights", &docs);
        assert!(s[0] > 0.0 && s[1] == 0.0);
        assert_eq!(ranks(&s), vec![1, 2]);
        assert_eq!(ranks(&[0.0, 0.0, 1.0]), vec![2, 3, 1]);
        assert!(bm25("x", &[]).is_empty());
    }

    #[test]
    fn the_state_line() {
        let opts = vec![
            "lights_on: Turns the lights on.".to_string(),
            "weather: Gives the weather.".to_string(),
            "none of these tools fits the request".to_string(),
        ];
        let st = state_for_options(" turn on the lights ", &opts);
        let (user, line) = st.split_once('\n').unwrap();
        assert_eq!(user, "User: turn on the lights");
        assert!(line.starts_with("BM25: lights_on score "));
        assert!(line.ends_with("; weather score 0.00 rank 2"));
        // every number follows a word: two [NUM] per option after extraction
        let ex = crate::num::extract(line);
        assert_eq!(ex.values.len(), 4);
        assert!(!ex.text.contains("[NUM][NUM]"));
        // without a tool option, the request alone
        assert_eq!(
            state_for_options("hi", &["none of these tools fits the request".to_string()]),
            "User: hi"
        );
    }
}
