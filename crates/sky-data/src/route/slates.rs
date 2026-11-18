//! Option lists (section 4.4). Tools are indices into the build's tool table; a slate is a list
//! of tools shown in order, with *none* always added last when it is rendered.

use std::collections::HashSet;

use crate::rng::Rng;

use super::NONE_OPTION;

/// Most tools a test slate shows (8 options with *none*).
pub const TEST_TOOLS: usize = 7;

/// A slate: the tools in the order shown, the position of the gold among them (`None`: the
/// gold is *none*), and the text of each option when it is not the tool's own (the renamed
/// gold, the swapped descriptions, the masked names).
#[derive(Clone, Debug, PartialEq)]
pub struct Slate {
    pub tools: Vec<usize>,
    pub gold: Option<usize>,
    pub texts: Vec<Option<String>>,
}

impl Slate {
    pub fn new(tools: Vec<usize>, gold_tool: Option<usize>) -> Self {
        let gold = gold_tool.and_then(|g| tools.iter().position(|&t| t == g));
        let texts = vec![None; tools.len()];
        Self { tools, gold, texts }
    }

    /// The options (the tools' texts, then *none*) and the one-hot target.
    pub fn render(&self, text: impl Fn(usize) -> String) -> (Vec<String>, Vec<f32>) {
        let mut opts: Vec<String> = self
            .tools
            .iter()
            .zip(&self.texts)
            .map(|(&t, o)| o.clone().unwrap_or_else(|| text(t)))
            .collect();
        opts.push(NONE_OPTION.to_string());
        let mut target = vec![0.0; opts.len()];
        target[self.gold.unwrap_or(opts.len() - 1)] = 1.0;
        (opts, target)
    }

    pub fn gold_tool(&self) -> Option<usize> {
        self.gold.map(|g| self.tools[g])
    }

    /// The slate without its gold (*none* becomes the gold).
    pub fn without_gold(&self) -> Self {
        let mut s = self.clone();
        if let Some(g) = s.gold.take() {
            s.tools.remove(g);
            s.texts.remove(g);
        }
        s
    }
}

/// The number of options of a training slate, *none* included: 2 to 8, uniform (4.4).
pub fn draw_options(rng: &mut Rng) -> usize {
    2 + rng.below(7)
}

/// Adds random tools of `pool` that `ok` accepts and that are not in `s` until it holds `want`
/// or the pool has no more.
pub fn fill(
    s: &mut Vec<usize>,
    want: usize,
    pool: &[usize],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) {
    if pool.is_empty() {
        return;
    }
    let mut tries = 0;
    while s.len() < want && tries < 64 * want.max(1) {
        tries += 1;
        let t = pool[rng.below(pool.len())];
        if !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    if s.len() < want {
        // a small or crowded pool: a scan in a random order
        let mut rest: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|t| !s.contains(t) && ok(*t))
            .collect();
        rest.sort_unstable();
        rest.dedup();
        rng.shuffle(&mut rest);
        for t in rest {
            if s.len() >= want {
                break;
            }
            s.push(t);
        }
    }
}

/// The tools of a slate around a gold: the gold, then `first` (the conversation's own tools, or
/// the nearest ones) that `ok` accepts, in order, then random tools of `pool`, to `n` tools;
/// shuffled.
pub fn around(
    gold: Option<usize>,
    first: &[usize],
    n: usize,
    pool: &[usize],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> Vec<usize> {
    let mut s: Vec<usize> = gold.into_iter().collect();
    for &t in first {
        if s.len() >= n {
            break;
        }
        if Some(t) != gold && !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    fill(&mut s, n, pool, rng, ok);
    s.truncate(n.max(gold.is_some() as usize));
    rng.shuffle(&mut s);
    s
}

/// A test-like slate (4.4): the conversation's own tools, then random tools of the split, to
/// [`TEST_TOOLS`]; a removed slate drops the gold and takes one more distractor by the same rule.
pub fn test_slate(
    gold: Option<usize>,
    own: &[usize],
    pool: &[usize],
    removed: bool,
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> Slate {
    let mut s: Vec<usize> = gold.into_iter().collect();
    for &t in own {
        if s.len() >= TEST_TOOLS {
            break;
        }
        if Some(t) != gold && !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    fill(&mut s, TEST_TOOLS, pool, rng, ok);
    if removed {
        if let Some(g) = gold {
            s.retain(|&t| t != g);
            fill(&mut s, TEST_TOOLS, pool, rng, &|t| t != g && ok(t));
        }
    }
    rng.shuffle(&mut s);
    Slate::new(s, if removed { None } else { gold })
}

/// s6-route v4 (D7, `docs/s6-route4-prereg.md` 4.3): [`fill`] from each layer in turn (the
/// tools of the right tool's own source first, then the whole pool; the remedy of 4.4 puts the
/// surface cells first), each layer until `s` holds `want` or the layer has no more.
pub fn fill_layers(
    s: &mut Vec<usize>,
    want: usize,
    layers: &[&[usize]],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) {
    for layer in layers {
        if s.len() >= want {
            break;
        }
        fill(s, want, layer, rng, ok);
    }
}

/// [`around`] with the random fill of [`fill_layers`].
pub fn around_layers(
    gold: Option<usize>,
    first: &[usize],
    n: usize,
    layers: &[&[usize]],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> Vec<usize> {
    let mut s: Vec<usize> = gold.into_iter().collect();
    for &t in first {
        if s.len() >= n {
            break;
        }
        if Some(t) != gold && !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    fill_layers(&mut s, n, layers, rng, ok);
    s.truncate(n.max(gold.is_some() as usize));
    rng.shuffle(&mut s);
    s
}

/// [`test_slate`] with the random fill of [`fill_layers`]; two calls on clones of one stream
/// give the same tools before the gold is removed.
pub fn test_slate_layers(
    gold: Option<usize>,
    own: &[usize],
    layers: &[&[usize]],
    removed: bool,
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> Slate {
    let mut s: Vec<usize> = gold.into_iter().collect();
    for &t in own {
        if s.len() >= TEST_TOOLS {
            break;
        }
        if Some(t) != gold && !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    fill_layers(&mut s, TEST_TOOLS, layers, rng, ok);
    if removed {
        if let Some(g) = gold {
            s.retain(|&t| t != g);
            fill_layers(&mut s, TEST_TOOLS, layers, rng, &|t| t != g && ok(t));
        }
    }
    rng.shuffle(&mut s);
    Slate::new(s, if removed { None } else { gold })
}

/// The same slate with every tool's name replaced by `tool_<k>` (distinct k of 1..=99) and the
/// description kept (7.3).
pub fn masked(s: &Slate, desc: impl Fn(usize) -> String, rng: &mut Rng) -> Slate {
    let mut ks: Vec<usize> = (1..=99).collect();
    rng.shuffle(&mut ks);
    let mut m = s.clone();
    for (i, &t) in s.tools.iter().enumerate() {
        m.texts[i] = Some(super::option_text(&format!("tool_{}", ks[i]), &desc(t)));
    }
    m
}

/// The same slate with the descriptions of the gold and of one random distractor swapped
/// (7.3); returns it and the position of the distractor, whose option now carries the gold's
/// description. `None` without a gold or a distractor.
pub fn swapped(
    s: &Slate,
    name: impl Fn(usize) -> String,
    desc: impl Fn(usize) -> String,
    rng: &mut Rng,
) -> Option<(Slate, usize)> {
    let g = s.gold?;
    let others: Vec<usize> = (0..s.tools.len()).filter(|&i| i != g).collect();
    if others.is_empty() {
        return None;
    }
    let d = others[rng.below(others.len())];
    let mut w = s.clone();
    w.texts[g] = Some(super::option_text(&name(s.tools[g]), &desc(s.tools[d])));
    w.texts[d] = Some(super::option_text(&name(s.tools[d]), &desc(s.tools[g])));
    Some((w, d))
}

/// Indices of `n` items of `0..len` drawn without replacement, in the drawn order.
pub fn draw(len: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..len).collect();
    rng.shuffle(&mut v);
    v.truncate(n);
    v
}

/// The `k` of `items` (by index) chosen to be removed slates: `round(share * len)` of them.
pub fn removed_set(len: usize, share: f64, rng: &mut Rng) -> HashSet<usize> {
    draw(len, (share * len as f64).round() as usize, rng)
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slates_render_with_none_last() {
        let s = Slate::new(vec![4, 2, 9], Some(2));
        let (o, t) = s.render(|i| format!("t{i}"));
        assert_eq!(o, vec!["t4", "t2", "t9", NONE_OPTION]);
        assert_eq!(t, vec![0.0, 1.0, 0.0, 0.0]);
        let r = s.without_gold();
        let (o, t) = r.render(|i| format!("t{i}"));
        assert_eq!(o, vec!["t4", "t9", NONE_OPTION]);
        assert_eq!(t, vec![0.0, 0.0, 1.0]);
        assert_eq!(s.gold_tool(), Some(2));
    }

    #[test]
    fn option_counts_are_two_to_eight() {
        let mut rng = Rng::new(1);
        let mut seen = HashSet::new();
        for _ in 0..500 {
            seen.insert(draw_options(&mut rng));
        }
        assert_eq!(seen, (2..=8).collect());
    }

    #[test]
    fn slates_around_a_gold() {
        let pool: Vec<usize> = (0..50).collect();
        let mut rng = Rng::new(3);
        let s = around(Some(7), &[3, 7, 4, 5], 4, &pool, &mut rng, &|t| t != 5);
        assert_eq!(s.len(), 4);
        for t in [7, 3, 4] {
            assert!(s.contains(&t));
        }
        assert!(!s.contains(&5));
        // a two-option slate is the gold alone
        let s = around(Some(7), &[3], 1, &pool, &mut rng, &|_| true);
        assert_eq!(s, vec![7]);
    }

    #[test]
    fn test_slates_hold_seven_tools() {
        let pool: Vec<usize> = (0..100).collect();
        let mut rng = Rng::new(5);
        let s = test_slate(Some(10), &[11, 12], &pool, false, &mut rng, &|t| t != 13);
        assert_eq!(s.tools.len(), TEST_TOOLS);
        assert_eq!(s.gold_tool(), Some(10));
        assert!(s.tools.contains(&11) && !s.tools.contains(&13));
        let r = test_slate(Some(10), &[11, 12], &pool, true, &mut rng, &|t| t != 13);
        assert_eq!(r.tools.len(), TEST_TOOLS);
        assert_eq!(r.gold, None);
        assert!(!r.tools.contains(&10));
    }

    #[test]
    fn reading_check_slates() {
        let s = Slate::new(vec![1, 2, 3], Some(2));
        let mut rng = Rng::new(9);
        let m = masked(&s, |t| format!("d{t}"), &mut rng);
        let (o, t) = m.render(|_| unreachable!());
        assert!(o[..3].iter().all(|x| x.starts_with("tool_")));
        assert!(o[1].ends_with(": d2"));
        assert_eq!(t, vec![0.0, 1.0, 0.0, 0.0]);
        let (w, d) = swapped(&s, |t| format!("n{t}"), |t| format!("d{t}"), &mut rng).unwrap();
        let (o, _) = w.render(|t| format!("n{t}: d{t}"));
        assert_eq!(o[1], format!("n2: d{}", s.tools[d]));
        assert_eq!(o[d], "n".to_string() + &s.tools[d].to_string() + ": d2");
        assert!(swapped(
            &Slate::new(vec![1], None),
            |_| String::new(),
            |_| String::new(),
            &mut rng
        )
        .is_none());
    }

    #[test]
    fn layered_fills_draw_the_first_layer_first() {
        let pool: Vec<usize> = (0..100).collect();
        let src: Vec<usize> = (10..14).collect();
        let mut rng = Rng::new(4);
        // the source holds enough: every random tool is of it
        let s = around_layers(Some(10), &[], 4, &[&src, &pool], &mut rng, &|_| true);
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|t| src.contains(t)));
        // too few: the whole source, then the pool
        let s = around_layers(Some(10), &[], 7, &[&src, &pool], &mut rng, &|_| true);
        assert_eq!(s.len(), 7);
        assert!(src.iter().all(|t| s.contains(t)));
        // the test-like slate and its removed twin share the tools before the removal
        let ok = |t: usize| t != 12;
        let a = test_slate_layers(Some(10), &[], &[&src, &pool], false, &mut Rng::new(8), &ok);
        let b = test_slate_layers(Some(10), &[], &[&src, &pool], true, &mut Rng::new(8), &ok);
        assert_eq!((a.tools.len(), b.tools.len()), (TEST_TOOLS, TEST_TOOLS));
        assert_eq!(a.gold_tool(), Some(10));
        assert_eq!(b.gold, None);
        assert!(!a.tools.contains(&12) && !b.tools.contains(&10));
        let shared = a.tools.iter().filter(|t| b.tools.contains(t)).count();
        assert_eq!(shared, TEST_TOOLS - 1);
        // one layer that is the pool: the v2 rule's tools
        let mut r1 = Rng::new(3);
        let mut r2 = Rng::new(3);
        assert_eq!(
            test_slate(Some(5), &[6], &pool, true, &mut r1, &ok),
            test_slate_layers(Some(5), &[6], &[&pool], true, &mut r2, &ok)
        );
    }

    #[test]
    fn small_pools_are_scanned() {
        let mut rng = Rng::new(2);
        let mut s = vec![0];
        fill(&mut s, 5, &[0, 1, 2], &mut rng, &|_| true);
        assert_eq!(s.len(), 3);
    }
}
