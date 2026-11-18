#!/usr/bin/env python3
"""The data checks of s6-route v5 (docs/s6-route5-prereg.md 4.4, with amendments 1 and 2), no model.

Standard library only; run with `python3 -I`. v4's checks (scripts/s6-route4-checks.py, with the
rules of scripts/s6-route4-blind.py) are imported as they are; v5 adds C6, the rules N1 to N6 (N6: amendment 2),
the restrictions to the exempt and to the drawn options, the tables by level and the distractor
counts. No remedy.

  s6-route5-checks.py --plan <b5>/counts/v5-plan.json --out <c6-plan.json> [--md <c6-plan.md>]
      C6 at plan time (amendment 1, decision 1): the plan's counts by group, kind, split and
      level, before any teacher call. Exit 0 when C6 holds in every judged cell, 2 when it fails:
      v5 stops before any teacher call, and the plan's counts by level are published.

  s6-route5-checks.py --build <b5> --requests <v2>/stage/requests.jsonl --tools <v2>/stage/tools.jsonl \\
      --pools <b5>/stage/pools5.json [--plan-counts <b5>/counts/v5-plan.json] \\
      [--val <b5>/val/v5_tools.jsonl ...] [--fresh <t5>/fresh/t5_tools.jsonl ...] \\
      [--fresh-manifest <t5>/fresh/FRESH5.json] --out <report.json> [--md <table.md>]
      the checks on the built records, in this order: C6 (confirmed with the same bar), then C1 to
      C5 as v4 (amendment 1's tolerances), then C2 and N1 to N6 judged in every judged cell (with
      --build, --plan-counts is required: N6 reads the plan's classes); exit:

  s6-route5-checks.py --compare-keys <keys.jsonl>
      amendment 2, reading 7: the keys `sky-data` computed for the pools' texts (the ignored test
      `real_pools_are_read` writes the file outside the data folder) against this script's
      reading; prints the counts that differ, exit 1 when any does.

  0  every judged check holds (C6, C1, C3, C2, N1 to N6);
  1  C1 or C3 fails: the build has a defect, fixed before anything else (4.4; C1: see 4.3);
  2  C6, a rule of C2 or one of N1 to N6 fails in a judged cell: the build stops and v5 stops
     before training, with the table published; there is no remedy (4.4, 11, amendment 2).

Cells: every (group, kind, split) of `<build>/records/<group>/<split>.jsonl` and each file of
--val, judged from 200 tool-gold slates (v4); the --fresh files (T5, 6.2) are reported and never
judged.

- C6, the cell holds (4.4): in every cell of at least 200 drawn slates (amendment 2, decision 2:
  the removed cells included), at most 0.10 of the drawn slates (a slate with at least one drawn
  distractor, tool-right and none-right alike) were built at L5 or L6; L6 ("outside the cell") is
  reported apart. The same function reads the plan's counts and the built records.
- N1 to N5 (4.4), each with its inverse, on the option texts with D4's score and rank taken off,
  judged with C2's tolerance c + 0.02 + 2 sqrt(c(1 - c)/n):
  N1 the description with the highest share of digit characters;
  N2 the description holding the word of the lowest frequency class, the class floor(log2 df), df
     the number of descriptions of the group's training pool (originals) holding the word (a word
     absent from the pool, or in one description, is class 0); a word is a run of letters,
     lowercased;
  N3 the name first in alphabetical order (case-folded, then by code point);
  N4 the description with the highest share of capitalised words after its first word;
  N5 the description with the highest mean Jaccard overlap of its lowercased word set with the
     other tool options of the slate.
- N6, none from the spread of the options (amendment 2), judged with the same tolerance on the
  families of C3 (per group and split, the renamed slates, and the other slates; each validation
  file), every slate of the family, tool-right and none-right: a slate's spread is 1 minus the
  largest share of its tool options that show one L0 key (the six features of 4.3 on the text as
  shown, the plan's classes of F2 and F3). The rule answers none when the spread is above the
  family's median spread (its inverse: below it); its accuracy is the share of none-right slates
  among the slates it answers none, against the chance pi, the family's share of none-right
  slates, with the tolerance pi + 0.02 + 2 sqrt(pi (1 - pi) / m), m the slates it answers none.
  Judged in a family of at least 200 tool-gold slates. Reported with it: the share of options in
  the modal L0 cell of each removed slate against the slate it comes from (mean, difference and
  its standard error).
- Reported, not judged: C2 and N1 to N5 on two restrictions of each tool-right slate (the right
  option with the exempt options only, and with the drawn options only, where each holds at least
  two options; R7 reads the option's position in the whole slate), and by level; C4 for the right
  option against the exempt and against the drawn options; the share of slates of each level; the
  distinct distractors of each cell and the largest share of one tool among them; the cell
  consistency (with --plan-counts: drawn options of a tool-right slate below L6 whose features, as
  this script reads them, do not match the right option's at the slate's level).
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('route4_checks', os.path.join(HERE, 's6-route4-checks.py'))
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)
blind = c4.blind

C6_BAR = 0.10
MIN_JUDGED = c4.MIN_JUDGED
C2_MARGIN = c4.C2_MARGIN
LEVELS = 7
WORD = re.compile(r'[^\W\d_]+')
DRAWN, EXEMPT, RIGHT = 'drawn', 'exempt', 'right'


# ------------------------------------------------------------------------------------------ C6

def c6_of(counts, judged=None):
    """C6 on one cell's counts ({'slates', 'tool-gold slates', 'drawn slates', 'slates by level',
    'drawn slates by level'}): the share of the drawn slates at L5 or L6 against 0.10. The cell is
    judged from 200 drawn slates (amendment 2, decision 2), unless `judged` says otherwise (the
    fresh files)."""
    d = counts['drawn slates']
    by = counts['drawn slates by level']
    share = (by[5] + by[6]) / d if d else 0.0
    j = d >= MIN_JUDGED if judged is None else judged
    return {'counts': counts, 'judged': j, 'share of the drawn slates at L5 or L6': share,
            'share of the drawn slates at L6 (outside the cell)': by[6] / d if d else 0.0,
            'share of the slates of each level': [x / max(counts['slates'], 1) for x in counts['slates by level']],
            'bar': C6_BAR, 'above the bar': share > C6_BAR, 'fails': j and share > C6_BAR}


def level_counts(recs):
    """The C6 counts of a list of records (fields `cell_level` and `roles`)."""
    c = {'slates': 0, 'tool-gold slates': 0, 'drawn slates': 0, 'slates by level': [0] * LEVELS,
         'drawn slates by level': [0] * LEVELS}
    for r in recs:
        lv = r.get('cell_level')
        if lv is None:
            continue
        c['slates'] += 1
        c['tool-gold slates'] += not c4.is_none(r)
        c['slates by level'][lv] += 1
        if DRAWN in r.get('roles', []):
            c['drawn slates'] += 1
            c['drawn slates by level'][lv] += 1
    return c


# -------------------------------------------------------------------------------- N1 to N5

def alpha_words(d):
    return [w.lower() for w in WORD.findall(d)]


def frequency_classes(tools_path, pools_path):
    """N2: per group, the number of the training pool's descriptions (originals) holding each
    word."""
    if not tools_path or not pools_path:
        return {}
    desc = {t['id']: t['desc'] for t in c4.load(tools_path)}
    with open(pools_path, encoding='utf-8') as f:
        pools = json.load(f)
    out = {}
    for g, ids in pools.items():
        if not isinstance(ids, list):
            continue
        df = collections.Counter()
        for t in ids:
            df.update(set(alpha_words(desc.get(t, ''))))
        out[g] = df
    return out


def word_class(df, w):
    n = df.get(w, 0)
    return int(math.floor(math.log2(n))) if n > 0 else 0


def n_feats(o, df):
    n, d = blind.split_opt(o)
    ws = alpha_words(d)
    rest = d.split()[1:]
    return {'n1 digits': sum(ch.isdigit() for ch in d) / len(d) if d else 0.0,
            'n2 rarest': min((word_class(df, w) for w in ws), default=None),
            'n3 name': (n.casefold(), n),
            'n4 capitals': sum(w[:1].isupper() for w in rest) / len(rest) if rest else 0.0,
            'n5 words': set(ws)}


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def n3_rank(F):
    keys = sorted({x['n3 name'] for x in F})
    at = {k: i for i, k in enumerate(keys)}
    return [-at[x['n3 name']] for x in F]


def n5_centrality(F):
    if len(F) < 2:
        return [0.0] * len(F)
    return [sum(jaccard(x['n5 words'], y['n5 words']) for j, y in enumerate(F) if j != i) / (len(F) - 1)
            for i, x in enumerate(F)]


def n_rules():
    """(label, tag, rule): N1 to N5 with their inverses."""
    big = float('inf')
    R = []

    def both(tag, name, inv, f):
        R.append((name, tag, f))
        R.append((inv, tag, c4.neg(f)))
    both('N1', 'N1 the highest share of digits in the description', 'N1 inverse: the lowest share of digits',
         lambda F: [x['n1 digits'] for x in F])
    both('N2', 'N2 the rarest word (lowest frequency class)', 'N2 inverse: the rarest word most frequent',
         lambda F: [-x['n2 rarest'] if x['n2 rarest'] is not None else -big for x in F])
    both('N3', 'N3 the first name in alphabetical order', 'N3 inverse: the last name', n3_rank)
    both('N4', 'N4 the highest share of capitalised words after the first', 'N4 inverse: the lowest share',
         lambda F: [x['n4 capitals'] for x in F])
    both('N5', 'N5 centrality: the highest mean word overlap with the other options',
         'N5 inverse: the lowest mean overlap', n5_centrality)
    return R


def by_position(rules):
    """v4's rules with R7 reading the option's position in the whole slate (for the restricted
    slates; on a whole slate it is the same rule)."""
    out = []
    for label, tag, f in rules:
        m = re.match(r'R7 (inverse: not )?position (\d+)$', label)
        if tag == 'R7' and m:
            p, inv = int(m.group(2)), bool(m.group(1))
            f = (lambda F, p=p: [-int(x['position'] == p) for x in F]) if inv else \
                (lambda F, p=p: [int(x['position'] == p) for x in F])
        out.append((label, tag, f))
    return out


def features5(r, srcmap, df):
    F = c4.features(r, srcmap)
    for x, o in zip(F, r['opts'][:-1]):
        x.update(n_feats(c4.plain(o), df))
    return F


def judge(rules, F_list, golds, judged):
    """Each rule's accuracy (a tie counts 1/|tie| when the right option is in it), the chance c
    (the mean of 1/options), the tolerance c + 0.02 + 2 sqrt(c (1 - c) / n), and whether it fails
    (only in a judged cell)."""
    n = len(F_list)
    if n == 0:
        return []
    c = sum(1 / len(f) for f in F_list) / n
    tol = c + C2_MARGIN + 2 * math.sqrt(c * (1 - c) / n)
    out = []
    for label, tag, rule in rules:
        acc = 0.0
        for f, g in zip(F_list, golds):
            v = rule(f)
            m = max(v)
            best = [i for i, x in enumerate(v) if x == m]
            if g in best:
                acc += 1 / len(best)
        acc /= n
        out.append({'rule': label, 'tag': tag, 'accuracy': acc, 'chance': c, 'tolerance': tol,
                    'fails': judged and acc > tol, 'above tolerance': acc > tol})
    out.sort(key=lambda x: -x['accuracy'])
    return out


def compact(rows):
    """A reported table: [rule, accuracy, chance, tolerance] per rule, the highest first, and the
    rules above their tolerance."""
    return {'rules': [[x['rule'], round(x['accuracy'], 4), round(x['chance'], 4), round(x['tolerance'], 4)]
                      for x in rows],
            'above tolerance': [x['rule'] for x in rows if x['above tolerance']]}


# ------------------------------------------------------------------------- cell consistency

def key5(o, classes):
    """The key of an option text as this script reads the six features (4.3), with the plan's
    classes of F2 and F3."""
    n, d = blind.split_opt(c4.plain(o))
    f = blind.feats(c4.plain(o), '')

    def cls(table, v):
        return table[v] if v < len(table) else 4
    form = 0 if f['third'] else (1 if f['imperative'] else 2)
    return (f['style'], cls(classes['names'], len(n)), cls(classes['words'], f['words']), f['period'], form,
            min(2, f['schema']))


def matches(a, b, level):
    """docs/s6-route5-prereg.md 4.3: whether key b matches anchor a at level (as
    crates/sky-data/src/route/v4.rs `surface::matches`)."""
    f1, f2, f3, f4, f5, f6 = a
    g1, g2, g3, g4, g5, g6 = b
    if level >= 6:
        return True
    if level == 0:
        return a == b
    if f4 != g4 or abs(f3 - g3) > 1:
        return False
    if level <= 4 and abs(f2 - g2) > 1:
        return False
    if level <= 3 and f1 != g1:
        return False
    if level <= 2 and f5 != g5:
        return False
    if level <= 1 and f6 != g6:
        return False
    return True


def consistency(recs, classes):
    bad = n = 0
    for r in recs:
        lv = r.get('cell_level')
        if lv is None or lv >= 6 or c4.is_none(r):
            continue
        g = c4.gold_of(r)
        kg = key5(r['opts'][g], classes)
        for o, role in zip(r['opts'], r.get('roles', [])):
            if role == DRAWN:
                n += 1
                bad += not matches(kg, key5(o, classes), lv)
    return {'drawn options of tool-right slates below L6': n, 'not matching the right option at the level': bad}


# --------------------------------------------------------------------------- N6 (amendment 2)

def spread(r, classes):
    """1 minus the largest share of a slate's tool options that show one L0 key (None without a
    tool option)."""
    keys = [key5(o, classes) for o in r['opts'][:-1]]
    if not keys:
        return None
    return 1 - max(collections.Counter(keys).values()) / len(keys)


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def n6(recs, classes, judged=True):
    """N6 on a family of slates: none answered when the spread is above the median (and, the
    inverse, below it); the share of none-right slates among those answers, against pi."""
    rows = [(spread(r, classes), c4.is_none(r)) for r in recs]
    rows = [(x, nn) for x, nn in rows if x is not None]
    n = len(rows)
    tool_gold = sum(not nn for _, nn in rows)
    out = {'slates': n, 'tool-gold slates': tool_gold, 'judged': judged and tool_gold >= MIN_JUDGED, 'rules': []}
    if not n:
        return out
    pi = sum(nn for _, nn in rows) / n
    med = median([x for x, _ in rows])
    out.update({'none share (chance)': pi, 'median spread': med})
    for label, pick in (('N6 none when the options spread above the median', lambda x: x > med),
                        ('N6 inverse: none when the options spread below the median', lambda x: x < med)):
        ans = [nn for x, nn in rows if pick(x)]
        m = len(ans)
        acc = sum(ans) / m if m else 0.0
        tol = pi + C2_MARGIN + 2 * math.sqrt(pi * (1 - pi) / m) if m else 1.0
        out['rules'].append({'rule': label, 'tag': 'N6', 'answered none': m, 'accuracy': acc, 'chance': pi,
                             'tolerance': tol, 'above tolerance': acc > tol, 'fails': out['judged'] and acc > tol})
    return out


def modal_share(r, classes):
    s_ = spread(r, classes)
    return None if s_ is None else 1 - s_


def removed_against_source(recs, classes):
    """Reported (amendment 2, decision 1): the share of options in the modal L0 cell of each
    removed slate against the slate it comes from (its field `from`), paired."""
    by = {r['uid']: r for r in recs}
    d = []
    for r in recs:
        if r.get('kind') != 'removed' or not r.get('from'):
            continue
        src = by.get('%s/%s' % (r['request'], r['from']))
        a, b = modal_share(r, classes), modal_share(src, classes) if src else None
        if a is not None and b is not None:
            d.append((a, b))
    n = len(d)
    if not n:
        return {'pairs': 0}
    diffs = [a - b for a, b in d]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    return {'pairs': n, 'removed: share in the modal cell': sum(a for a, _ in d) / n,
            'source: share in the modal cell': sum(b for _, b in d) / n,
            'difference (removed - source)': mean, 'its standard error': sd / math.sqrt(n)}


def compare_keys(path):
    """Amendment 2, reading 7: the keys of sky-data (`real_pools_are_read`) against key5."""
    with open(path, encoding='utf-8') as f:
        head = json.loads(f.readline())
        classes = head['classes of F2 and F3']
        styles = head['styles']
        n, differ, by = 0, 0, collections.Counter()
        tools = set()
        names = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')
        for line in f:
            x = json.loads(line)
            k = x['key']
            rust = (styles[k[0]], k[1], k[2], bool(k[3]), k[4], k[5])
            py = key5(x['option'], classes)
            n += 1
            if rust != py:
                differ += 1
                tools.add(x['id'])
                for nm, a, b in zip(names, rust, py):
                    by[nm] += a != b
    return {'texts': n, 'texts whose keys differ': differ, 'tools whose keys differ': len(tools),
            'by feature': dict(by)}


# ------------------------------------------------------------------------------------ a cell

def restricted(F_list, golds, recs, part):
    """The right option with the options of one role only (`exempt` or `drawn`), where that
    leaves at least two options."""
    sub_F, sub_g = [], []
    for F, g, r in zip(F_list, golds, recs):
        roles = r.get('roles') or []
        keep = [i for i in range(len(F)) if i == g or (i < len(roles) and roles[i] == part)]
        if len(keep) >= 2:
            sub_F.append([F[i] for i in keep])
            sub_g.append(keep.index(g))
    return sub_F, sub_g


SURFACE = ['period', 'third', 'imperative', 'words', 'chars', 'schema', 'digits', 'namelen']


def surface_by_role(F_list, golds, recs):
    """C4 (reported): the means of the surface features for the right option, the exempt options
    and the drawn options."""
    acc = {k: collections.Counter() for k in (RIGHT, EXEMPT, DRAWN)}
    cnt = collections.Counter()
    for F, g, r in zip(F_list, golds, recs):
        roles = r.get('roles') or []
        for i, x in enumerate(F):
            role = RIGHT if i == g else (roles[i] if i < len(roles) else '?')
            if role not in acc:
                continue
            cnt[role] += 1
            for k in SURFACE:
                acc[role][k] += float(x[k])
            acc[role]['style ' + x['style']] += 1
    return {role: {k: v / cnt[role] for k, v in sorted(acc[role].items())} if cnt[role] else {}
            for role in acc} | {'options': dict(cnt)}


def distractors(recs):
    """The distinct tools used as distractors (drawn, and every option that is not right) and the
    largest share of one tool among them."""
    out = {}
    for part in (DRAWN, 'not right'):
        c = collections.Counter()
        for r in recs:
            g = c4.gold_of(r)
            roles = r.get('roles') or []
            for i, t in enumerate(r['tools']):
                if i == g:
                    continue
                if part == DRAWN and (i >= len(roles) or roles[i] != DRAWN):
                    continue
                c[t] += 1
        tot = sum(c.values())
        out[part] = {'options': tot, 'distinct tools': len(c),
                     'largest share of one tool': max(c.values()) / tot if tot else 0.0}
    return out


def cell_report5(name, recs, srcmap, df, classes=None, judged=True):
    out = c4.cell_report(name, recs, srcmap, judged=judged)
    out['C6'] = c6_of(level_counts(recs), judged=(None if judged else False))
    tg = [r for r in recs if not c4.is_none(r)]
    golds = [c4.gold_of(r) for r in tg]
    with_flags = bool(recs) and all(r.get('rewritten') is not None for r in recs)
    F = [features5(r, srcmap, df) for r in tg]
    out['N'] = judge(n_rules(), F, golds, out['judged'])
    allr = by_position(c4.rule_set(with_flags)) + n_rules()
    out['restrictions (reported)'] = {}
    for part in (EXEMPT, DRAWN):
        sf, sg = restricted(F, golds, tg, part)
        out['restrictions (reported)']['the right option with the %s options only' % part] = dict(
            {'slates': len(sf)}, **compact(judge(allr, sf, sg, False)))
    by = collections.defaultdict(list)
    for f, g, r in zip(F, golds, tg):
        by[r.get('cell_level')].append((f, g))
    out['by level (reported)'] = {
        'L%s' % lv: dict({'tool-gold slates': len(v)}, **compact(judge(allr, [x[0] for x in v], [x[1] for x in v], False)))
        for lv, v in sorted(by.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}
    out['C4 by role (reported)'] = surface_by_role(F, golds, tg)
    out['distractors (reported)'] = distractors(recs)
    if classes:
        out['cell consistency (reported)'] = consistency(recs, classes)
    return out


# ------------------------------------------------------------------------------------ verdicts

MEANING = {
    'C6 plan': 'C6 fails at plan time: in at least one judged cell more than 10% of the drawn slates cannot be '
               'matched to their cell beyond L4, even with the intents\' smaller minimum. v5 stops before any teacher '
               'call, and the plan\'s counts by level are published. A coarser cell would need its own '
               'pre-registration (amendment 1, decision 6).',
    'C6': 'C6 fails after the build: the false-negative replacements pushed slates out of their cells. v5 stops '
          'before training, and the counts are published with the teacher\'s replacements (amendment 1, decision 6).',
    'C2': 'C2 fails: the surface cell does not make the slates blind. No model is trained, and the table is '
          'published (11).',
    'N5': 'N5 fails: even with the hard distractors anchored on the request, the right tool is the slate\'s centre '
          'in word overlap. No model is trained, and the table is published (11).',
    'N': 'Another N rule fails: a request-blind cue that the cell does not see remains. No model is trained, and '
         'the table is published, naming the rule and the cells (11).',
    'N6': 'N6 fails: the spread of a slate\'s options over the cells tells whether none is right, without the '
          'request. No model is trained, and the table is published, naming the families; there is no remedy '
          '(amendment 2).',
    'defect': 'C1 or C3 fails: the build has a defect and is fixed before anything else (4.4; C1: 4.3).',
    'pass': 'Every judged check holds: C6, C1, C3, C2 and N1 to N6.',
}


def plan_report(counts_json):
    with open(counts_json, encoding='utf-8') as f:
        p = json.load(f)
    cells = {c['cell']: c6_of(c['counts']) for c in p['C6 cells']}
    fails = [k for k, c in cells.items() if c['fails']]
    v = {'C6 fails': fails, 'exit': 2 if fails else 0,
         'meaning': [MEANING['C6 plan']] if fails else ['C6 holds at plan time in every judged cell: the '
                                                        'false-negative job may run (amendment 1, decision 1).']}
    return {'about': 's6-route v5: C6 at plan time (docs/s6-route5-prereg.md 4.4, amendment 1), no teacher call',
            'plan': counts_json, 'bar': C6_BAR, 'cells judged from (drawn slates; amendment 2)': MIN_JUDGED,
            'cells': cells, 'verdict': v}


def run(build, requests, tools, pools, plan_counts, val, fresh, fresh_manifest):
    if build and not plan_counts:
        raise SystemExit('--build needs --plan-counts: N6 reads the plan\'s classes of F2 and F3 (amendment 2)')
    srcmap = c4.tool_sources(requests)
    dfs = frequency_classes(tools, pools)
    classes = None
    if plan_counts:
        with open(plan_counts, encoding='utf-8') as f:
            classes = json.load(f).get('classes of F2 and F3')
    cells, records = {}, {}
    if build:
        rd = os.path.join(build, 'records')
        for g in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
            for fn in sorted(os.listdir(os.path.join(rd, g))):
                if not fn.endswith('.jsonl'):
                    continue
                split = fn[:-len('.jsonl')]
                by = collections.defaultdict(list)
                for r in c4.load(os.path.join(rd, g, fn)):
                    by[r['kind']].append(r)
                for kind, recs in sorted(by.items()):
                    records[(g, kind, split)] = recs
                    c = cell_report5('%s %s %s' % (g, kind, split), recs, srcmap, dfs.get(g, {}), classes)
                    c['kind'] = 'records'
                    cells['records %s %s %s' % (g, kind, split)] = c
    for p in val or []:
        c = cell_report5(p, c4.load(p), srcmap, dfs.get('tools', {}), classes)
        c['kind'] = 'val'
        cells['val ' + os.path.basename(p)] = c
    fresh_classes = None
    if fresh_manifest and os.path.exists(fresh_manifest):
        with open(fresh_manifest, encoding='utf-8') as f:
            fresh_classes = json.load(f).get('counts', {}).get('classes of F2 and F3')
    fresh_recs = {}
    for p in fresh or []:
        fresh_recs[p] = c4.load(p)
        c = cell_report5(p, fresh_recs[p], srcmap, dfs.get('tools', {}), None, judged=False)
        c['kind'] = 'fresh (reported, never judged)'
        cells['fresh ' + os.path.basename(p)] = c
    # N6 (amendment 2) on the families of C3, and the removed slates against their sources
    families, removed_table = {}, {}
    if classes:
        by_gs = collections.defaultdict(lambda: {'renamed': [], 'other': []})
        for (g, kind, split), recs in records.items():
            by_gs[(g, split)]['renamed' if kind == 'renamed' else 'other'].extend(recs)
        for (g, split), d in sorted(by_gs.items()):
            for part, recs in d.items():
                if recs:
                    families['records %s %s %s' % (g, split, part)] = n6(recs, classes)
            removed_table['%s %s' % (g, split)] = removed_against_source(d['other'], classes)
        for p in val or []:
            families['val ' + os.path.basename(p)] = n6(c4.load(p), classes)
    if fresh_classes:
        for p, recs in fresh_recs.items():
            families['fresh ' + os.path.basename(p)] = n6(recs, fresh_classes, judged=False)
    # C6 first (4.4): whether C2 and N1 to N5 are read on cell-matched slates at all
    c6_fail = [k for k, c in cells.items() if c['C6']['fails']]
    c3 = c4.none_share_checks(cells, records)
    c1_fail = [k for k, c in cells.items() if c.get('C1', {}).get('fails')]
    pk = [0, 0, 0, 0]
    for c in cells.values():
        x = c.get('C1')
        if x and c.get('kind') == 'records':
            pk[0] += x['counts (right rewritten, others rewritten)'][0]
            pk[1] += x['right tools']
            pk[2] += x['counts (right rewritten, others rewritten)'][1]
            pk[3] += x['other tool options']
    c1_pooled = None
    if pk[1]:
        pr, po = pk[0] / pk[1], pk[2] / max(pk[3], 1)
        c1_pooled = {'P(rewritten | right tool)': pr, 'right tools': pk[1],
                     'P(rewritten | other tool option)': po, 'other tool options': pk[3],
                     'bar': c4.C1_POOLED,
                     'fails': abs(pr - 0.5) > c4.C1_POOLED or abs(po - 0.5) > c4.C1_POOLED}
        if c1_pooled['fails']:
            c1_fail.append('pooled')
    c3_fail = [x['cell'] for x in c3 if x['fails']]
    c2_fail = [(k, r['rule'], r['tag']) for k, c in cells.items() for r in c['C2'] if r['fails']]
    n_fail = [(k, r['rule'], r['tag']) for k, c in cells.items() for r in c.get('N', []) if r['fails']]
    n_fail += [(k, r['rule'], r['tag']) for k, f in families.items() for r in f['rules'] if r['fails']]
    c5 = None
    if fresh_manifest and os.path.exists(fresh_manifest):
        with open(fresh_manifest, encoding='utf-8') as f:
            m = json.load(f)
        c5 = {"Toucan tools within v2's cluster rule (never shown)":
              m.get('counts', {}).get("Toucan tools within v2's cluster rule (never shown)"),
              'enforced by': 'sky-data route toucan (stops on a violation)'}
    defect = bool(c1_fail or c3_fail)
    stop = bool(c6_fail or c2_fail or n_fail)
    meaning = []
    if defect:
        meaning.append(MEANING['defect'])
    if c6_fail:
        meaning.append(MEANING['C6'])
    if c2_fail:
        meaning.append(MEANING['C2'])
    if any(t == 'N5' for _, _, t in n_fail):
        meaning.append(MEANING['N5'])
    if any(t == 'N6' for _, _, t in n_fail):
        meaning.append(MEANING['N6'])
    if any(t not in ('N5', 'N6') for _, _, t in n_fail):
        meaning.append(MEANING['N'])
    if not meaning:
        meaning.append(MEANING['pass'])
    verdict = {
        'C6 fails': c6_fail, 'C1 fails': c1_fail, 'C3 fails': c3_fail,
        'C2 fails': [{'cell': k, 'rule': r, 'tag': t} for k, r, t in c2_fail],
        'N fails': [{'cell': k, 'rule': r, 'tag': t} for k, r, t in n_fail],
        'defect (C1 or C3)': defect, 'v5 stops before training (C6, C2 or N; no remedy)': stop,
        'exit': 1 if defect else (2 if stop else 0), 'meaning': meaning,
    }
    return {'about': 's6-route v5: the data checks of docs/s6-route5-prereg.md 4.4 (no model)',
            'rules': {'C6': 'at most %.2f of the drawn slates of a judged cell at L5 or L6' % C6_BAR,
                      'C1': '|rate - 0.5| <= 3 sqrt(0.25 / n) per cell and rate; pooled |rate - 0.5| <= 0.01 (v4, amendment 1)',
                      'C2 and N1 to N6': 'c + 0.02 + 2 sqrt(c(1-c)/n) (N6: c the none share, n the slates answered none)', 'cells judged from (tool-gold slates; C6: drawn slates)': MIN_JUDGED},
            'cells': cells, 'N6 families': families,
            'removed slates against their sources (reported; amendment 2)': removed_table,
            'C1 pooled': c1_pooled, 'C3': c3, 'C5': c5, 'verdict': verdict}


def markdown(rep, top=5):
    v = rep['verdict']
    if 'plan' in rep:
        out = ['# s6-route v5: C6 at plan time (docs/s6-route5-prereg.md 4.4, amendment 1)', '',
               'exit %d; C6 fails in %d judged cells' % (v['exit'], len(v['C6 fails'])), '']
        out += v['meaning'] + ['']
        out += ['| cell | slates | tool-gold | drawn | judged | L5 or L6 | L6 | L0 ... L6 (drawn) | fails |',
                '|---|---:|---:|---:|---|---:|---:|---|---|']
        for k, c in rep['cells'].items():
            n = c['counts']
            out.append('| %s | %d | %d | %d | %s | %.4f | %.4f | %s | %s |' % (
                k, n['slates'], n['tool-gold slates'], n['drawn slates'], 'yes' if c['judged'] else 'no',
                c['share of the drawn slates at L5 or L6'], c['share of the drawn slates at L6 (outside the cell)'],
                ' '.join(str(x) for x in n['drawn slates by level']), c['fails']))
        return '\n'.join(out) + '\n'
    out = ['# s6-route v5: the data checks (docs/s6-route5-prereg.md 4.4)', '']
    out += ['exit %d; C6 fails %d; C1 fails %d; C3 fails %d; C2 fails %d; N fails %d' % (
        v['exit'], len(v['C6 fails']), len(v['C1 fails']), len(v['C3 fails']), len(v['C2 fails']),
        len(v['N fails'])), ''] + v['meaning'] + ['']
    out += ['| cell | slates | tool-gold | judged | C6 L5/L6 | C1 diff | best rule (C2, N) | accuracy | tolerance | failing |',
            '|---|---:|---:|---|---:|---:|---|---:|---:|---:|']
    for k, c in rep['cells'].items():
        rules = sorted(c['C2'] + c.get('N', []), key=lambda x: -x['accuracy'])
        best = rules[0] if rules else None
        c1 = c.get('C1', {}).get('difference (reported)')
        out.append('| %s | %d | %d | %s | %.4f | %s | %s | %s | %s | %d |' % (
            k, c['slates'], c['tool-gold slates'], 'yes' if c['judged'] else 'no',
            c['C6']['share of the drawn slates at L5 or L6'], '%+.4f' % c1 if c1 is not None else '',
            best['rule'] if best else '', '%.4f' % best['accuracy'] if best else '',
            '%.4f' % best['tolerance'] if best else '', sum(r['fails'] for r in rules)))
    out += ['', '## N6 (amendment 2)', '', '| family | slates | none share | median spread | rule | answered none | accuracy | tolerance | fails |',
            '|---|---:|---:|---:|---|---:|---:|---:|---|']
    for k, f in rep.get('N6 families', {}).items():
        for r in f['rules']:
            out.append('| %s | %d | %.4f | %.4f | %s | %d | %.4f | %.4f | %s |' % (
                k, f['slates'], f['none share (chance)'], f['median spread'], r['rule'], r['answered none'],
                r['accuracy'], r['tolerance'], r['fails']))
    out += ['', '## Removed slates against their sources (reported)', '',
            '| group split | pairs | removed: modal share | source: modal share | difference | SE |', '|---|---:|---:|---:|---:|---:|']
    for k, t in rep.get('removed slates against their sources (reported; amendment 2)', {}).items():
        if t.get('pairs'):
            out.append('| %s | %d | %.4f | %.4f | %+.4f | %.4f |' % (
                k, t['pairs'], t['removed: share in the modal cell'], t['source: share in the modal cell'],
                t['difference (removed - source)'], t['its standard error']))
    out += ['', '## C3', '', '| cell | slates | none | share | fails |', '|---|---:|---:|---:|---|']
    out += ['| %s | %d | %d | %.4f | %s |' % (x['cell'], x['slates'], x['none'], x['share'], x['fails'])
            for x in rep['C3']]
    for k, c in rep['cells'].items():
        rules = sorted(c['C2'] + c.get('N', []), key=lambda x: -x['accuracy'])
        out += ['', '### %s: the %d highest rules' % (k, top), '', '| rule | accuracy | chance | tolerance |',
                '|---|---:|---:|---:|']
        out += ['| %s | %.4f | %.4f | %.4f |' % (r['rule'], r['accuracy'], r['chance'], r['tolerance'])
                for r in rules[:top]]
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--plan', help="C6 at plan time: <b5>/counts/v5-plan.json")
    ap.add_argument('--compare-keys', help="amendment 2, reading 7: the keys sky-data wrote for the pools' texts")
    ap.add_argument('--build')
    ap.add_argument('--requests')
    ap.add_argument('--tools', help="N2: v2's stage/tools.jsonl (the original descriptions)")
    ap.add_argument('--pools', help="N2: <b5>/stage/pools5.json (the group's training pool)")
    ap.add_argument('--plan-counts', help="the classes of F2 and F3 (<b5>/counts/v5-plan.json), for the cell consistency")
    ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--fresh', nargs='*', default=[])
    ap.add_argument('--fresh-manifest')
    ap.add_argument('--out')
    ap.add_argument('--md')
    a = ap.parse_args(argv)
    if not a.compare_keys and not a.out:
        ap.error('--out is required')
    if a.compare_keys:
        res = compare_keys(a.compare_keys)
        print(json.dumps(res))
        return 1 if res['texts whose keys differ'] else 0
    if a.plan:
        if a.build or a.val or a.fresh:
            ap.error('--plan reads the plan\'s counts alone')
        rep = plan_report(a.plan)
    else:
        rep = run(a.build, a.requests, a.tools, a.pools, a.plan_counts, a.val, a.fresh, a.fresh_manifest)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(rep, f, indent=1, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    if a.md:
        with open(a.md, 'w') as f:
            f.write(markdown(rep))
    v = rep['verdict']
    print(json.dumps({k: (len(x) if isinstance(x, list) and k != 'meaning' else x) for k, x in v.items()}))
    return v['exit']


if __name__ == '__main__':
    sys.exit(main())
