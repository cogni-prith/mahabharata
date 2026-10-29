"""Bundle the curated lineage table for the explorer's Lineage mode.

annotations/lineage/mahabharata_lineage.csv is hand-authored and carries no
citation. It is not extracted from the corpus, so it never enters the asserted
graph and never reaches training. What this script adds is a check against the
text: each parent edge is looked up among the asserted bornTo and
hasDivineProgenitor triples, and where the text states it the edge carries
that citation. Everything else stays labelled curated and uncited.

The match is on the exact name. "Pandu, Arjuna" is not corroborated although
the graph records Dhananjaya bornTo Pandu, because Dhananjaya is an unresolved
appellation and following it would be guessing an entity link.

Out: out/lineage-data.js, build/lineage_report.md
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'annotations/lineage'
KIN = ['bornTo', 'hasDivineProgenitor']


def cite(row):
    if not row.cited or pd.isna(row.adhyaya_ordinal):
        return 'uncited'
    return '%s %d' % (row.parva, int(row.adhyaya_ordinal))


def kin_facts(names):
    """Asserted kin triples touching a lineage name, grouped with support."""
    rel = pd.read_parquet(ROOT / 'build/reviewed_relations.parquet')
    rel = rel[(rel.status == 'asserted') & rel.predicate.isin(KIN)]
    rel = rel[rel.subject.isin(names) | rel.object.isin(names)]
    facts = {}
    for key, g in rel.groupby(['subject', 'predicate', 'object']):
        places = g.drop_duplicates(['parva', 'adhyaya_ordinal', 'sent_id'])
        cites = sorted({cite(r) for r in places.itertuples()}, key=lambda c: c == 'uncited')
        facts[key] = {'child': key[0], 'predicate': key[1], 'parent': key[2],
                      'support': len(places), 'citations': cites[:3]}
    return facts


CHECKED_ROLE = {'father': 'father', 'mother': 'mother', 'parent': 'parent',
                'adoptive_father': 'adoptive father', 'adoptive_mother': 'adoptive mother',
                'divine_father': 'divine father'}


def checked_facts(names):
    """Phase 4k parent relations a model read and a second model confirmed
    (build/kinship_direct_candidates.parquet, 90% right on a fresh hand-judged
    50), plus judge-undecided ones a person then marked correct. Same exact-name
    rule as the text facts. Kept apart from both the table and the text: the
    page shows them as their own layer, which the reader can switch off."""
    frames = []
    if (ROOT / 'build/kinship_direct_candidates.parquet').exists():
        frames.append(pd.read_parquet(ROOT / 'build/kinship_direct_candidates.parquet'))
    rescued = ROOT / 'annotations/kinship_review_judged.csv'
    if rescued.exists():
        r = pd.read_csv(rescued)
        frames.append(r[r.verdict == 'correct'])
    if not frames:
        return {}
    rel = pd.concat(frames, ignore_index=True)
    # "b is the father of a": b is the parent, a the child.
    rel = rel[rel.type.isin(list(CHECKED_ROLE)) & (rel.a.isin(names) | rel.b.isin(names)) & (rel.a != rel.b)]
    facts = {}
    for (child, parent), g in rel.groupby(['a', 'b']):
        cites = sorted({cite(r) for r in g.itertuples()}, key=lambda c: c == 'uncited')
        roles = sorted({CHECKED_ROLE[t] for t in g.type})
        facts[(child, parent)] = {'child': child, 'parent': parent, 'role': '/'.join(roles),
                                  'support': len(g), 'citations': cites[:3]}
    return facts


def generations(parents, names):
    """Longest path from a root, so a child always sits below every parent."""
    depth, seen = {}, set()

    def visit(n, stack):
        if n in depth:
            return depth[n]
        if n in stack:
            raise ValueError('lineage cycle through %s' % n)
        stack.add(n)
        depth[n] = 1 + max((visit(p, stack) for p in parents[n]), default=-1)
        stack.discard(n)
        return depth[n]

    for n in names:
        visit(n, seen)
    # Longest path alone leaves Satyavati at generation 0 and Vyasa at 1, with
    # edges dropping forty rows to their children. Three rules, repeated until
    # nothing moves: a childless person sits level with their deepest sibling,
    # a parent sits just above their highest child, a child sits below every
    # parent. Each only ever moves someone down, so the loop settles. Applying
    # the sibling rule to parents too does not settle: the table puts Vyasa and
    # Vichitravirya both over Pandu, a generation apart, and the rules chase
    # each other to thousands of rows. Families joined only by a marriage
    # (Drupada's line to the Pandavas) have no shared anchor and keep long edges.
    kids = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            kids[p].append(c)
    for _ in range(len(names)):
        before = dict(depth)
        for n in names:
            if not kids[n] and parents[n]:
                depth[n] = max([depth[n]] + [depth[s] for p in parents[n] for s in kids[p]])
        for n in sorted(names, key=lambda n: -depth[n]):
            if kids[n]:
                depth[n] = max(depth[n], min(depth[c] for c in kids[n]) - 1)
        for n in sorted(names, key=lambda n: depth[n]):
            if parents[n]:
                depth[n] = max(depth[n], max(depth[p] for p in parents[n]) + 1)
        if depth == before:
            break
    low = min(depth.values())
    depth = {n: d - low for n, d in depth.items()}
    return depth


def main():
    raw = pd.read_csv(SRC / 'mahabharata_lineage.csv')
    config = json.loads((SRC / 'mahabharata_config.json').read_text())
    rows = raw.dropna().astype(str).apply(lambda c: c.str.strip())
    dupes = rows[rows.duplicated()]
    selfs = rows[rows.Parent == rows.Child]
    rows = rows.drop_duplicates()
    rows = rows[rows.Parent != rows.Child]

    names = sorted(set(rows.Parent) | set(rows.Child))
    parents = defaultdict(list)
    for p, c in zip(rows.Parent, rows.Child):
        parents[c].append(p)
    depth = generations(parents, names)
    assert all(depth[p] < depth[c] for c, ps in parents.items() for p in ps), 'a parent drawn below a child'

    clan, clashes = {}, []
    for group, members in config['clans'].items():
        for m in members:
            if m in clan and clan[m] != group:
                clashes.append((m, clan[m], group))
            clan.setdefault(m, group)
    absent = sorted(m for m in clan if m not in {n.lower() for n in names})

    facts = kin_facts(names)
    checked = checked_facts(names)
    edges = []
    for p, c in zip(rows.Parent, rows.Child):
        hit = [f for (s, _, o), f in facts.items() if s == c and o == p]
        edges.append({'parent': p, 'child': c, 'text': hit, 'checked': [checked[(c, p)]] if (c, p) in checked else []})
    lineage_pairs = {(e['child'], e['parent']) for e in edges}
    graph_only = [f for (s, _, o), f in facts.items() if (s, o) not in lineage_pairs]
    text_pairs = lineage_pairs | {(f['child'], f['parent']) for f in graph_only}
    checked_only = [f for k, f in checked.items() if k not in text_pairs]

    data = {
        'source': 'annotations/lineage/mahabharata_lineage.csv',
        'nodes': [{'id': n, 'generation': depth[n], 'clan': clan.get(n.lower(), 'other')} for n in names],
        'edges': edges,
        'graph_only': graph_only,
        'checked_only': checked_only,
        'colors': config['colors'],
        'initial': [n for n in config['initial_nodes'] if n in names],
    }
    out = ROOT / 'out/lineage-data.js'
    out.write_text('/* Generated by src/lineage_export.py. Do not edit. */\nwindow.LINEAGE=' +
                   json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n')

    crowded = {c: ps for c, ps in parents.items() if len(ps) > 2}
    corroborated = sum(bool(e['text']) for e in edges)
    lines = [
        '# Lineage export run report', '',
        'Input: `%s`, %d rows.' % (data['source'], len(raw)),
        'Output: `out/lineage-data.js`, %d people, %d parent edges, %d generations.' % (
            len(names), len(edges), max(depth.values()) + 1), '',
        '## Rejected rows', '',
        '- exact duplicate rows dropped: %d %s' % (len(dupes), ', '.join('%s > %s' % t for t in zip(dupes.Parent, dupes.Child))),
        '- self-parent rows dropped: %d' % len(selfs), '',
        '## Against the text', '',
        'An edge is corroborated when the asserted graph holds `(child, bornTo | hasDivineProgenitor, parent)` '
        'under the same exact names. No appellation or spelling variant is followed.', '',
        '- corroborated: %d of %d (%.1f%%)' % (corroborated, len(edges), 100 * corroborated / len(edges)),
        '- curated only, no supporting triple: %d' % (len(edges) - corroborated),
        '- asserted kin triples on lineage names that the table does not hold: %d' % len(graph_only),
        '- table edges a model-read, judge-checked relation also states: %d' % sum(bool(e['checked']) for e in edges),
        '- model-read, judge-checked parent links in neither the table nor the text: %d' % len(checked_only), '',
        '## Needs a human', '',
        'The table has one Parent column, so biological, legal and divine parents are not told apart. '
        'Children with more than two listed parents:', '',
    ] + ['- %s: %s' % (c, ', '.join(ps)) for c, ps in sorted(crowded.items())] + [
        '', 'Clan conflicts in the config, first listing kept: ' +
        (', '.join('%s (%s, %s)' % t for t in clashes) or 'none'),
        '', 'Config clan names with no node in the table: ' + (', '.join(absent) or 'none'), '']
    (ROOT / 'build/lineage_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
