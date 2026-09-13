"""Phase 8b. Is cosine similarity a candidate generator for the alias merge?

Phase 8 noticed that the embedding puts an entity next to its own epithets
without being told to: Vasudeva beside Madhusudana, Govinda, Achyuta, Janardana.
That was a hand reading of a dozen rows, which is a hypothesis and not a result.
This measures it against the only independent evidence on record,
`build/appellations.csv`: 246 (appellation, candidate_entity) rows read off
apposition in the prose, all left unmerged because none of them resolved.

There is no gold alias mapping, so nothing here is a precision score. What can
be measured is agreement. If the cosine ranking and the apposition evidence
point at the same entity far more often than chance allows, two independent
signals agree and the pair is worth a human ruling. If they do not, Phase 8's
observation was the dozen rows I happened to read.

Nothing is merged. The output is a ranked shortlist for a human, and every row
in it is a hypothesis from a model trained below the Phase 6 entry gate.
"""
import json
import numpy as np
import pandas as pd

KS = (1, 5, 10, 25)


def main():
    idx = json.load(open("out/embedding_index.json"))
    ents = idx["entities"]
    e = np.fromfile("out/entity_emb.bin", dtype="float32").reshape(len(ents), idx["entity_dim"])
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    np.fill_diagonal(sim, -2.0)
    ei = {n: i for i, n in enumerate(ents)}

    rel = pd.read_parquet("build/relations.parquet")
    n_triples = len(rel[["subject", "predicate", "object"]].drop_duplicates())
    deg = pd.concat([rel.subject, rel.object]).value_counts()

    a = pd.read_csv("build/appellations.csv")
    # rank of every entity in the cosine order for each appellation, so a pair's
    # rank is a lookup rather than a scan
    rank = {}
    for n in a.appellation.unique():
        if n in ei:
            order = np.argsort(-sim[ei[n]])
            rank[n] = {int(j): r + 1 for r, j in enumerate(order)}

    rows = []
    for t in a.itertuples():
        if t.appellation not in rank or t.candidate_entity not in ei:
            continue
        j = ei[t.candidate_entity]
        rows.append(dict(appellation=t.appellation, candidate=t.candidate_entity,
                         appos_conf=t.confidence, evidence=t.evidence_count,
                         status=t.status, prior=t.prior_says,
                         cos=float(sim[ei[t.appellation], j]),
                         cos_rank=rank[t.appellation][j],
                         appellation_degree=int(deg.get(t.appellation, 0)),
                         candidate_degree=int(deg.get(t.candidate_entity, 0))))
    d = pd.DataFrame(rows)
    N = len(ents)

    # agreement on the single best candidate per appellation
    best = []
    for n, g in d.groupby("appellation"):
        top_appos = g.sort_values("appos_conf", ascending=False).iloc[0]
        top_cos = g.sort_values("cos_rank").iloc[0]
        best.append(dict(appellation=n, n_candidates=len(g),
                         appos_pick=top_appos.candidate, cos_pick=top_cos.candidate,
                         agree=top_appos.candidate == top_cos.candidate,
                         cos_rank_of_appos_pick=int(top_appos.cos_rank),
                         degree=int(top_appos.appellation_degree)))
    b = pd.DataFrame(best)

    L = ["# Phase 8b. Cosine similarity as an alias candidate generator", "",
         "**Below the Phase 6 entry gate.** The model was trained on %d triples" % n_triples,
         "against a stated floor of 10,000, on instruction. Nothing below is merged and",
         "nothing below is asserted. Every row is a hypothesis for a human ruling.", "",
         "There is no gold alias mapping for this corpus, so none of this is precision.",
         "It is agreement between two independent signals: apposition read off the prose",
         "in Phase 2, and cosine distance in the trained embedding.", "",
         "%d of the %d rows in `build/appellations.csv` have both names embedded and are"
         % (len(d), len(a)),
         "scored here. %d appellations, %d entities in the space, so a random guess puts"
         % (d.appellation.nunique(), N),
         "a candidate at rank %.0f on average and inside the top 10 %.1f%% of the time."
         % ((N + 1) / 2, 1000.0 / N), "",
         "## Does cosine find the apposition candidate", "",
         "| k | apposition pairs with the candidate in the cosine top k | share | chance |",
         "|---|---|---|---|"]
    for k in KS:
        h = (d.cos_rank <= k).sum()
        L.append("| %d | %d of %d | %.1f%% | %.1f%% |"
                 % (k, h, len(d), 100.0 * h / len(d), 100.0 * k / N))
    L += ["",
          "Median cosine rank of an apposition-evidenced candidate: %.0f of %d."
          % (d.cos_rank.median(), N),
          "Mean %.0f. On the %d pairs with two or more apposition sightings the median is"
          % (d.cos_rank.mean(), (d.evidence >= 2).sum()),
          "%.0f." % d[d.evidence >= 2].cos_rank.median(), "",
          "## Do the two signals pick the same entity", "",
          "For each appellation, the candidate apposition ranks first against the",
          "candidate cosine ranks first.", "",
          "| appellations | both signals pick the same entity | share |", "|---|---|---|",
          "| %d | %d | %.1f%% |" % (len(b), b.agree.sum(), 100.0 * b.agree.mean()),
          "",
          "Restricted to appellations with more than one candidate, where the pick is a",
          "real choice: %d of %d, %.1f%%."
          % (b[b.n_candidates > 1].agree.sum(), (b.n_candidates > 1).sum(),
             100.0 * b[b.n_candidates > 1].agree.mean()), "",
          "## Shortlist for a human ruling", "",
          "Pairs where apposition and cosine agree, the appellation has at least two",
          "asserted edges, and the candidate is inside the cosine top 10. Sorted by",
          "cosine. `predicted`, `uncited`, not merged.", ""]
    sl = d[(d.cos_rank <= 10) & (d.appellation_degree >= 2)].sort_values("cos")
    sl = sl[sl.apply(lambda r: b.set_index("appellation").loc[r.appellation, "agree"]
                     if r.appellation in set(b.appellation) else False, axis=1)]
    sl = sl.sort_values("cos", ascending=False)
    L += ["| appellation | candidate | cosine | cosine rank | apposition sightings | Phase 2 status | canonical_id prior | citation |",
          "|---|---|---|---|---|---|---|---|"]
    fc = a.set_index(["appellation", "candidate_entity"]).first_citation
    for r in sl.itertuples():
        c = fc.get((r.appellation, r.candidate), None)
        # A prior_only row has no apposition span behind it, so it has no address.
        # It stays on the list and stays visibly uncited rather than borrowing one.
        L.append("| %s | %s | %.3f | %d | %d | %s | %s | %s |"
                 % (r.appellation, r.candidate, r.cos, r.cos_rank, r.evidence,
                    r.status, r.prior if isinstance(r.prior, str) else "none",
                    c if isinstance(c, str) else "**uncited**"))
    # The ruling is a reading, so the passage is printed here rather than left
    # as an address to go and find. Verbatim from the master CSV through Phase 2.
    ex = a.set_index(["appellation", "candidate_entity"]).evidence_sentence
    L += ["", "### The passages, in the same order", ""]
    for r in sl.itertuples():
        s = ex.get((r.appellation, r.candidate), None)
        L += ["**%s &rarr; %s**, %s" % (r.appellation, r.candidate,
                                        fc.get((r.appellation, r.candidate)) or "**uncited**"),
              "" if not isinstance(s, str) else "> " + s.strip(),
              "" if isinstance(s, str) else
              "> No apposition span behind this pair. The prior claims it and the text "
              "never writes the two names together.", ""]
    dup = sl.candidate.value_counts()
    dup = dup[dup > 1]
    if len(dup):
        L += ["", "Conflation risk on this list. %d candidate%s receive more than one "
              % (len(dup), "" if len(dup) == 1 else "s"),
              "appellation: " + ", ".join("%s (%d)" % (k, v) for k, v in dup.items()) + ".",
              "That is the shape of the failure Phase 2 measured, a patronymic epithet",
              "shared by a lineage collapsing onto one bearer. Bhargava is every",
              "descendant of Bhrigu, not one man. Read the passages before merging any of",
              "these, and expect some of them to stay unmerged."]
    L += ["", "%d pairs. " % len(sl) +
          "Every one of them still needs a reading of the cited passage: two",
          "weak signals agreeing is a reason to look, not a resolution. The hard rule",
          "stands, a genuinely ambiguous referent keeps its candidate set.", ""]
    open("build/alias_candidates.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    d.to_csv("build/alias_candidates.csv", index=False)
    print("scored %d pairs / %d appellations; top10 %.1f%% vs chance %.1f%%; "
          "same pick %.1f%%; shortlist %d"
          % (len(d), d.appellation.nunique(), 100.0 * (d.cos_rank <= 10).mean(),
             1000.0 / N, 100.0 * b.agree.mean(), len(sl)))


if __name__ == "__main__":
    main()
