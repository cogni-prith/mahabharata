"""Phase 8. Where the embedding disagrees with the graph.

For every entity, the top-k neighbours in the asserted graph and the top-k
neighbours by cosine in the embedding, and the overlap between the two sets.
Ranked by lowest overlap. Graph neighbours are ranked by how many asserted
triples join the pair, which is the only weight the graph carries.

Low degree is the obvious confound: an entity with three edges has few ways to
agree with fifteen cosine neighbours. Degree is reported on every row and the
ranked table is cut at degree >= k so the head of the list is not just the
sparse tail of the graph. The unrestricted list is written underneath.

Below the Phase 6 entry gate. The overlaps measure a model fitted to a few
triples, so a low overlap here is a lead to check against the text, not a
finding about the epic.
"""
import collections, json
import numpy as np
import pandas as pd

K = 10


def main():
    idx = json.load(open("out/embedding_index.json"))
    ents = idx["entities"]
    e = np.fromfile("out/entity_emb.bin", dtype="float32").reshape(len(ents), idx["entity_dim"])
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    np.fill_diagonal(sim, -2.0)

    ei = {n: i for i, n in enumerate(ents)}
    w = collections.Counter()
    rel = pd.read_parquet("build/relations.parquet")
    n_triples = len(rel[["subject", "predicate", "object"]].drop_duplicates())
    for t in rel.itertuples():
        if t.subject in ei and t.object in ei and t.subject != t.object:
            a, b = sorted((t.subject, t.object))
            w[(a, b)] += 1
    nb = collections.defaultdict(list)
    for (a, b), c in w.items():
        nb[a].append((c, b))
        nb[b].append((c, a))

    rows = []
    for i, n in enumerate(ents):
        g = [x for _, x in sorted(nb[n], key=lambda t: (-t[0], t[1]))]
        gk = g[:K]
        ek = [ents[j] for j in np.argsort(-sim[i])[:K]]
        ov = len(set(gk) & set(ek))
        rows.append((n, len(g), ov / len(gk) if gk else float("nan"), ov, gk, ek))

    d = pd.DataFrame(rows, columns=["entity", "degree", "overlap", "shared",
                                    "graph_top", "emb_top"])
    d = d.dropna(subset=["overlap"]).sort_values(["overlap", "degree"],
                                                 ascending=[True, False])

    def tbl(sub):
        L = ["| entity | degree | overlap | shared | top-%d graph neighbours | top-%d embedding neighbours |" % (K, K),
             "|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            L.append("| %s | %d | %.2f | %d | %s | %s |"
                     % (r.entity, r.degree, r.overlap, r.shared,
                        ", ".join(r.graph_top), ", ".join(r.emb_top)))
        return L

    hi = d[d.degree >= K]
    L = ["# Phase 8. Graph versus embedding neighbourhoods", "",
         "**Below the Phase 6 entry gate.** The embedding was trained on %d" % n_triples,
         "triples against a stated floor of 10,000, on instruction. A low overlap",
         "below is a candidate to check against Debroy's prose, not a result about",
         "the Mahabharata.", "",
         "k = %d. Graph neighbours ranked by the number of asserted triples joining" % K,
         "the pair. Embedding neighbours ranked by cosine. Overlap is the size of the",
         "intersection over k, or over the degree where the degree is below k.", "",
         "Entities scored: %d of %d (the rest have no asserted edge to another"
         % (len(d), len(ents)),
         "embedded entity). Mean overlap %.3f, and %.3f among the %d entities of"
         % (d.overlap.mean(), hi.overlap.mean(), len(hi)),
         "degree >= %d." % K, "",
         "## Overlap distribution", "",
         "| shared of %d | entities | of which degree >= %d |" % (K, K),
         "|---|---|---|"]
    for v in sorted(d.shared.unique()):
        L.append("| %d | %d | %d |" % (v, (d.shared == v).sum(), (hi.shared == v).sum()))
    L += ["",
          "The ranking the phase asks for is dominated by a tie. %d of the %d scored"
          % ((d.shared == 0).sum(), len(d)),
          "entities share nothing at all between the two neighbourhoods, so ordering by",
          "overlap orders the tie by degree and the list below is really the highest-",
          "degree entities in the graph. That is a property of this model, not of the",
          "epic: see docs/PHASE8_INSPECTION.md.", "",
          "## Ranked by lowest overlap, degree >= %d" % K, ""]
    L += tbl(hi.head(20))
    L += ["", "## Ranked by lowest overlap, all degrees", "",
          "Kept for completeness. The head of this list is dominated by degree 1 and",
          "2, where a single edge cannot intersect ten cosine neighbours and the low",
          "overlap carries no information.", ""]
    L += tbl(d.head(20))
    L += ["", "## Highest overlap, degree >= %d" % K, "",
          "The other end, where the vectors reproduce the edges.", ""]
    L += tbl(hi.sort_values(["overlap", "degree"], ascending=[False, False]).head(10))
    L += ["", "## Karna", ""]
    k = d[d.entity == "Karna"]
    L += tbl(k) if len(k) else ["Karna does not appear in the scored set."]
    L += [""]
    open("build/divergence.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    d.to_csv("build/divergence.csv", index=False)
    print("scored %d entities, %d at degree >= %d, mean overlap %.3f / %.3f"
          % (len(d), len(hi), K, d.overlap.mean(), hi.overlap.mean()))


if __name__ == "__main__":
    main()
