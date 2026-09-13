"""Phase 7. UMAP projection and the static page's data file.

The page is opened from disk, so fetch() is not available: browsers refuse
cross-origin requests for file:// URLs and a local .bin or .json counts as one.
The data therefore ships as a .js file that assigns one global, which a
<script> tag loads under file:// without a server.

Two projections are written. The plan asks for 2D and that stays; the page
draws the 3D one because a third axis separates clusters the 2D collapses.
Both use the same seed.

Everything here is below the Phase 6 entry gate and the flag rides along in the
data so the page can say so on every screen.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import umap

SEED = 20260917
K = 8


def project(e, d):
    xy = umap.UMAP(n_components=d, random_state=SEED, n_neighbors=15,
                   min_dist=0.1, metric="cosine").fit_transform(e)
    return (xy - xy.min(0)) / (xy.max(0) - xy.min(0))


def main():
    idx = json.load(open("out/embedding_index.json"))
    ents, rels = idx["entities"], idx["relations"]
    e = np.fromfile("out/entity_emb.bin", dtype="float32").reshape(len(ents), idx["entity_dim"])
    r = np.fromfile("out/relation_emb.bin", dtype="float32").reshape(len(rels), idx["relation_dim"])

    xy, xyz = project(e, 2), project(e, 3)
    cl = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(e)

    ei = {n: i for i, n in enumerate(ents)}
    ri = {n: i for i, n in enumerate(rels)}
    rel = pd.read_parquet("build/relations.parquet")
    seen, edges = {}, []
    cols = ["subject", "predicate", "object", "parva", "adhyaya_ordinal",
            "cited", "evidence", "confidence", "depth"]
    for t in rel[cols].itertuples():
        if t.subject not in ei or t.object not in ei or t.predicate not in ri:
            continue
        k = (t.subject, t.predicate, t.object)
        if k in seen:
            edges[seen[k]][4] += 1          # another supporting sentence
            continue
        seen[k] = len(edges)
        cited = bool(t.cited) and t.adhyaya_ordinal == t.adhyaya_ordinal
        a = "%s %d" % (t.parva, int(t.adhyaya_ordinal)) if cited else t.parva
        ev = t.evidence if isinstance(t.evidence, str) else ""
        edges.append([ei[t.subject], ri[t.predicate], ei[t.object], a, 1,
                      1 if cited else 0, ev[:300], round(float(t.confidence), 3),
                      int(t.depth)])

    # Per-entity facts the side panel shows. Mention counts say how much of the
    # corpus a name actually occupies; the graph degree alone hides that.
    men = pd.read_parquet("build/mentions.parquet")
    counts = men.surface.str.strip().value_counts()
    tel = pd.read_parquet("build/tellings.parquet")
    narr = tel.narrator.value_counts()
    meta = {}
    for n in ents:
        m = {}
        if n in counts.index:
            m["mentions"] = int(counts[n])
        if n in narr.index:
            m["tells"] = int(narr[n])
        if m:
            meta[n] = m

    # Appellation candidates, unresolved by design. The hard rule forbids
    # picking one, so the page shows the set with its evidence and says so.
    ap = pd.read_csv("build/appellations.csv")
    alias = {}
    for t in ap.itertuples():
        if t.candidate_entity in ei:
            alias.setdefault(t.candidate_entity, []).append(
                [t.appellation, round(float(t.confidence), 3), int(t.evidence_count), t.status])
    for v in alias.values():
        v.sort(key=lambda x: -x[1])

    data = {"entities": ents, "relations": rels, "dim": idx["entity_dim"],
            "xy": np.round(xy, 4).tolist(), "xyz": np.round(xyz, 4).tolist(),
            "cluster": cl.tolist(),
            "emb": np.round(e, 4).ravel().tolist(),
            "rel_emb": np.round(r, 4).ravel().tolist(),
            "edges": edges, "meta": meta, "alias": alias,
            "umap_seed": SEED, "train_seed": idx["seed"],
            "model": idx["model"], "below_gate": True}
    with open("out/projection.js", "w") as f:
        f.write("window.KG=")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    # The gate asks for projection.json by name; the page cannot read it from
    # disk, so it holds the coordinates and the seed and nothing else.
    json.dump({"seed": SEED, "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1,
               "below_gate": True,
               "coords": {n: [round(float(a), 4), round(float(b), 4)]
                          for n, (a, b) in zip(ents, xy)},
               "coords3d": {n: [round(float(v), 4) for v in p]
                            for n, p in zip(ents, xyz)}},
              open("out/projection.json", "w"))
    print("projected %d entities in 2D and 3D, %d edges, umap seed %d"
          % (len(ents), len(edges), SEED))


if __name__ == "__main__":
    main()
