"""Phase 7. UMAP projection and the static page's data file.

The page is opened from disk, so fetch() is not available: browsers refuse
cross-origin requests for file:// URLs and a local .bin or .json counts as one.
The data therefore ships as a .js file that assigns one global, which a
<script> tag loads under file:// without a server.

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


def main():
    idx = json.load(open("out/embedding_index.json"))
    ents, rels = idx["entities"], idx["relations"]
    e = np.fromfile("out/entity_emb.bin", dtype="float32").reshape(len(ents), idx["entity_dim"])
    r = np.fromfile("out/relation_emb.bin", dtype="float32").reshape(len(rels), idx["relation_dim"])

    xy = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15,
                   min_dist=0.1, metric="cosine").fit_transform(e)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    cl = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(e)

    ei = {n: i for i, n in enumerate(ents)}
    ri = {n: i for i, n in enumerate(rels)}
    rel = pd.read_parquet("build/relations.parquet")
    seen, edges = set(), []
    for t in rel[["subject", "predicate", "object", "parva", "adhyaya_ordinal"]].itertuples():
        if t.subject in ei and t.object in ei and t.predicate in ri:
            k = (t.subject, t.predicate, t.object)
            if k in seen:
                continue
            seen.add(k)
            a = "%s %s" % (t.parva, int(t.adhyaya_ordinal)) if t.adhyaya_ordinal == t.adhyaya_ordinal else t.parva
            edges.append([ei[t.subject], ri[t.predicate], ei[t.object], a])

    data = {"entities": ents, "relations": rels, "dim": idx["entity_dim"],
            "xy": np.round(xy, 4).tolist(), "cluster": cl.tolist(),
            "emb": np.round(e, 4).ravel().tolist(),
            "rel_emb": np.round(r, 4).ravel().tolist(),
            "edges": edges, "umap_seed": SEED, "train_seed": idx["seed"],
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
                          for n, (a, b) in zip(ents, xy)}},
              open("out/projection.json", "w"))
    print("projected %d entities, %d edges, umap seed %d" % (len(ents), len(edges), SEED))


if __name__ == "__main__":
    main()
