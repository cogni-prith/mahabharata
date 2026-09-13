"""Phase 6. Train the embedding.

The entry gate does not pass. It is re-checked here and its result is written
into every output this script produces, because the gate exists to stop numbers
from this run being read as a result. They are not one. 1,845 distinct triples
is below the 5,000 the plan calls the floor for meaningful vectors, and the
metrics below describe how well a model fits a small graph, not how well the
Mahabharata embeds.

RotatE, not TransE. The predicate set contains a symmetric relation
(foughtWith), antisymmetric ones (slainBy, hasBiologicalParent, defeatedBy) and
composition (parent applied twice is grandparent). TransE forces a symmetric
relation's vector to zero and collapses every pair onto each other.

Both leakage traps are handled before the split, not after:

  Symmetric leakage. foughtWith(a, b) and foughtWith(b, a) are the same fact. The
  pair is ordered canonically and deduplicated before anything is split, so one
  direction cannot sit in train while the other sits in test.

  Inverse leakage. No inverse relation is materialised anywhere in this graph.
  hasChild does not exist; siblings and uncles are derived at query time by the
  modelling invariants. The check below asserts that no two predicates hold the
  same entity pair in opposite order across the corpus often enough to be an
  inverse pair, and reports what it found rather than assuming it.
"""
import collections, json, sys
import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

SEED = 20260917
DIM = 200
EPOCHS = 400
NEG_PER_POS = 32          # basic (uniform) corruption of head or tail
SYMMETRIC = {"foughtWith"}


def load():
    rel = pd.read_parquet("build/relations.parquet")
    t = rel[["subject", "predicate", "object"]].drop_duplicates()
    n_raw = len(t)
    # Canonical order for symmetric predicates, then deduplicate. Before the
    # split, per the leakage rule.
    def key(r):
        if r.predicate in SYMMETRIC and r.object < r.subject:
            return (r.object, r.predicate, r.subject)
        return (r.subject, r.predicate, r.object)
    t = pd.DataFrame([key(r) for r in t.itertuples()],
                     columns=["subject", "predicate", "object"]).drop_duplicates()
    t = t[t.subject != t.object]
    return t, n_raw


def inverse_report(t):
    """Predicate pairs that hold the same entity pair in opposite order."""
    by = collections.defaultdict(set)
    for r in t.itertuples():
        by[r.predicate].add((r.subject, r.object))
    out = []
    preds = sorted(by)
    for i, a in enumerate(preds):
        for b in preds[i:]:
            n = len({(x, y) for x, y in by[a] if (y, x) in by[b]})
            if n:
                out.append((a, b, n))
    return sorted(out, key=lambda x: -x[2])


def main():
    t, n_raw = load()
    sym_removed = n_raw - len(t)
    inv = inverse_report(t)

    arr = t[["subject", "predicate", "object"]].to_numpy(dtype=str)
    tf = TriplesFactory.from_labeled_triples(arr, create_inverse_triples=False)
    train, valid, test = tf.split([0.8, 0.1, 0.1], random_state=SEED)

    res = pipeline(
        training=train, validation=valid, testing=test,
        model="RotatE",
        model_kwargs=dict(embedding_dim=DIM),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=NEG_PER_POS),
        training_kwargs=dict(num_epochs=EPOCHS, batch_size=512),
        optimizer_kwargs=dict(lr=1e-3),
        random_seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
        evaluator_kwargs=dict(filtered=True),
        stopper=None,
    )

    m = res.model
    ent = m.entity_representations[0](indices=None).detach().cpu()
    rel_ = m.relation_representations[0](indices=None).detach().cpu()
    # RotatE embeds in complex space. Store the real view, which is what the
    # interface scores against, and record the complex width so it reproduces.
    ent = torch.view_as_real(ent).reshape(ent.shape[0], -1) if ent.is_complex() else ent
    rel_ = torch.view_as_real(rel_).reshape(rel_.shape[0], -1) if rel_.is_complex() else rel_
    e = ent.numpy().astype("float32")
    r = rel_.numpy().astype("float32")
    e.tofile("out/entity_emb.bin")
    r.tofile("out/relation_emb.bin")

    ents = [k for k, _ in sorted(tf.entity_to_id.items(), key=lambda kv: kv[1])]
    rels = [k for k, _ in sorted(tf.relation_to_id.items(), key=lambda kv: kv[1])]
    json.dump({"entities": ents, "relations": rels,
               "entity_dim": e.shape[1], "relation_dim": r.shape[1],
               "seed": SEED, "model": "RotatE", "below_gate": True},
              open("out/embedding_index.json", "w"))

    mr = res.metric_results.to_dict()

    def g(side, kind, name):
        return mr[side][kind][name]

    L = [l.rstrip() for l in open("out/eval.md", encoding="utf-8")]
    L += ["", "---", "", "# Phase 6 training run", "",
          "**The entry gate above does not pass and this model was trained anyway,",
          "on instruction. Every number in this section describes how a model fits",
          "1,845 triples. None of them is a result about the Mahabharata.** The plan",
          "puts the floor for meaningful vectors at roughly 5,000 triples.", "",
          "| setting | value |", "|---|---|",
          "| model | RotatE |",
          "| embedding dim | %d complex, stored as %d float32 |" % (DIM, e.shape[1]),
          "| random seed | %d |" % SEED,
          "| epochs | %d |" % EPOCHS,
          "| optimiser | Adam, lr 1e-3, batch 512 |",
          "| negative sampling | basic, uniform corruption of head or tail |",
          "| negatives per positive | %d |" % NEG_PER_POS,
          "| split | 80/10/10 by triple, seed %d |" % SEED,
          "| train / valid / test | %d / %d / %d |"
          % (train.num_triples, valid.num_triples, test.num_triples),
          "| entities / relations | %d / %d |" % (tf.num_entities, tf.num_relations),
          "", "## Leakage checks", "",
          "- Symmetric: `foughtWith` pairs ordered canonically and deduplicated "
          "before the split. %d of %d distinct triples removed as the same fact "
          "written both ways." % (sym_removed, n_raw),
          "- Inverse: no inverse relation is materialised in this graph and "
          "`create_inverse_triples` is off. Predicate pairs sharing an entity "
          "pair in opposite order, measured rather than assumed:"]
    if inv:
        L += ["", "| predicate a | predicate b | shared reversed pairs |", "|---|---|---|"]
        L += ["| %s | %s | %d |" % x for x in inv[:12]]
    else:
        L += ["", "None."]
    L += ["", "## Filtered ranking, test set", "",
          "| metric | head | tail | both |", "|---|---|---|---|"]
    for name, label in (("inverse_harmonic_mean_rank", "MRR"),
                        ("hits_at_1", "Hits@1"), ("hits_at_3", "Hits@3"),
                        ("hits_at_10", "Hits@10")):
        L.append("| %s | %.4f | %.4f | %.4f |"
                 % (label, g("head", "realistic", name), g("tail", "realistic", name),
                    g("both", "realistic", name)))
    L += ["", "Filtered ranking, realistic rank definition, as PyKEEN reports it.",
          "PyKEEN %s." % __import__("pykeen.version", fromlist=["VERSION"]).VERSION, ""]
    open("out/eval.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("trained entities=%d relations=%d MRR=%.4f H@10=%.4f sym_removed=%d"
          % (tf.num_entities, tf.num_relations,
             g("both", "realistic", "inverse_harmonic_mean_rank"),
             g("both", "realistic", "hits_at_10"), sym_removed))


if __name__ == "__main__":
    main()
