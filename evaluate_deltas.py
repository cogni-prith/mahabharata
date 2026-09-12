#!/usr/bin/env python3
"""Memorization-vs-recognition delta study, scored against HUMAN GOLD.

On the 8 human-annotated test parvas, three predictors are compared, each
scored against the human gold labels (B3) with strict span-F1:

  B1  = untrained en_core_web_lg (zero-shot, OntoNotes labels remapped)
  B2  = LLM silver labels themselves (the teacher's own accuracy vs gold)
  S   = spaCy-lg trained on the 10 LLM-only parvas (the distilled student)

Deltas (all vs gold, micro span-F1):
  Δ1 = F1(B2) − F1(B1)   how much the LLM teacher beats the zero-shot floor
  Δ2 = F1(S)  − F1(B1)   how much distillation training beats the floor  <-- the real test
  Δ3 = F1(B2) − F1(S)    teacher → student accuracy gap

Also reports a token-level label-agreement table (the original "agreement rate"
framing) and a per-parva / per-label breakdown. All 8 human parvas are kept in
the aggregate (per user); the per-parva table exposes the two barely-reviewed
parvas (MahaPrasthanika, Mausala) so their inflation is visible, not hidden.

Usage: .venv/bin/python evaluate_deltas.py [trained_model_dir]
Writes train_delta/delta_results.json.
"""
import glob, json, os, sys
from collections import Counter

import spacy
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import filter_spans

MODEL = sys.argv[1] if len(sys.argv) > 1 else "train_delta/model/model-best"
TEST_DIR = "train_delta/data/test"

# Off-the-shelf lg emits OntoNotes labels; map them generously into our schema.
# CONCEPT / WEAPON / CELESTIAL have no OntoNotes source -> B1 scores 0 there by construction.
MAP = {"PERSON": "PERSON", "NORP": "NORP", "GPE": "GPE", "ORG": "DYNASTY",
       "LOC": "GPE", "FAC": "GPE", "WORK_OF_ART": "SACRED_TEXT", "LAW": "SACRED_TEXT",
       "EVENT": "EVENT"}

PARVAS = sorted(os.path.basename(p)[:-len(".gold.spacy")]
                for p in glob.glob(f"{TEST_DIR}/*.gold.spacy"))


def run_pipeline(nlp, gold_doc, remap=None):
    """Predict on the gold tokenization so tokens align 1:1; return a Doc."""
    d = Doc(nlp.vocab, words=[t.text for t in gold_doc])
    for _, proc in nlp.pipeline:
        d = proc(d)
    if remap is not None:
        ents = [Span(d, e.start, e.end, remap[e.label_]) for e in d.ents if e.label_ in remap]
        d.ents = filter_spans(ents)
    return d


def silver_pred(gold_doc, silver_doc):
    """Cast the silver doc's ents onto the gold doc's vocab (identical tokens)."""
    d = Doc(gold_doc.vocab, words=[t.text for t in gold_doc])
    d.ents = [Span(d, e.start, e.end, e.label_) for e in silver_doc.ents]
    return d


def token_labels(doc):
    """Per-token entity type (or 'O'), collapsing B-/I-."""
    labs = ["O"] * len(doc)
    for e in doc.ents:
        for i in range(e.start, e.end):
            labs[i] = e.label_
    return labs


def main():
    student = spacy.load(MODEL)
    base = spacy.load("en_core_web_lg")

    # Accumulators for the aggregate (over all parvas concatenated).
    ex = {"B1": [], "B2": [], "S": []}
    tok_agree = {"B1": [0, 0], "B2": [0, 0], "S": [0, 0]}  # [correct, total]
    support = Counter()
    per_parva = {}

    for name in PARVAS:
        gold_docs = list(DocBin().from_disk(f"{TEST_DIR}/{name}.gold.spacy").get_docs(student.vocab))
        silver_docs = list(DocBin().from_disk(f"{TEST_DIR}/{name}.silver.spacy").get_docs(student.vocab))
        assert len(gold_docs) == len(silver_docs)

        p_ex = {"B1": [], "B2": [], "S": []}
        for gd, sd in zip(gold_docs, silver_docs):
            for e in gd.ents:
                support[e.label_] += 1
            preds = {
                "B1": run_pipeline(base, gd, MAP),
                "B2": silver_pred(gd, sd),
                "S":  run_pipeline(student, gd),
            }
            g_tok = token_labels(gd)
            for k, pd in preds.items():
                e_obj = Example(pd, gd)
                ex[k].append(e_obj); p_ex[k].append(e_obj)
                p_tok = token_labels(pd)
                tok_agree[k][0] += sum(1 for a, b in zip(p_tok, g_tok) if a == b)
                tok_agree[k][1] += len(g_tok)

        per_parva[name] = {
            k: round(Scorer().score(p_ex[k])["ents_f"] * 100, 2) for k in p_ex
        }

    # Aggregate scores.
    scores = {k: Scorer().score(ex[k]) for k in ex}
    total = sum(support.values())
    labels = sorted(support, key=lambda l: -support[l])

    def micro(k):  return scores[k]["ents_f"] * 100
    def per_type(k, lab): return scores[k]["ents_per_type"].get(lab, {}).get("f", 0.0) * 100

    def macro(k): return sum(per_type(k, l) for l in labels) / len(labels)
    def weighted(k): return sum(per_type(k, l) * support[l] for l in labels) / total

    D1 = micro("B2") - micro("B1")
    D2 = micro("S")  - micro("B1")
    D3 = micro("B2") - micro("S")

    # ---------- report ----------
    print(f"model : {MODEL}")
    print(f"test  : {len(PARVAS)} human-gold parvas, {sum(len(list(DocBin().from_disk(f'{TEST_DIR}/{n}.gold.spacy').get_docs(student.vocab))) for n in PARVAS)} sents, {total} gold entities\n")

    print("=== STRICT SPAN-F1 vs HUMAN GOLD (micro / macro / weighted) ===")
    print(f"{'system':<34}{'micro':>8}{'macro':>8}{'weighted':>10}")
    print("-" * 60)
    names = {"B1": "B1  untrained lg (zero-shot)",
             "S":  "S   trained lg (distilled)",
             "B2": "B2  LLM silver (teacher)"}
    for k in ("B1", "S", "B2"):
        print(f"{names[k]:<34}{micro(k):>8.2f}{macro(k):>8.2f}{weighted(k):>10.2f}")

    print("\n=== DELTAS (micro span-F1, vs human gold) ===")
    print(f"  Δ1  F1(B2) − F1(B1)   LLM teacher vs zero-shot floor : {D1:+.2f}")
    print(f"  Δ2  F1(S)  − F1(B1)   distillation vs zero-shot floor: {D2:+.2f}   <-- real test")
    print(f"  Δ3  F1(B2) − F1(S)    teacher → student gap          : {D3:+.2f}")

    print("\n=== PER-LABEL span-F1 vs gold ===")
    print(f"{'LABEL':<13}{'support':>9}{'B1':>8}{'S':>8}{'B2':>8}")
    print("-" * 46)
    for lab in labels:
        print(f"{lab:<13}{support[lab]:>9}{per_type('B1',lab):>8.2f}{per_type('S',lab):>8.2f}{per_type('B2',lab):>8.2f}")

    print("\n=== PER-PARVA micro span-F1 vs gold  (⚠ = <0.1% human edits) ===")
    unreviewed = {"MahaPrasthanika", "Mausala"}
    print(f"{'parva':<16}{'B1':>8}{'S':>8}{'B2':>8}")
    print("-" * 40)
    for name in PARVAS:
        flag = " ⚠" if name in unreviewed else ""
        pp = per_parva[name]
        print(f"{name:<16}{pp['B1']:>8.2f}{pp['S']:>8.2f}{pp['B2']:>8.2f}{flag}")

    print("\n=== TOKEN-LEVEL label agreement vs gold (%) ===")
    for k in ("B1", "S", "B2"):
        c, t = tok_agree[k]
        print(f"  {names[k]:<34}{100*c/t:>7.2f}")

    print("\n(all vs HUMAN GOLD. B2/silver near-perfect on the two ⚠ parvas is expected —")
    print(" those had ~0 human edits, so silver ≈ gold there by construction.)")

    out = {
        "model": MODEL, "parvas": PARVAS, "total_gold_entities": total,
        "micro": {k: round(micro(k), 2) for k in ex},
        "macro": {k: round(macro(k), 2) for k in ex},
        "weighted": {k: round(weighted(k), 2) for k in ex},
        "deltas": {"D1_B2_minus_B1": round(D1, 2),
                   "D2_S_minus_B1": round(D2, 2),
                   "D3_B2_minus_S": round(D3, 2)},
        "per_label_f1": {lab: {k: round(per_type(k, lab), 2) for k in ex} for lab in labels},
        "per_parva_micro_f1": per_parva,
        "token_agreement_pct": {k: round(100 * tok_agree[k][0] / tok_agree[k][1], 2) for k in ex},
        "support": dict(support),
    }
    with open("train_delta/delta_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote train_delta/delta_results.json")


if __name__ == "__main__":
    main()
