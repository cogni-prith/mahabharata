#!/usr/bin/env python3
"""Extra metrics for the student model (fast: student-only, no lg).

Dumps train_delta/extra_metrics.json:
  - per-label precision / recall / F1 (span-level) for the student vs human gold
  - a token-level confusion matrix (gold entity type incl. O  x  predicted)
Also parses train_delta/logs/train.log into a training curve.
"""
import glob, json, os, re
from collections import Counter

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.scorer import Scorer

MODEL = "train_delta/model/model-best"
TEST_DIR = "train_delta/data/test"
LABELS = ["PERSON", "DYNASTY", "NORP", "GPE", "CONCEPT", "WEAPON",
          "SACRED_TEXT", "EVENT", "CELESTIAL"]
ORDER = LABELS + ["O"]

nlp = spacy.load(MODEL)


def tok_types(doc):
    labs = ["O"] * len(doc)
    for e in doc.ents:
        for i in range(e.start, e.end):
            labs[i] = e.label_
    return labs


examples = []
conf = Counter()  # (gold_type, pred_type) -> count
for gp in sorted(glob.glob(f"{TEST_DIR}/*.gold.spacy")):
    gold_docs = list(DocBin().from_disk(gp).get_docs(nlp.vocab))
    for gd in gold_docs:
        pd = Doc(nlp.vocab, words=[t.text for t in gd])
        for _, proc in nlp.pipeline:
            pd = proc(pd)
        examples.append(Example(pd, gd))
        for g, p in zip(tok_types(gd), tok_types(pd)):
            conf[(g, p)] += 1

sc = Scorer().score(examples)
per = sc["ents_per_type"]
pr = {lab: {"p": round(per.get(lab, {}).get("p", 0.0) * 100, 2),
            "r": round(per.get(lab, {}).get("r", 0.0) * 100, 2),
            "f": round(per.get(lab, {}).get("f", 0.0) * 100, 2)}
      for lab in LABELS}

# Confusion matrix (rows=gold, cols=pred), row-normalized to % for the heatmap.
matrix = [[conf.get((g, p), 0) for p in ORDER] for g in ORDER]
row_pct = []
for row in matrix:
    tot = sum(row) or 1
    row_pct.append([round(100 * c / tot, 2) for c in row])

# Training curve from the log.
steps, devf, nerloss = [], [], []
pat = re.compile(r"^\s*\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
for line in open("train_delta/logs/train.log", errors="ignore"):
    m = pat.match(line)
    if m:
        steps.append(int(m.group(1)))
        nerloss.append(float(m.group(3)))
        devf.append(float(m.group(4)))

out = {
    "labels": LABELS, "order": ORDER,
    "student_pr": pr,
    "confusion_counts": matrix,
    "confusion_row_pct": row_pct,
    "training_curve": {"steps": steps, "dev_f": devf, "ner_loss": nerloss},
}
json.dump(out, open("train_delta/extra_metrics.json", "w"), indent=2)
print("wrote train_delta/extra_metrics.json")
print("student P/R/F per label:")
for lab in LABELS:
    print(f"  {lab:<13} P={pr[lab]['p']:6.2f}  R={pr[lab]['r']:6.2f}  F={pr[lab]['f']:6.2f}")
print(f"training curve points: {len(steps)}")
