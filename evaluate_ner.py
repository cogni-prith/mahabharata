#!/usr/bin/env python3
"""Evaluate a trained spaCy NER model on a .spacy DocBin and report per-label
P/R/F1 with support, plus micro-, macro-, and weighted-F1.

NOTE: scores are vs the *silver* (LLM) dev labels -> they measure fidelity to the
LLM teacher, NOT accuracy. Real accuracy comes only from the human-gold set.

Usage: python evaluate_ner.py [model_dir] [data.spacy]
"""
import sys
from collections import Counter

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.scorer import Scorer

model = sys.argv[1] if len(sys.argv) > 1 else "train_spacy_lg/model/model-best"
data = sys.argv[2] if len(sys.argv) > 2 else "train_spacy_lg/data/dev.spacy"

nlp = spacy.load(model)
gold_docs = list(DocBin().from_disk(data).get_docs(nlp.vocab))

support = Counter()
examples = []
for gd in gold_docs:
    for e in gd.ents:
        support[e.label_] += 1
    # Predict on the *gold tokenization* so tokens align 1:1 (docs were built
    # from explicit words, not raw text).
    pred = Doc(nlp.vocab, words=[t.text for t in gd])
    for _, proc in nlp.pipeline:
        pred = proc(pred)
    examples.append(Example(pred, gd))

scores = Scorer().score(examples)
per_type = scores["ents_per_type"]
total_support = sum(support.values())

print(f"model : {model}")
print(f"data  : {data}   ({len(gold_docs)} docs, {total_support} gold entities)")
print(f"\n{'LABEL':<13}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}")
print("-" * 47)
weighted = macro = 0.0
for lab in sorted(per_type, key=lambda l: -support[l]):
    p, r, f = per_type[lab]["p"], per_type[lab]["r"], per_type[lab]["f"]
    s = support[lab]
    weighted += f * s
    macro += f
    print(f"{lab:<13}{p*100:>8.2f}{r*100:>8.2f}{f*100:>8.2f}{s:>10}")
n_lab = len(per_type)
print("-" * 47)
print(f"{'micro-F1':<13}{'':>8}{'':>8}{scores['ents_f']*100:>8.2f}{total_support:>10}")
print(f"{'macro-F1':<13}{'':>8}{'':>8}{(macro/n_lab)*100:>8.2f}{n_lab:>10}  (labels)")
print(f"{'weighted-F1':<13}{'':>8}{'':>8}{(weighted/total_support)*100:>8.2f}{total_support:>10}")
print("\n(silver dev labels: this is fidelity to the LLM teacher, not accuracy)")
