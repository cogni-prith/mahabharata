#!/usr/bin/env python3
"""B1 vs fine-tuned: compare untrained en_core_web_lg NER (zero-shot) against the
fine-tuned model, BOTH scored against the LLM silver dev labels ("truth" here).

Off-the-shelf lg emits OntoNotes labels, so they are mapped generously into the
9-type schema. CONCEPT and WEAPON have no OntoNotes source -> the baseline scores
0 on them by construction (flagged, not hidden).

Usage: python compare_baseline.py [trained_model] [data.spacy]
"""
import sys
from collections import Counter

import spacy
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import filter_spans

trained = sys.argv[1] if len(sys.argv) > 1 else "train_spacy_lg/model/model-best"
data = sys.argv[2] if len(sys.argv) > 2 else "train_spacy_lg/data/dev.spacy"

# Generous OntoNotes -> our-schema map for the zero-shot baseline.
MAP = {"PERSON": "PERSON", "NORP": "NORP", "GPE": "GPE", "ORG": "DYNASTY",
       "LOC": "GPE", "FAC": "GPE", "WORK_OF_ART": "SACRED_TEXT", "LAW": "SACRED_TEXT",
       "EVENT": "EVENT"}

base = spacy.load("en_core_web_lg")                 # untrained (OntoNotes) NER
tuned = spacy.load(trained)                          # fine-tuned on LLM silver
gold_docs = list(DocBin().from_disk(data).get_docs(tuned.vocab))

support = Counter()
for gd in gold_docs:
    support.update(e.label_ for e in gd.ents)
total = sum(support.values())


def run(nlp, gd, remap=None):
    d = Doc(nlp.vocab, words=[t.text for t in gd])
    for _, proc in nlp.pipeline:
        d = proc(d)
    if remap is not None:
        ents = [Span(d, e.start, e.end, remap[e.label_]) for e in d.ents if e.label_ in remap]
        d.ents = filter_spans(ents)
    return d


def evaluate(nlp, remap=None):
    exs = [Example(run(nlp, gd, remap), gd) for gd in gold_docs]
    return Scorer().score(exs)


s_base = evaluate(base, MAP)
s_tuned = evaluate(tuned, None)

labels = sorted(support, key=lambda l: -support[l])
pt_b, pt_t = s_base["ents_per_type"], s_tuned["ents_per_type"]

print(f"data: {data}  ({len(gold_docs)} docs, {total} gold entities)  truth = LLM silver labels\n")
print(f"{'LABEL':<13}{'support':>8}{'F1 untrained':>14}{'F1 trained':>13}{'Δ (gain)':>11}")
print("-" * 59)
wb = wt = mb = mt = 0.0
for lab in labels:
    fb = pt_b.get(lab, {}).get("f", 0.0) * 100
    ft = pt_t.get(lab, {}).get("f", 0.0) * 100
    s = support[lab]
    wb += fb * s; wt += ft * s; mb += fb; mt += ft
    flag = "  (no OntoNotes source)" if lab in ("CONCEPT", "WEAPON") else ""
    print(f"{lab:<13}{s:>8}{fb:>14.2f}{ft:>13.2f}{ft-fb:>+11.2f}{flag}")
n = len(labels)
print("-" * 59)
mib, mit = s_base["ents_f"] * 100, s_tuned["ents_f"] * 100
print(f"{'micro-F1':<13}{total:>8}{mib:>14.2f}{mit:>13.2f}{mit-mib:>+11.2f}")
print(f"{'macro-F1':<13}{n:>8}{mb/n:>14.2f}{mt/n:>13.2f}{(mt-mb)/n:>+11.2f}")
print(f"{'weighted-F1':<13}{total:>8}{wb/total:>14.2f}{wt/total:>13.2f}{(wt-wb)/total:>+11.2f}")
print(f"\nmicro-P/R  untrained: {s_base['ents_p']*100:.1f}/{s_base['ents_r']*100:.1f}"
      f"   trained: {s_tuned['ents_p']*100:.1f}/{s_tuned['ents_r']*100:.1f}")
print("\n(all scores vs LLM silver labels = agreement with the teacher, not accuracy)")
