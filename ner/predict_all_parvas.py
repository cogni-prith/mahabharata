#!/usr/bin/env python3
"""Annotate all 18 parvas with the trained spaCy-lg student, one CSV per parva.

For each parva, predicts NER on the CSV's own tokenization and writes:
  sentence_no, word, llm_tag, pred_tag [, human_tag]
- llm_tag  = the original LLM silver label (from the LLM CSV).
- pred_tag = the trained spaCy-lg student's prediction (B-/I-/O).
- human_tag = human gold, only for the 8 human-checked parvas (rows are 1:1
  aligned to the LLM file, so it's joined by row index).

⚠ The 10 TRAIN parvas are IN-SAMPLE for this model (it trained on their silver
labels), so their pred≈llm agreement is memorization, not generalization. The 8
human parvas are the honest held-out set. This is flagged per-file in the summary.

Output: train_delta/predictions/<Parva>.csv  +  _summary.csv
"""
import csv, os
from collections import Counter

import spacy
from spacy.tokens import Doc, DocBin, Span  # noqa
from spacy.util import filter_spans  # noqa

MODEL = "train_delta/model/model-best"
LLM_DIR = "Mahabharata LLM annotated-20260801T163642Z-1-001/Mahabharata LLM annotated"
HUM_DIR = "Mahabharat_Human_Annotated_Data-20260801T163709Z-1-001/Mahabharat_Human_Annotated_Data"
OUT = "train_delta/predictions"

# canonical name -> (llm_file, human_file_or_None, is_train)
PARVAS = {
    "01_AdiParva":        ("AdiParva_tagged_filled.csv", None, True),
    "02_SabhaParva":      ("sabha_parva_tagged.csv", None, True),
    "03_AranyakaParva":   ("Aranyaka_tagged.csv", None, True),
    "04_VirataParva":     ("virata_parva_tagged.csv", None, True),
    "05_UdyogaParva":     ("udoyogaparva_filled LLM.csv", None, True),
    "06_BhishmaParva":    ("Bhisma_tagged.csv", "Bhisma_tagged-checked.csv", False),
    "07_DronaParva":      ("DronaParva_tagged.csv", None, True),
    "08_KarnaParva":      ("KarnaParva_tagged_filled.csv", "KarnaParvaRe.csv", False),
    "09_ShalyaParva":     ("ShalyaParva_tagged_filled.csv", "ShalyaParvaRe.csv", False),
    "10_SauptikaParva":   ("Souptika_tagged.csv", "Souptika_tagged-checked.csv", False),
    "11_StriParva":       ("stri_parva_tagged.csv", None, True),
    "12_ShantiParva":     ("shanti_parva.csv", None, True),
    "13_AnushasanaParva": ("anushasana_tagged.csv", None, True),
    "14_AshvamedhaParva": ("AshwamedhaParva_tagged.csv", None, True),
    "15_AshramaVasikaParva": ("AshramaVasika_parva_filled LLM.csv", "AshramaVasika_parva_filled.csv", False),
    "16_MausalaParva":    ("Mausala_tagged.csv", "Mausala_tagged-checked.csv", False),
    "17_MahaPrasthanikaParva": ("MahaPrasthanika_tagged_filled.csv", "MahaPrasthanika_tagged_filled.csv", False),
    "18_SvargarohanaParva": ("Savargarohana_Parva_filled.csv", "Savargarohana_Parva_filled.csv", False),
}

O_TOKENS = {"", "o", "o(clear entity)", "o (clear entity)"}


def norm(raw):
    t = (raw or "").strip()
    if t.lower() in O_TOKENS:
        return "O"
    return t if t[:2] in ("B-", "I-") else "O"


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row.get("sentence_no"), ((row.get("word") or "").strip() or "_"), norm(row.get("tag"))


def pred_bio(doc):
    bio = ["O"] * len(doc)
    for e in doc.ents:
        bio[e.start] = "B-" + e.label_
        for i in range(e.start + 1, e.end):
            bio[i] = "I-" + e.label_
    return bio


def main():
    os.makedirs(OUT, exist_ok=True)
    nlp = spacy.load(MODEL)
    summary = []

    for name, (lf, hf, is_train) in PARVAS.items():
        rows = list(read_rows(os.path.join(LLM_DIR, lf)))  # (sn, word, llm_tag)
        human = None
        if hf:
            human = [norm(t) for _, _, t in read_rows(os.path.join(HUM_DIR, hf))]
            if len(human) != len(rows):  # safety: truncate to min, warn
                n = min(len(human), len(rows))
                human = human[:n]

        # group into sentences (by sentence_no), predict, emit in row order
        out_path = os.path.join(OUT, f"{name}.csv")
        fields = ["sentence_no", "word", "llm_tag", "pred_tag"] + (["human_tag"] if hf else [])
        n_pred = Counter()
        agree_llm = [0, 0]
        agree_hum = [0, 0]
        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout); w.writerow(fields)
            i, N = 0, len(rows)
            while i < N:
                sn = rows[i][0]
                j = i
                while j < N and rows[j][0] == sn:
                    j += 1
                words = [rows[k][1] for k in range(i, j)]
                doc = Doc(nlp.vocab, words=words)
                for _, proc in nlp.pipeline:
                    doc = proc(doc)
                bio = pred_bio(doc)
                for off, k in enumerate(range(i, j)):
                    sn_k, word_k, llm_k = rows[k]
                    p = bio[off]
                    if p != "O":
                        n_pred[p[2:]] += 1
                    agree_llm[1] += 1
                    if p == llm_k:
                        agree_llm[0] += 1
                    rec = [sn_k, word_k, llm_k, p]
                    if hf:
                        h = human[k] if human and k < len(human) else "O"
                        rec.append(h)
                        agree_hum[1] += 1
                        if p == h:
                            agree_hum[0] += 1
                    w.writerow(rec)
                i = j

        summary.append({
            "parva": name, "split": "train(in-sample)" if is_train else "test(held-out)",
            "tokens": len(rows),
            "pred_entities": sum(n_pred.values()),
            "tok_agree_vs_llm_%": round(100 * agree_llm[0] / max(1, agree_llm[1]), 2),
            "tok_agree_vs_human_%": (round(100 * agree_hum[0] / max(1, agree_hum[1]), 2) if hf else ""),
        })
        print(f"  {name:<26} {summary[-1]['split']:<18} "
              f"tokens={len(rows):>7}  pred_ents={sum(n_pred.values()):>6}  "
              f"vs_llm={summary[-1]['tok_agree_vs_llm_%']:>6}%  "
              f"vs_human={summary[-1]['tok_agree_vs_human_%']}")

    with open(os.path.join(OUT, "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nwrote {len(summary)} parva CSVs + _summary.csv to {OUT}/")


if __name__ == "__main__":
    main()
