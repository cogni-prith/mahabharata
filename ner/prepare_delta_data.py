#!/usr/bin/env python3
"""Prepare data for the memorization-vs-recognition delta study.

Split (parva-disjoint, no leakage):
  TRAIN  = the 10 LLM-annotated parvas that have NO human-checked counterpart.
  TEST   = the 8 human-annotated parvas. For each we emit TWO DocBins over the
           SAME tokenization: a GOLD one (human `tag`) and a SILVER one (LLM
           `tag`). Row alignment between the human and LLM CSVs is 1:1, so the
           tokens are identical and only the labels differ.

Everything is built as Doc(vocab, words=...) so tokens map 1:1 to BIO tags.

Outputs under train_delta/:
  data/train.spacy, data/dev.spacy          (internal 90/10 carve of TRAIN)
  data/test/<parva>.gold.spacy              (human labels)
  data/test/<parva>.silver.spacy            (LLM labels)
  split_manifest.json                        (provenance + counts)
"""
import csv, json, os, random, sys
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc, DocBin, Span
from spacy.util import filter_spans

LLM_DIR = "Mahabharata LLM annotated-20260801T163642Z-1-001/Mahabharata LLM annotated"
HUM_DIR = "Mahabharat_Human_Annotated_Data-20260801T163709Z-1-001/Mahabharat_Human_Annotated_Data"
OUT = "train_delta"
SEED = 42
DEV_RATIO = 0.10

# TEST parvas: (canonical name, human file, llm file). All 8 human-checked parvas.
TEST_PAIRS = [
    ("AshramaVasika", "AshramaVasika_parva_filled.csv", "AshramaVasika_parva_filled LLM.csv"),
    ("Bhishma",       "Bhisma_tagged-checked.csv",      "Bhisma_tagged.csv"),
    ("Karna",         "KarnaParvaRe.csv",               "KarnaParva_tagged_filled.csv"),
    ("MahaPrasthanika","MahaPrasthanika_tagged_filled.csv","MahaPrasthanika_tagged_filled.csv"),
    ("Mausala",       "Mausala_tagged-checked.csv",     "Mausala_tagged.csv"),
    ("Svargarohana",  "Savargarohana_Parva_filled.csv", "Savargarohana_Parva_filled.csv"),
    ("Shalya",        "ShalyaParvaRe.csv",              "ShalyaParva_tagged_filled.csv"),
    ("Sauptika",      "Souptika_tagged-checked.csv",    "Souptika_tagged.csv"),
]
TEST_LLM_FILES = {p[2] for p in TEST_PAIRS}  # LLM files that belong to TEST -> excluded from TRAIN

O_TOKENS = {"", "o", "o(clear entity)", "o (clear entity)"}


def normalize_tag(raw):
    """Return 'O', 'B-LABEL', or 'I-LABEL'. Malformed 'O-LABEL' typos -> O."""
    t = (raw or "").strip()
    if t.lower() in O_TOKENS:
        return "O"
    if t[:2] in ("B-", "I-"):
        return t
    return "O"  # e.g. 'O-PERSON' typos


def read_sentences(path):
    """Yield (words, tags) per sentence_no, in file order. Empty words -> '_'."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cur_no, words, tags = None, [], []
        for row in reader:
            sn = row.get("sentence_no")
            w = (row.get("word") or "").strip() or "_"
            t = normalize_tag(row.get("tag"))
            if sn != cur_no and cur_no is not None:
                yield words, tags
                words, tags = [], []
            cur_no = sn
            words.append(w)
            tags.append(t)
        if words:
            yield words, tags


def spans_from_bio(doc, tags):
    """Non-overlapping spans honoring B- boundaries; promote stray I- to a new span."""
    spans, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue
        label = t[2:]
        j = i + 1
        while j < n and tags[j] == "I-" + label:
            j += 1
        spans.append(Span(doc, i, j, label=label))
        i = j
    return filter_spans(spans)


def build_docbin(vocab, path, count_labels=None):
    db = DocBin(store_user_data=True)
    n_sent = n_span = 0
    for words, tags in read_sentences(path):
        doc = Doc(vocab, words=words)
        doc.ents = spans_from_bio(doc, tags)
        if count_labels is not None:
            for e in doc.ents:
                count_labels[e.label_] += 1
        n_span += len(doc.ents)
        n_sent += 1
        db.add(doc)
    return db, n_sent, n_span


def read_rows(path):
    """Yield (sentence_no, word, normalized_tag) per row, in file order."""
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = (row.get("word") or "").strip() or "_"
            yield row.get("sentence_no"), w, normalize_tag(row.get("tag"))


def build_test_pair(vocab, gold_path, silver_path):
    """Build gold + silver DocBins over ONE shared segmentation.

    Rows are 1:1 aligned between the two files, but sentence_no boundaries can
    differ. Use the gold file's sentence_no + words as canonical for both, so
    gold and silver docs are structurally identical and doc-aligned for scoring.
    Returns (gold_db, silver_db, n_sent, gold_spans, silver_spans, gc, sc).
    """
    grows = list(read_rows(gold_path))
    srows = list(read_rows(silver_path))
    assert len(grows) == len(srows), f"row mismatch {len(grows)} vs {len(srows)}"

    g_db, s_db = DocBin(store_user_data=True), DocBin(store_user_data=True)
    gc, sc = Counter(), Counter()
    n_sent = g_spans = s_spans = 0
    words, gtags, stags = [], [], []
    cur = None

    def flush():
        nonlocal n_sent, g_spans, s_spans
        if not words:
            return
        gd = Doc(vocab, words=words)
        gd.ents = spans_from_bio(gd, gtags)
        sd = Doc(vocab, words=words)
        sd.ents = spans_from_bio(sd, stags)
        for e in gd.ents:
            gc[e.label_] += 1
        for e in sd.ents:
            sc[e.label_] += 1
        g_spans += len(gd.ents); s_spans += len(sd.ents); n_sent += 1
        g_db.add(gd); s_db.add(sd)

    for (gsn, gw, gt), (_, _, st) in zip(grows, srows):
        if gsn != cur and cur is not None:
            flush(); words, gtags, stags = [], [], []
        cur = gsn
        words.append(gw); gtags.append(gt); stags.append(st)
    flush()
    return g_db, s_db, n_sent, g_spans, s_spans, gc, sc


def main():
    nlp = spacy.blank("en")
    os.makedirs(f"{OUT}/data/test", exist_ok=True)

    # ---- TRAIN: 10 LLM-only parvas ----
    train_files = sorted(
        fn for fn in os.listdir(LLM_DIR)
        if fn.endswith(".csv") and fn not in TEST_LLM_FILES
    )
    assert len(train_files) == 10, f"expected 10 train parvas, got {len(train_files)}: {train_files}"

    train_docs = []
    train_label_counts = Counter()
    per_train = {}
    for fn in train_files:
        db, ns, nsp = build_docbin(nlp.vocab, os.path.join(LLM_DIR, fn), train_label_counts)
        docs = list(db.get_docs(nlp.vocab))
        train_docs.extend(docs)
        per_train[fn] = {"sentences": ns, "spans": nsp}

    rng = random.Random(SEED)
    rng.shuffle(train_docs)
    n_dev = max(1, round(len(train_docs) * DEV_RATIO))
    dev_docs, tr_docs = train_docs[:n_dev], train_docs[n_dev:]

    db_tr = DocBin(store_user_data=True, docs=tr_docs)
    db_dev = DocBin(store_user_data=True, docs=dev_docs)
    db_tr.to_disk(f"{OUT}/data/train.spacy")
    db_dev.to_disk(f"{OUT}/data/dev.spacy")

    # ---- TEST: 8 human parvas, gold + silver DocBins over identical tokens ----
    per_test = {}
    for name, hf, lf in TEST_PAIRS:
        g_db, s_db, ns, g_sp, s_sp, gc, sc = build_test_pair(
            nlp.vocab, os.path.join(HUM_DIR, hf), os.path.join(LLM_DIR, lf))
        g_db.to_disk(f"{OUT}/data/test/{name}.gold.spacy")
        s_db.to_disk(f"{OUT}/data/test/{name}.silver.spacy")
        per_test[name] = {
            "sentences": ns,
            "gold_spans": g_sp, "silver_spans": s_sp,
            "gold_labels": dict(gc), "silver_labels": dict(sc),
        }

    manifest = {
        "seed": SEED, "dev_ratio": DEV_RATIO,
        "train_parvas": train_files,
        "train_sentences_total": len(train_docs),
        "train_sentences_train": len(tr_docs),
        "train_sentences_dev": len(dev_docs),
        "train_label_counts": dict(train_label_counts),
        "per_train_file": per_train,
        "test_parvas": [p[0] for p in TEST_PAIRS],
        "per_test_parva": per_test,
    }
    with open(f"{OUT}/split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"TRAIN: {len(train_files)} parvas, {len(train_docs)} sents "
          f"({len(tr_docs)} train / {len(dev_docs)} dev), {sum(train_label_counts.values())} spans")
    print("TEST parvas (gold spans / silver spans):")
    for name in [p[0] for p in TEST_PAIRS]:
        t = per_test[name]
        print(f"  {name:<16} {t['sentences']:>6} sents  gold={t['gold_spans']:>6}  silver={t['silver_spans']:>6}")
    print(f"\nwrote {OUT}/data/train.spacy, dev.spacy, test/*.{{gold,silver}}.spacy, split_manifest.json")


if __name__ == "__main__":
    main()
