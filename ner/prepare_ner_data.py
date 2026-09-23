#!/usr/bin/env python3
"""Convert the 50% LLM-annotated Mahabharata parvas (per-word BIO CSVs) into
spaCy training data (DocBin) plus a documented, reproducible train/dev split.

Input : one CSV per parva with columns  sentence_no, sentence, word, [POS,] tag
        where `tag` is BIO ("B-PERSON", "I-DYNASTY", ...) and O is written as
        an empty string OR the literal string "O(Clear Entity)".

Output (under --out):
  data/train.spacy    spaCy DocBin, ~90% of the sentences (training corpus)
  data/dev.spacy      spaCy DocBin, ~10% held back only so `spacy train` has an
                      evaluation set for checkpoint selection.  This is an
                      *internal* split of the 50% LLM training half; the OTHER
                      50% (the 9 hold-out parvas) is never read here.
  split_manifest.json full provenance: files, seed, ratio, per-parva & per-label
                      counts, and the intended (not-materialised) hold-out parvas.

Docs are built directly from the CSV's own word tokenization, so the token
stream is 1:1 with the BIO tags -- no char-offset re-alignment is needed.
"""
import argparse, csv, glob, json, os, random, sys
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc, DocBin, Span

csv.field_size_limit(10_000_000)

# O may be encoded as "", "O", or the LLM's verbose "O(Clear Entity)".
O_TOKENS = {"", "o", "o(clear entity)", "o (clear entity)"}

# The 8 entity types present in the LLM annotation (no LOC in this corpus).
LABELS = ["PERSON", "DYNASTY", "NORP", "CONCEPT", "GPE", "WEAPON", "SACRED_TEXT", "EVENT"]

# The 18 canonical parvas -> which half each provided file belongs to is derived
# from the filenames actually present; the complement is the intended hold-out.
CANON = {
    "adi": "01_Adi", "sabha": "02_Sabha", "aranyaka": "03_Aranyaka", "vana": "03_Aranyaka",
    "virata": "04_Virata", "udoyoga": "05_Udyoga", "udyoga": "05_Udyoga", "bhishma": "06_Bhishma",
    "drona": "07_Drona", "karna": "08_Karna", "shalya": "09_Shalya", "sauptika": "10_Sauptika",
    "souptika": "10_Sauptika", "stri": "11_Stri", "shanti": "12_Shanti", "anushasana": "13_Anushasana",
    "ashvamedhika": "14_Ashvamedhika", "ashramavasika": "15_Ashramavasika",
    "mausala": "16_Mausala", "mahaprasthanika": "17_Mahaprasthanika", "svargarohana": "18_Svargarohana",
}


def parva_of(filename):
    key = os.path.basename(filename).lower()
    for frag, canon in CANON.items():
        if frag in key:
            return canon
    return os.path.splitext(os.path.basename(filename))[0]


def normalize_tag(raw):
    """Return 'O' or 'B-LABEL'/'I-LABEL' for a valid label; 'O' otherwise."""
    t = (raw or "").strip()
    if t.lower() in O_TOKENS:
        return "O"
    if "-" in t:
        pre, lab = t.split("-", 1)
        pre, lab = pre.strip().upper(), lab.strip().upper()
        if pre in ("B", "I") and lab in LABELS:
            return f"{pre}-{lab}"
    return "O"


def read_sentences(path):
    """Yield (sent_no, [words], [norm_tags]) preserving row (token) order."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cur_no, words, tags = None, [], []
        for row in reader:
            sno = row.get("sentence_no")
            w = (row.get("word") or "")
            w = w if w.strip() else "_"          # Doc() forbids empty tokens
            tag = normalize_tag(row.get("tag"))
            if sno != cur_no and cur_no is not None:
                yield cur_no, words, tags
                words, tags = [], []
            cur_no, words = sno, words
            words.append(w); tags.append(tag)
        if words:
            yield cur_no, words, tags


def spans_from_bio(doc, tags):
    """Turn a BIO tag list into non-overlapping Spans, honouring B- boundaries.
    A stray I- (no matching open entity) is promoted to a new entity."""
    spans, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue
        lab = t[2:]
        j = i + 1
        while j < n and tags[j] == f"I-{lab}":   # B-lab or O or B/I-other ends it
            j += 1
        spans.append(Span(doc, i, j, label=lab))
        i = j
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir",
                    default="Mahabharata LLM annotated-20260729T005907Z-1-001/Mahabharata LLM annotated",
                    help="directory of per-parva BIO CSVs (the 50%% training half)")
    ap.add_argument("--out", default="train_spacy_lg", help="output directory")
    ap.add_argument("--dev-ratio", type=float, default=0.10,
                    help="fraction of each parva held back as the internal dev set")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the train/dev split")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.csv")))
    if not files:
        sys.exit(f"No CSVs found under {args.indir!r}")

    vocab = spacy.blank("en").vocab
    # examples[parva] = list of (doc, n_ents)
    by_parva = defaultdict(list)
    label_counts = Counter()
    dropped_ents = 0

    for f in files:
        parva = parva_of(f)
        for sno, words, tags in read_sentences(f):
            doc = Doc(vocab, words=words)
            spans = spans_from_bio(doc, tags)
            try:
                doc.ents = spans
            except ValueError:                    # overlap safety net
                from spacy.util import filter_spans
                kept = filter_spans(spans)
                dropped_ents += len(spans) - len(kept)
                doc.ents = kept
            for e in doc.ents:
                label_counts[e.label_] += 1
            by_parva[parva].append(doc)

    # Reproducible, parva-stratified split: shuffle each parva independently.
    rng = random.Random(args.seed)
    train_db, dev_db = DocBin(store_user_data=True), DocBin(store_user_data=True)
    split_stats = {}
    tr_lab, dv_lab = Counter(), Counter()
    for parva in sorted(by_parva):
        docs = by_parva[parva][:]
        rng.shuffle(docs)
        n_dev = max(1, round(len(docs) * args.dev_ratio)) if len(docs) > 1 else 0
        dev_docs, train_docs = docs[:n_dev], docs[n_dev:]
        for d in train_docs:
            train_db.add(d); tr_lab.update(e.label_ for e in d.ents)
        for d in dev_docs:
            dev_db.add(d); dv_lab.update(e.label_ for e in d.ents)
        split_stats[parva] = {"sentences": len(docs), "train": len(train_docs), "dev": len(dev_docs)}

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    train_path = os.path.join(args.out, "data", "train.spacy")
    dev_path = os.path.join(args.out, "data", "dev.spacy")
    train_db.to_disk(train_path)
    dev_db.to_disk(dev_path)

    train_parvas = sorted(by_parva)
    holdout_parvas = sorted(set(CANON.values()) - set(train_parvas))
    manifest = {
        "description": "spaCy NER training data built from the 50% LLM-annotated Mahabharata half.",
        "source_files": [os.path.basename(f) for f in files],
        "input_dir": args.indir,
        "labels": LABELS,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "split_method": (
            "Whole parvas were assigned to the training half upstream (these 9 files). "
            "For training mechanics only, each parva's sentences are shuffled with the fixed "
            "seed and the first dev_ratio fraction becomes the internal dev set; the rest is train. "
            "The 50% hold-out (the complementary 9 parvas) is NOT read by this script."
        ),
        "train_parvas": train_parvas,
        "holdout_parvas_intended": holdout_parvas,
        "totals": {
            "sentences": sum(s["sentences"] for s in split_stats.values()),
            "train_sentences": sum(s["train"] for s in split_stats.values()),
            "dev_sentences": sum(s["dev"] for s in split_stats.values()),
            "entities": sum(label_counts.values()),
            "overlapping_entities_dropped": dropped_ents,
        },
        "entities_per_label": dict(label_counts.most_common()),
        "train_entities_per_label": dict(tr_lab.most_common()),
        "dev_entities_per_label": dict(dv_lab.most_common()),
        "per_parva": split_stats,
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"Parvas (training half): {len(train_parvas)} -> {train_parvas}")
    print(f"Sentences: {manifest['totals']['sentences']}  "
          f"train={manifest['totals']['train_sentences']}  dev={manifest['totals']['dev_sentences']}")
    print(f"Entities: {manifest['totals']['entities']}  (dropped overlaps: {dropped_ents})")
    print(f"Per label: {dict(label_counts.most_common())}")
    print(f"Wrote {train_path}, {dev_path}, and split_manifest.json")


if __name__ == "__main__":
    main()
