"""
make_parva_input.py
===================
Build the per-word input CSV that tag_csv.py expects, for ONE parva.

    python make_parva_input.py Virata           # -> 04_Virata_input.csv
    python make_parva_input.py Bhishma           # -> 06_Bhishma_input.csv
    python make_parva_input.py Virata my_out.csv # custom output name

Source : latest mahabharata_ner_by_sentence_master*.csv (rows whose `parva` matches).
Output : <parva_code>_input.csv  with columns  sentence_no, sentence, word, POS, tag
         (tag blank — tag_csv.py fills it). Tokenisation + POS from spaCy en_core_web_sm.
"""
import csv
import glob
import os
import sys

import pandas as pd


def load_parva(match):
    masters = glob.glob("mahabharata_ner_by_sentence_master*.csv")
    if not masters:
        sys.exit("No mahabharata_ner_by_sentence_master*.csv found in this folder.")
    src = max(masters, key=os.path.getmtime)
    df = pd.read_csv(src)
    sub = df[df["parva"].str.contains(match, case=False, na=False)].copy()
    if sub.empty:
        avail = sorted(df["parva"].unique())
        sys.exit(f"No parva matching {match!r}. Available: {avail}")
    code = sub["parva"].iloc[0]          # e.g. "04_Virata"
    print(f"Source: {src}  ->  {len(sub)} sentences for parva {code!r}")
    return sub, code


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python make_parva_input.py <parva-name> [output.csv]\n"
                 "  e.g. python make_parva_input.py Udyoga")
    match = sys.argv[1]

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    except Exception as e:
        sys.exit(f"spaCy en_core_web_sm needed for tokenisation/POS: {e}")

    sub, code = load_parva(match)
    out = sys.argv[2] if len(sys.argv) > 2 else f"{code}_input.csv"

    n_words = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_no", "sentence", "word", "POS", "tag"])
        for doc, sno in zip(
                nlp.pipe(sub["sentence"].astype(str), batch_size=64),
                sub["global_sent_id"].astype(str)):
            for t in doc:
                if t.is_space:
                    continue
                w.writerow([sno, doc.text, t.text, t.tag_ or t.pos_ or "X", ""])
                n_words += 1
    print(f"Wrote {out}: {n_words} word rows.")
    print(f"Next:  export GEMINI_API_KEY=...  &&  "
          f"python tag_csv.py {out} {code}_tagged.csv")


if __name__ == "__main__":
    main()
