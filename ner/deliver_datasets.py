#!/usr/bin/env python3
"""Package two parallel, comparable datasets for delivery:

  delivery/spacy_annotated/<NN_Parva>.csv       tag = trained spaCy-lg prediction
  delivery/llm_annotated_baseline/<NN_Parva>.csv tag = LLM silver (baseline)

Both use the same clean schema: sentence_no, sentence, word, tag  (tag normalized
to O / B-LABEL / I-LABEL). Same tokenization and row order in both, so a row-wise
diff of the `tag` column is exactly spaCy-vs-LLM. Then zips the whole folder.
"""
import csv, os, shutil

from predict_all_parvas import PARVAS, LLM_DIR, norm  # reuse the canonical mapping

PRED_DIR = "train_delta/predictions"
DELIV = "train_delta/delivery"
SP = os.path.join(DELIV, "spacy_annotated")
LM = os.path.join(DELIV, "llm_annotated_baseline")


def read_llm(path):
    """rows of (sentence_no, sentence, word, llm_tag_normalized) in file order."""
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield (row.get("sentence_no"), (row.get("sentence") or ""),
                   ((row.get("word") or "").strip() or "_"), norm(row.get("tag")))


def read_pred(path):
    with open(path, encoding="utf-8") as f:
        return [r["pred_tag"] for r in csv.DictReader(f)]


def main():
    for d in (SP, LM):
        os.makedirs(d, exist_ok=True)
    hdr = ["sentence_no", "sentence", "word", "tag"]
    for name, (lf, hf, is_train) in PARVAS.items():
        llm_rows = list(read_llm(os.path.join(LLM_DIR, lf)))
        preds = read_pred(os.path.join(PRED_DIR, f"{name}.csv"))
        assert len(preds) == len(llm_rows), f"{name}: {len(preds)} preds vs {len(llm_rows)} rows"

        with open(os.path.join(LM, f"{name}.csv"), "w", newline="", encoding="utf-8") as fl, \
             open(os.path.join(SP, f"{name}.csv"), "w", newline="", encoding="utf-8") as fs:
            wl, ws = csv.writer(fl), csv.writer(fs)
            wl.writerow(hdr); ws.writerow(hdr)
            for (sn, sent, word, llm_tag), pred in zip(llm_rows, preds):
                wl.writerow([sn, sent, word, llm_tag])
                ws.writerow([sn, sent, word, pred])
        print(f"  {name:<26} rows={len(preds):>7}  ({'train' if is_train else 'test'})")

    readme = os.path.join(DELIV, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Mahabharata NER — two parallel annotated datasets (18 parvas each)\n"
            "==================================================================\n\n"
            "spacy_annotated/           tag = trained spaCy en_core_web_lg prediction\n"
            "llm_annotated_baseline/    tag = LLM silver annotation (the baseline)\n\n"
            "Schema (both): sentence_no, sentence, word, tag   (tag = O / B-LABEL / I-LABEL)\n"
            "Same tokenization and row order in both folders, so a row-wise diff of the\n"
            "'tag' column is exactly spaCy-vs-LLM for that word.\n\n"
            "Files 01..05,07,11..14 = the 10 parvas the spaCy model was TRAINED on\n"
            "(its labels there are in-sample / memorized). Files 06,08,09,10,15,16,17,18\n"
            "= the 8 human-checked, held-out parvas (the honest test set).\n"
            "8 entity types: PERSON, DYNASTY, NORP, CONCEPT, GPE, WEAPON, SACRED_TEXT, EVENT\n"
            "(+ rare CELESTIAL). Labels normalized to O for non-entities.\n"
        )

    zip_base = "train_delta/mahabharata_spacy_vs_llm_annotations"
    shutil.make_archive(zip_base, "zip", DELIV)
    print(f"\nfolders: {SP}/  {LM}/")
    print(f"zip:     {zip_base}.zip")


if __name__ == "__main__":
    main()
