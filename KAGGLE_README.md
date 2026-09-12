# Running the Mahabharata NER pipeline on Kaggle

The script `mahabharata_ner_final.py` needs **no code changes** to run on Kaggle —
it already auto-discovers the corpus (`/kaggle/input/*/*.pdf`) and writes outputs
under `/kaggle/working`. Use the notebook `mahabharata_ner_kaggle.ipynb`.

## Steps
1. **Create a Kaggle dataset** containing both files:
   - `TheMahabharata.pdf`
   - `mahabharata_ner_final.py`
   (Or two datasets — the notebook searches all of `/kaggle/input/`.)
2. **New Notebook** → upload `mahabharata_ner_kaggle.ipynb` (or paste the cells).
3. **Add Data** → attach the dataset(s) from step 1.
4. **Settings → Internet → ON** (required for pip installs + spaCy model download).
5. **Settings → Accelerator → None** — this pipeline is CPU-bound
   (CRF + scikit-learn). A GPU gives no benefit here.
6. Run all cells. For the long full run, use **Save Version → Save & Run All
   (Commit)** so it runs headless within Kaggle's 12-hour limit.

## What gets installed (rest ships with the Kaggle image)
`sklearn-crfsuite`, `seqeval`, `lime`, `pymupdf`, and the `en_core_web_sm` spaCy model.
Already present on Kaggle: numpy, pandas, scipy, scikit-learn, matplotlib, spacy,
xgboost, shap.

## Smoke test first (recommended)
In cell 2 of the notebook, set `os.environ["NER_FAST"] = "1"` to run the entire
pipeline on a 250-page slice in ~3 minutes and confirm everything works, then
remove it for the full corpus run.

## Outputs
Everything lands in `/kaggle/working/outputs/run_<timestamp>/`:
- `tables/model_comparison.csv` — the leaderboard (every model)
- `tables/per_class_<Model>.csv` — per-class P/R/F1 for each model
- `tables/vectorizer_comparison.csv` — feature-representation comparison
- `figures/*.png`, `explanations/*` (SHAP/LIME), `models/*`, `manifest.json`

Cell 6 zips the run directory to `/kaggle/working/mahabharata_outputs.zip`.
