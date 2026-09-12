#!/usr/bin/env python3
# final_ml_pinnacle_script_streamlined.py
"""
A streamlined script to train and evaluate ONLY the novel Stacking Ensemble model.
- Assumes baseline model scores are already known and provided by the user.
- Trains the necessary components (CRF, SVM, and XGBoost Meta-Model).
- Evaluates the final ensemble and compares it to the known baseline scores.
"""
import gc
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn_crfsuite import CRF
import xgboost as xgb
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. INPUT YOUR KNOWN BASELINE SCORES HERE
# ==============================================================================
# Replace these placeholder values with the results from your previous runs.
# Using the CRF as the strongest baseline.
KNOWN_BASELINE_SCORES = {
    "model_name": "Baseline (CRF)",
    "accuracy": 0.9911,  # <-- Enter your CRF accuracy here
    "f1_score": 0.9907   # <-- Enter your CRF F1-score here
}

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

print("\n[INFO] Loading and preparing dataset...")
try:
    df = pd.read_csv("/kaggle/input/mahabharata225/mahabharata_ml_dataset.csv")
except FileNotFoundError:
    print("\n❌ ERROR: 'mahabharata_ml_dataset.csv' not found.")
    exit()

df.dropna(subset=['token', 'ner_tag_iob'], inplace=True)
FILTER_THRESHOLD = 60
tag_counts = df['ner_tag_iob'].value_counts()
rare_tags = tag_counts[tag_counts < FILTER_THRESHOLD].index
df = df[~df['ner_tag_iob'].isin(rare_tags)]

class SentenceGetter(object):
    def __init__(self, data):
        self.data = data
        agg_func = lambda s: [(t, p, l) for t, p, l in zip(s["token"].values.tolist(), s["pos_tag"].values.tolist(), s["ner_tag_iob"].values.tolist())]
        self.grouped = self.data.groupby(self.data['token'].eq('.').cumsum()).apply(agg_func)
        self.sentences = [s for s in self.grouped if s]

getter = SentenceGetter(df)
sentences = getter.sentences

def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    features = {'bias': 1.0, 'word.lower()': word.lower(), 'word.isupper()': word.isupper(), 'word.istitle()': word.istitle(), 'postag': postag}
    if i > 0:
        word1, postag1 = sent[i-1][0], sent[i-1][1]
        features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle(), '-1:postag': postag1})
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1, postag1 = sent[i+1][0], sent[i+1][1]
        features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle(), '+1:postag': postag1})
    else:
        features['EOS'] = True
    return features

def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent): return [label for _, _, label in sent]

# Create a unified train/test split
train_sentences, test_sentences = train_test_split(sentences, test_size=0.25, random_state=42)

# ==============================================================================
# TRAIN THE BASE MODELS FOR THE ENSEMBLE
# ==============================================================================
print(f"\n{'='*20} TRAINING ENSEMBLE'S BASE MODELS {'='*20}")

# --- Prepare data for CRF ---
X_train_crf = [sent2features(s) for s in train_sentences]
y_train_crf = [sent2labels(s) for s in train_sentences]

# ✅ Added missing test set
X_test_crf = [sent2features(s) for s in test_sentences]
y_test_crf = [sent2labels(s) for s in test_sentences]

# --- Train Base Model 1: CRF ---
print("[INFO] Training CRF model...")
crf = CRF(
    algorithm='lbfgs', 
    c1=0.1, c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train_crf, y_train_crf)

# --- Train Base Model 1: CRF ---
print("[INFO] Training CRF model...")
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train_crf, y_train_crf)

# --- Prepare data for SVM ---
def flatten_and_format_for_svm(sentence_list):
    features, labels = [], []
    for sent in sentence_list:
        for token, pos, label in sent:
            features.append(f"{token} {pos}")
            labels.append(label)
    return features, labels

X_train_svm_text, y_train_svm = flatten_and_format_for_svm(train_sentences)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_svm_csr = vectorizer.fit_transform(X_train_svm_text)

# --- Train Base Model 2: Linear SVM ---
print("[INFO] Training Linear SVM model...")
svm = LinearSVC(max_iter=1000, dual=True)
svm.fit(X_train_svm_csr, y_train_svm)

# ==============================================================================
# TRAIN THE META-MODEL (STACKING)
# ==============================================================================
print(f"\n{'='*20} TRAINING STACKING META-MODEL {'='*20}")

# --- Create "Enriched" Meta-Features ---
print("[INFO] Creating rich meta-features for the meta-model...")
crf_train_preds_flat = [tag for sent in crf.predict(X_train_crf) for tag in sent]
svm_train_preds = svm.predict(X_train_svm_csr)
y_train_flat = [label for sent in y_train_crf for label in sent]
all_train_features = [feature for sent in X_train_crf for feature in sent]

meta_features_train = []
for i in tqdm(range(len(y_train_flat)), desc="Generating Meta-Features"):
    new_features = all_train_features[i].copy()
    new_features['crf_pred'] = crf_train_preds_flat[i]
    new_features['svm_pred'] = svm_train_preds[i]
    meta_features_train.append(new_features)

# --- Train the XGBoost Meta-Model ---
if meta_features_train:
    print(f"[INFO] Training XGBoost meta-model on {len(meta_features_train)} samples...")
    meta_vectorizer = DictVectorizer()
    X_meta_train_vec = meta_vectorizer.fit_transform(meta_features_train)

    le_meta = LabelEncoder()
    y_meta_train_enc = le_meta.fit_transform(y_train_flat)

    meta_model = xgb.XGBClassifier(
        objective='multi:softmax', use_label_encoder=False,
        eval_metric='mlogloss', n_estimators=100, random_state=42,
        tree_method='gpu_hist' # Use GPU for training
    )
    meta_model.fit(X_meta_train_vec, y_meta_train_enc)
else:
    print("[WARN] Could not create meta-features. Meta-model was not trained.")
    meta_model = None

# ==============================================================================
# DEFINE AND EVALUATE THE FINAL ENSEMBLE
# ==============================================================================
print(f"\n{'='*20} EVALUATING THE FINAL STACKING ENSEMBLE {'='*20}")

def predict_stacking_ensemble(sentences, crf_model, svm_model, svm_vectorizer, meta_model, meta_vectorizer):
    # Get base predictions for the entire test set
    print("[INFO] Getting predictions from base models...")
    crf_preds_flat = [tag for sent in tqdm(crf_model.predict([sent2features(s) for s in sentences]), desc="CRF Predicting") for tag in sent]
    svm_features_text, _ = flatten_and_format_for_svm(sentences)
    svm_preds_flat = svm_model.predict(svm_vectorizer.transform(svm_features_text))
    
    # Create meta-features for the test set
    print("[INFO] Creating meta-features for test set...")
    all_test_features = [feature for sent in [sent2features(s) for s in sentences] for feature in sent]
    meta_features_test = []
    for i in tqdm(range(len(all_test_features)), desc="Generating Meta-Features for Test"):
        new_features = all_test_features[i].copy()
        new_features['crf_pred'] = crf_preds_flat[i]
        new_features['svm_pred'] = svm_preds_flat[i]
        meta_features_test.append(new_features)
        
    # Get final predictions from the meta-model
    print("[INFO] Getting final predictions from meta-model...")
    X_meta_test_vec = meta_vectorizer.transform(meta_features_test)
    final_preds_enc = meta_model.predict(X_meta_test_vec)
    final_predictions = le_meta.inverse_transform(final_preds_enc)
    
    return final_predictions

# --- Get predictions from the ensemble model ---
y_test_flat = [label for sent in y_test_crf for label in sent]

if meta_model:
    ensemble_preds_flat = predict_stacking_ensemble(test_sentences, crf, svm, vectorizer, meta_model, meta_vectorizer)
    
    # --- Calculate metrics ---
    ensemble_accuracy = accuracy_score(y_test_flat, ensemble_preds_flat)
    ensemble_f1 = f1_score(y_test_flat, ensemble_preds_flat, average='weighted')
    ensemble_f1_improvement = ((ensemble_f1 - KNOWN_BASELINE_SCORES["f1_score"]) / KNOWN_BASELINE_SCORES["f1_score"]) * 100

    # --- Generate Final Report ---
    print("\n" + "#"*60)
    print("### FINAL MODEL PERFORMANCE REPORT ###")
    print("#"*60 + "\n")

    summary_data = {
        "Model": [KNOWN_BASELINE_SCORES["model_name"], "Proposed (Stacking Ensemble)"],
        "Accuracy": [KNOWN_BASELINE_SCORES["accuracy"], ensemble_accuracy],
        "Weighted F1-Score": [KNOWN_BASELINE_SCORES["f1_score"], ensemble_f1],
        "% F1 Improvement vs Baseline": ["-", f"{ensemble_f1_improvement:.2f}%"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n--- Detailed Classification Report for Proposed Ensemble Model ---")
    print(classification_report(y_test_flat, ensemble_preds_flat, digits=4, zero_division='warn'))

else:
    print("Could not evaluate ensemble as meta-model was not trained.")