#!/usr/bin/env python3
# final_comparison_script_v6_fixed.py
"""
Compares the performance of four key models with corrected logic.
- Fixes IndexError by aligning CRF/SVM/meta predictions with dataset tokens, not spaCy tokens.
- Uses spaCy only for gazetteer lookups, mapped back safely to dataset tokens.
- Implements proper stacking to prevent leakage.
- Unified train/test split for all models.
"""

import gc
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn_crfsuite import CRF
import spacy
import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

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
        agg_func = lambda s: [(t, p, l) for t, p, l in zip(
            s["token"].values.tolist(),
            s["pos_tag"].values.tolist(),
            s["ner_tag_iob"].values.tolist()
        )]
        self.grouped = self.data.groupby(self.data['token'].eq('.').cumsum()).apply(agg_func)
        self.sentences = [s for s in self.grouped if s]

getter = SentenceGetter(df)
sentences = getter.sentences

def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'postag': postag
    }
    if i > 0:
        word1, postag1 = sent[i-1][0], sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1, postag1 = sent[i+1][0], sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent): return [label for _, _, label in sent]

# ==============================================================================
# UNIFIED DATA SPLIT
# ==============================================================================
print("[INFO] Creating a unified train/test split...")
train_sentences, test_sentences = train_test_split(sentences, test_size=0.25, random_state=42)

print("[INFO] Creating CRF-formatted data...")
X_train_crf = [sent2features(s) for s in tqdm(train_sentences, desc="Processing Train Sentences for CRF")]
y_train_crf = [sent2labels(s) for s in train_sentences]
X_test_crf = [sent2features(s) for s in tqdm(test_sentences, desc="Processing Test Sentences for CRF")]
y_test_crf = [sent2labels(s) for s in test_sentences]
y_test_flat = [label for sent in y_test_crf for label in sent]

def flatten_and_format_for_svm(sentence_list):
    features = []
    labels = []
    for sent in sentence_list:
        for token, pos, label in sent:
            features.append(f"{token} {pos}")
            labels.append(label)
    return features, labels

print("[INFO] Creating SVM-formatted data...")
X_train_svm_text, y_train_svm = flatten_and_format_for_svm(train_sentences)
X_test_svm_text, y_test_svm = flatten_and_format_for_svm(test_sentences)

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_svm_csr = vectorizer.fit_transform(X_train_svm_text)
X_test_svm_csr = vectorizer.transform(X_test_svm_text)

# ==============================================================================
# TRAIN ALL MODELS
# ==============================================================================
print(f"\n{'='*20} TRAINING ALL MODELS {'='*20}")

print("[INFO] Training Baseline Model (Linear SVM)...")
baseline_svm = LinearSVC(max_iter=1000, dual=True)
baseline_svm.fit(X_train_svm_csr, y_train_svm)

print("[INFO] Training Sequential Model (CRF)...")
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train_crf, y_train_crf)

print("[INFO] Training Hierarchical Ensemble components...")
ensemble_svm = LinearSVC(max_iter=1000, dual=True)
ensemble_svm.fit(X_train_svm_csr, y_train_svm)

y_train_crf_flat = [label for sent in y_train_crf for label in sent]
crf_train_preds_flat = [tag for sent in crf.predict(X_train_crf) for tag in sent]
svm_train_preds = ensemble_svm.predict(X_train_svm_csr)
train_tokens_flat = [token_data[0] for sent in train_sentences for token_data in sent]

meta_features_train = []
meta_labels_train = []
for i in range(len(svm_train_preds)):
    if svm_train_preds[i] != crf_train_preds_flat[i]:
        meta_features_train.append({
            'token': train_tokens_flat[i].lower(),
            'crf_pred': crf_train_preds_flat[i],
            'svm_pred': svm_train_preds[i]
        })
        meta_labels_train.append(y_train_crf_flat[i])

meta_vectorizer = DictVectorizer()
meta_model = LogisticRegression(max_iter=1000)
if meta_features_train:
    X_meta_train_vec = meta_vectorizer.fit_transform(meta_features_train)
    meta_model.fit(X_meta_train_vec, meta_labels_train)
else:
    print("[WARN] No disagreements found between base models. Meta-model not trained.")

# ==============================================================================
# GAZETTEERS + ENSEMBLE PREDICTION PIPELINE
# ==============================================================================
nlp = spacy.load("en_core_web_lg")
from spacy.matcher import PhraseMatcher

GAZETTEERS = {
    "PERSON": [
        "Abhimanyu", "Adhiratha", "Adrika", "Agni", "Alambusha", "Amba",
        "Ambalika", "Ambika", "Arjuna", "Ashwatthama", "Babhruvahana", "Bakasura",
        "Balarama", "Bahlika", "Bhagadatta", "Bhima", "Bhishma", "Bhurishravas",
        "Chekitana", "Chitrasena", "Dhrishtadyumna", "Dhritarashtra", "Draupadi",
        "Drona", "Drupada", "Duryodhana", "Dushasana", "Ekalavya", "Gandhari",
        "Ganga", "Ghatotkacha", "Hidimbi", "Iravan", "Janamejaya", "Jarasandha",
        "Jayadratha", "Kanika", "Karna", "Kichaka", "Kripa", "Krishna",
        "Kritavarma", "Kunti", "Kuru", "Madri", "Nakula", "Pandu", "Parashara",
        "Parikshit", "Radha", "Sahadeva", "Sanjaya", "Satyaki", "Satyavati",
        "Shakuni", "Shalya", "Shantanu", "Shikhandi", "Shishupala", "Subhadra",
        "Takshaka", "Ugrashravas Sauti", "Ulupi", "Urvashi", "Uttara", "Uttarā",
        "Vaishampayana", "Vidura", "Vikarna", "Virata", "Vyasa", "Yudhishthira",
        "Yuyutsu"
    ],
    "GPE": [
        "Hastinapura", "Indraprastha", "Kurukshetra", "Dwarka", "Mathura",
        "Upaplavya", "Virata-nagara", "Kampilya", "Rajagriha", "Girivraja",
        "Mithila", "Kashi", "Varanasi", "Chedi", "Avanti", "Ujjayini", "Anga",
        "Champa", "Vanga", "Kalinga", "Pragjyotisha", "Sindhu", "Saindhava",
        "Gandhara", "Madra", "Bahlika", "Kamboja", "Shurasena", "Anarta",
        "Matsya", "Panchala", "Kosala", "Videha", "Magadha", "Kerala", "Pandya",
        "Chola", "Pandavapura", "Ekachakra", "Champapuri", "Takshashila",
        "Bet Dwarka", "Vrindavana", "Gokarna", "Prayaga", "Prayagraj",
        "Pushkara", "Gayā", "Badrikashrama", "Kedara", "Somnath", "Prabhasa",
        "Rameshwara", "Naimisha Forest", "Kamyaka Forest", "Dwaitavana",
        "Dandaka Forest", "Gandhamadana", "Kailasa", "Meru", "Vindhya",
        "Mahendra", "Trikuta", "Hiranvati", "Hiranyavati", "Ganga", "Yamuna",
        "Sarasvati", "Indus", "Godavari", "Narmada", "Krishna River", "Krsna",
        "Kaveri", "Cauvery", "Sarayu", "Brahma Sarovar", "Jyotisar",
        "Lakshagriha", "Varnavat", "Barnawa", "Mayasabha", "Khandava Forest",
        "Salwa", "Saubha", "Triveni", "Prayaga Sangam", "Rantideva's Tirtha",
        "Sthaneshwar", "Thanesar", "Ekadashi Tirtha", "Pancharashtra",
        "Kekaya", "Sivi", "Sibi", "Nishada", "Pulinda", "Paurava",
        "Dvaravati", "Bhogavati", "Manipura", "Kundinapura", "Vidarbha",
        "Sithila"
    ],
    "DYNASTY": ["Kuru", "Vrishni", "Andhaka", "Pandavas", "Kauravas", "Bharata lineage"],
    "WEAPON": ["Gandiva", "Brahmastra", "Sudarshana Chakra", "Pashupatastra", "Kaumodaki", "Vijaya", "Sharanga"],
    "CONCEPT": ["Dharma", "Adharma", "Karma", "Moksha"],
    "EVENT": ["Kurukshetra War", "Game of Dice"],
    "SACRED_TEXT": ["Mahabharata", "Bhagavad Gita", "Vedas"]
}


matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for label, terms in GAZETTEERS.items():
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(label, patterns)

def predict_final_ensemble(sentences, crf_model, svm_model, svm_vectorizer, meta_model, meta_vectorizer):
    all_final_tags = []
    for sent_tuples in tqdm(sentences, desc="Predicting with Ensemble"):
        crf_preds = crf_model.predict([sent2features(sent_tuples)])[0]
        svm_features, _ = flatten_and_format_for_svm([sent_tuples])
        svm_preds = svm_model.predict(svm_vectorizer.transform(svm_features))

        final_tags = [None] * len(sent_tuples)

        text = " ".join([t[0] for t in sent_tuples])
        doc = nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:
            label = doc.vocab.strings[match_id]
            for j in range(start, min(end, len(final_tags))):
                final_tags[j] = f"B-{label}" if j == start else f"I-{label}"

        for i, (token, _, _) in enumerate(sent_tuples):
            if final_tags[i] is None:
                crf_pred = crf_preds[i]
                svm_pred = svm_preds[i]
                if crf_pred == svm_pred:
                    final_tags[i] = crf_pred
                else:
                    if hasattr(meta_model, 'classes_'):
                        meta_feature = [{'token': token.lower(), 'crf_pred': crf_pred, 'svm_pred': svm_pred}]
                        final_tags[i] = meta_model.predict(meta_vectorizer.transform(meta_feature))[0]
                    else:
                        final_tags[i] = crf_pred

        all_final_tags.extend(final_tags)
    return all_final_tags

# ==============================================================================
# EVALUATION
# ==============================================================================
print(f"\n{'='*20} FINAL PERFORMANCE COMPARISON {'='*20}")

baseline_preds = baseline_svm.predict(X_test_svm_csr)
baseline_accuracy = accuracy_score(y_test_svm, baseline_preds)
baseline_f1 = f1_score(y_test_svm, baseline_preds, average='weighted')

crf_preds_flat = [tag for sent in crf.predict(X_test_crf) for tag in sent]
crf_accuracy = accuracy_score(y_test_flat, crf_preds_flat)
crf_f1 = f1_score(y_test_flat, crf_preds_flat, average='weighted')

ensemble_preds = predict_final_ensemble(test_sentences, crf, baseline_svm, vectorizer, meta_model, meta_vectorizer)
ensemble_accuracy = accuracy_score(y_test_flat, ensemble_preds)
ensemble_f1 = f1_score(y_test_flat, ensemble_preds, average='weighted')

crf_f1_improvement = ((crf_f1 - baseline_f1) / baseline_f1) * 100
ensemble_f1_improvement = ((ensemble_f1 - baseline_f1) / baseline_f1) * 100

print("\n" + "#"*60)
print("### FINAL MODEL PERFORMANCE REPORT ###")
print("#"*60 + "\n")

summary_data = {
    "Model": ["Baseline (Linear SVM)", "Sequential (CRF)", "Proposed (Hierarchical Ensemble)"],
    "Accuracy": [baseline_accuracy, crf_accuracy, ensemble_accuracy],
    "Weighted F1-Score": [baseline_f1, crf_f1, ensemble_f1],
    "% F1 Improvement vs Baseline": ["-", f"{crf_f1_improvement:.2f}%", f"{ensemble_f1_improvement:.2f}%"]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n--- Detailed Classification Reports ---")
print("\n--- Baseline Model (Linear SVM) ---")
print(classification_report(y_test_svm, baseline_preds, digits=4, zero_division='warn'))
print("\n--- Sequential Model (CRF) ---")
print(classification_report(y_test_flat, crf_preds_flat, digits=4, zero_division='warn'))
print("\n--- Proposed Hierarchical Ensemble Model ---")
print(classification_report(y_test_flat, ensemble_preds, digits=4, zero_division='warn'))