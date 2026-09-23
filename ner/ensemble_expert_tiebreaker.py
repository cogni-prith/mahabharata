#!/usr/bin/env python3
# final_ensemble_expert_tiebreaker_v2.py
"""
Implements and evaluates the enhanced "Confidence-Gated Hierarchical Ensemble" model.
- Base Models: A high-precision gazetteer, a high-recall CRF, and a robust Linear SVM.
- Meta-Model: A powerful XGBoost classifier trained specifically on disagreements
              between the CRF and SVM to act as an expert tie-breaker.
- Compares the final ensemble performance against the standalone CRF model.
"""
import gc
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF
import xgboost as xgb
import spacy
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

# Create CRF-formatted data
X_train_crf = [sent2features(s) for s in train_sentences]
y_train_crf = [sent2labels(s) for s in train_sentences]
X_test_crf = [sent2features(s) for s in test_sentences]
y_test_crf = [sent2labels(s) for s in test_sentences]

# Create SVM-formatted data
def flatten_and_format_for_svm(sentence_list):
    features, labels = [], []
    for sent in sentence_list:
        for token, pos, label in sent:
            features.append(f"{token} {pos}")
            labels.append(label)
    return features, labels

X_train_svm_text, y_train_svm = flatten_and_format_for_svm(train_sentences)
X_test_svm_text, y_test_svm = flatten_and_format_for_svm(test_sentences)

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_svm_csr = vectorizer.fit_transform(X_train_svm_text)
X_test_svm_csr = vectorizer.transform(X_test_svm_text)


# ==============================================================================
# TRAIN THE BASE MODELS
# ==============================================================================
print(f"\n{'='*20} TRAINING BASE MODELS {'='*20}")
# --- Train Base Model 1: CRF ---
print("[INFO] Training CRF model...")
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train_crf, y_train_crf)

# --- Train Base Model 2: Linear SVM ---
print("[INFO] Training Linear SVM model...")
svm = LinearSVC(max_iter=1000, dual=True)
svm.fit(X_train_svm_csr, y_train_svm)


# ==============================================================================
# TRAIN THE META-MODEL (EXPERT TIE-BREAKER)
# ==============================================================================
print(f"\n{'='*20} TRAINING META-MODEL (EXPERT TIE-BREAKER) {'='*20}")

# --- Create Meta-Features with Rich Context from disagreements ---
print("[INFO] Creating rich meta-features for the tie-breaker...")
crf_train_preds_flat = [tag for sent in crf.predict(X_train_crf) for tag in sent]
svm_train_preds = svm.predict(X_train_svm_csr)
y_train_flat = [label for sent in y_train_crf for label in sent]
all_train_features = [feature for sent in X_train_crf for feature in sent]

meta_features_train = []
meta_labels_train = []

for i in range(len(y_train_flat)):
    if crf_train_preds_flat[i] != svm_train_preds[i]:
        # Add the original CRF features
        new_features = all_train_features[i].copy()
        # Add the conflicting predictions as new features
        new_features['crf_pred'] = crf_train_preds_flat[i]
        new_features['svm_pred'] = svm_train_preds[i]
        meta_features_train.append(new_features)
        meta_labels_train.append(y_train_flat[i])

# --- Train the XGBoost Meta-Model ---
if meta_features_train:
    print(f"[INFO] Training XGBoost meta-model on {len(meta_features_train)} disagreement cases...")
    meta_vectorizer = DictVectorizer()
    X_meta_train_vec = meta_vectorizer.fit_transform(meta_features_train)

    le_meta = LabelEncoder()
    y_meta_train_enc = le_meta.fit_transform(meta_labels_train)

    meta_model = xgb.XGBClassifier(
        objective='multi:softmax', use_label_encoder=False,
        eval_metric='mlogloss', n_estimators=100, random_state=42
    )
    meta_model.fit(X_meta_train_vec, y_meta_train_enc)
else:
    print("[WARN] No disagreements found between base models. Meta-model was not trained.")
    meta_model = None


# ==============================================================================
# DEFINE AND EVALUATE THE FINAL ENSEMBLE
# ==============================================================================
print(f"\n{'='*20} EVALUATING THE FINAL ENSEMBLE MODEL {'='*20}")
nlp = spacy.load("en_core_web_lg")
GAZETTEERS = { "PERSON": ["Arjuna", "Krishna", "Bhima", "Karna"], "GPE": ["Kurukshetra", "Hastinapura"], "WEAPON": ["Gandiva"], "DYNASTY": ["Pandavas", "Kauravas"] }
matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr="LOWER")
for label, terms in GAZETTEERS.items():
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(label, patterns)

def predict_final_ensemble(sentences, crf_model, svm_model, svm_vectorizer, meta_model, meta_vectorizer):
    final_predictions = []
    
    # Get base predictions for the entire test set
    crf_preds = crf_model.predict([sent2features(s) for s in sentences])
    svm_features_text, _ = flatten_and_format_for_svm(sentences)
    svm_preds = svm_model.predict(svm_vectorizer.transform(svm_features_text))
    
    token_idx = 0
    for sent_idx, sent_tuples in enumerate(tqdm(sentences, desc="Predicting with Ensemble")):
        sent_features = sent2features(sent_tuples)
        doc = nlp(" ".join([t[0] for t in sent_tuples]))
        
        # Stage 1: Gazetteer Gate
        gazetteer_tags = ['O'] * len(doc)
        matches = matcher(doc)
        for match_id, start, end in matches:
            label = doc.vocab.strings[match_id]
            gazetteer_tags[start] = f"B-{label}"
            for i in range(start + 1, end): gazetteer_tags[i] = f"I-{label}"
            
        sent_final_tags = []
        for i in range(len(sent_tuples)):
            if gazetteer_tags[i] != 'O':
                sent_final_tags.append(gazetteer_tags[i])
            else:
                # Stage 2: Statistical Consensus
                crf_pred = crf_preds[sent_idx][i]
                svm_pred = svm_preds[token_idx]
                if crf_pred == svm_pred:
                    sent_final_tags.append(crf_pred)
                else:
                    # Stage 3: Meta-Learner Tie-Breaker
                    if meta_model:
                        meta_feature = sent_features[i].copy()
                        meta_feature['crf_pred'] = crf_pred
                        meta_feature['svm_pred'] = svm_pred
                        
                        meta_feature_vec = meta_vectorizer.transform([meta_feature])
                        correction_enc = meta_model.predict(meta_feature_vec)
                        correction = le_meta.inverse_transform(correction_enc)
                        sent_final_tags.append(correction[0])
                    else: # Fallback if meta-model wasn't trained
                        sent_final_tags.append(crf_pred) # Trust the CRF
            token_idx += 1
        final_predictions.extend(sent_final_tags)
        
    return final_predictions

# --- Get predictions from both models ---
y_test_flat = [label for sent in y_test_crf for label in sent]
crf_preds_flat = [tag for sent in crf.predict(X_test_crf) for tag in sent]
ensemble_preds_flat = predict_final_ensemble(test_sentences, crf, svm, vectorizer, meta_model, meta_vectorizer)

print("\n--- Base CRF Model Performance ---")
print(classification_report(y_test_flat, crf_preds_flat, digits=4, zero_division='warn'))

print("\n--- Final Enhanced Ensemble Model Performance ---")
print(classification_report(y_test_flat, ensemble_preds_flat, digits=4, zero_division='warn'))
[INFO] Loading and preparing dataset...

==================== TRAINING BASE MODELS ====================
[INFO] Training CRF model...
[INFO] Training Linear SVM model...

==================== TRAINING META-MODEL (EXPERT TIE-BREAKER) ====================
[INFO] Creating rich meta-features for the tie-breaker...
[INFO] Training XGBoost meta-model on 22684 disagreement cases...

==================== EVALUATING THE FINAL ENSEMBLE MODEL ====================
--- Base CRF Model Performance ---
               precision    recall  f1-score   support

   B-CARDINAL     0.9439    0.9125    0.9280      3303
    B-CONCEPT     1.0000    0.9981    0.9990      1580
       B-DATE     0.9257    0.8897    0.9073       798
    B-DYNASTY     0.9964    1.0000    0.9982      1385
      B-EVENT     0.7500    0.3947    0.5172        38
        B-FAC     0.9500    0.3115    0.4691        61
        B-GPE     0.8811    0.7609    0.8166      2689
        B-LAW     0.9643    0.8182    0.8852        66
        B-LOC     0.7602    0.5508    0.6388       236
       B-NORP     0.6962    0.5877    0.6373       308
    B-ORDINAL     0.9635    0.9635    0.9635       301
        B-ORG     0.8090    0.7251    0.7648      3154
     B-PERSON     0.8996    0.9437    0.9211     16921
    B-PRODUCT     0.6796    0.6685    0.6740       184
   B-QUANTITY     0.8750    0.6364    0.7368        22
B-SACRED_TEXT     1.0000    1.0000    1.0000       259
       B-TIME     0.8382    0.6867    0.7550       166
     B-WEAPON     1.0000    0.9916    0.9958       119
B-WORK_OF_ART     0.5556    0.2174    0.3125        92
   I-CARDINAL     0.9486    0.9306    0.9395       793
       I-DATE     0.9003    0.8728    0.8864       983
    I-DYNASTY     0.9939    1.0000    0.9969       813
      I-EVENT     1.0000    0.8889    0.9412        18
        I-GPE     0.8333    0.3125    0.4545        48
        I-LAW     0.9362    0.8073    0.8670       109
        I-LOC     0.4222    0.5278    0.4691        36
        I-ORG     0.5331    0.3341    0.4108       434
     I-PERSON     0.7421    0.5009    0.5981       563
   I-QUANTITY     0.7600    0.6333    0.6909        30
       I-TIME     0.8036    0.5882    0.6792       153
I-WORK_OF_ART     0.6667    0.2124    0.3221       113
            O     0.9966    0.9981    0.9974    578351

     accuracy                         0.9911    614126
    macro avg     0.8445    0.7082    0.7554    614126
 weighted avg     0.9907    0.9911    0.9907    614126


--- Final Enhanced Ensemble Model Performance ---
               precision    recall  f1-score   support

   B-CARDINAL     0.9551    0.9071    0.9304      3303
    B-CONCEPT     1.0000    0.9981    0.9990      1580
       B-DATE     0.9266    0.8860    0.9058       798
    B-DYNASTY     0.9964    1.0000    0.9982      1385
      B-EVENT     0.7692    0.5263    0.6250        38
        B-FAC     0.9091    0.3279    0.4819        61
        B-GPE     0.8801    0.7616    0.8166      2689
        B-LAW     0.9643    0.8182    0.8852        66
        B-LOC     0.7303    0.5508    0.6280       236
       B-NORP     0.6859    0.6169    0.6496       308
    B-ORDINAL     0.9604    0.9668    0.9636       301
        B-ORG     0.8054    0.7321    0.7670      3154
     B-PERSON     0.9016    0.9436    0.9221     16921
    B-PRODUCT     0.6771    0.7065    0.6915       184
   B-QUANTITY     0.8750    0.6364    0.7368        22
B-SACRED_TEXT     1.0000    1.0000    1.0000       259
       B-TIME     0.8333    0.7229    0.7742       166
     B-WEAPON     1.0000    0.9916    0.9958       119
B-WORK_OF_ART     0.5758    0.2065    0.3040        92
   I-CARDINAL     0.9474    0.9306    0.9389       793
       I-DATE     0.8995    0.8739    0.8865       983
    I-DYNASTY     0.9939    1.0000    0.9969       813
      I-EVENT     1.0000    0.8889    0.9412        18
        I-GPE     0.7895    0.3125    0.4478        48
        I-LAW     0.9362    0.8073    0.8670       109
        I-LOC     0.4348    0.5556    0.4878        36
        I-ORG     0.5226    0.3203    0.3971       434
     I-PERSON     0.7368    0.4973    0.5938       563
   I-QUANTITY     0.7600    0.6333    0.6909        30
       I-TIME     0.8053    0.5948    0.6842       153
I-WORK_OF_ART     0.6667    0.2124    0.3221       113
            O     0.9967    0.9981    0.9974    578351

     accuracy                         0.9912    614126
    macro avg     0.8417    0.7164    0.7602    614126
 weighted avg     0.9908    0.9912    0.9908    614126

#!/usr/bin/env python3
# final_ensemble_evaluation_script.py
"""
A streamlined script to train and evaluate ONLY the novel Hierarchical Ensemble model.
- Assumes baseline and CRF model scores are already known.
- Trains the necessary components for the ensemble.
- Evaluates the final ensemble and compares it to the known baseline scores.
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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. INPUT YOUR KNOWN BASELINE SCORES HERE
# ==============================================================================
# Replace these placeholder values with the results from your previous runs.
KNOWN_BASELINE_SCORES = {
    "model_name": "Baseline (Linear SVM)",
    "accuracy": 0.983212,
    "f1_score": 0.981430
}

# **NEW**: Add the known scores for the CRF model
KNOWN_CRF_SCORES = {
    "model_name": "Sequential (CRF)",
    "accuracy": 0.991113,
    "f1_score": 0.990736
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
        agg_func = lambda s: [(t, p, l) for t, p, l in zip(s["token"].values.tolist(), s["pos_tag"].values.tolist(),
                                                           s["ner_tag_iob"].values.tolist())]
        self.grouped = self.data.groupby(self.data['token'].eq('.').cumsum()).apply(agg_func)
        self.sentences = [s for s in self.grouped if s]


getter = SentenceGetter(df)
sentences = getter.sentences


def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    features = {'bias': 1.0, 'word.lower()': word.lower(), 'word.isupper()': word.isupper(),
                'word.istitle()': word.istitle(), 'postag': postag}
    if i > 0:
        word1, postag1 = sent[i - 1][0], sent[i - 1][1]
        features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle(), '-1:postag': postag1})
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1, postag1 = sent[i + 1][0], sent[i + 1][1]
        features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle(), '+1:postag': postag1})
    else:
        features['EOS'] = True
    return features


def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent): return [label for _, _, label in sent]


print("[INFO] Creating a unified train/test split...")
train_sentences, test_sentences = train_test_split(sentences, test_size=0.25, random_state=42)
y_test_crf = [sent2labels(s) for s in test_sentences]

# ==============================================================================
# TRAIN THE HIERARCHICAL ENSEMBLE'S COMPONENTS
# ==============================================================================
print(f"\n{'=' * 20} TRAINING ENSEMBLE COMPONENTS {'=' * 20}")

# --- Create data formats for the models ---
X_train_crf = [sent2features(s) for s in train_sentences]
y_train_crf = [sent2labels(s) for s in train_sentences]


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

# --- Train Base Model 1: CRF ---
print("[INFO] Training CRF component...")
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train_crf, y_train_crf)

# --- Train Base Model 2: Linear SVM ---
print("[INFO] Training SVM component...")
svm = LinearSVC(max_iter=1000, dual=True)
svm.fit(X_train_svm_csr, y_train_svm)

# --- Train the Meta-Model (Tie-Breaker) ---
print("[INFO] Training Meta-Model component...")
y_train_crf_flat = [label for sent in y_train_crf for label in sent]
crf_train_preds_flat = [tag for sent in crf.predict(X_train_crf) for tag in sent]
svm_train_preds = svm.predict(X_train_svm_csr)
train_tokens_flat = [token_data[0] for sent in train_sentences for token_data in sent]

meta_features_train = []
meta_labels_train = []
for i in range(len(svm_train_preds)):
    if svm_train_preds[i] != crf_train_preds_flat[i]:
        meta_features_train.append({'token': train_tokens_flat[i].lower(), 'crf_pred': crf_train_preds_flat[i],
                                    'svm_pred': svm_train_preds[i]})
        meta_labels_train.append(y_train_crf_flat[i])

meta_vectorizer = DictVectorizer()
meta_model = LogisticRegression(max_iter=1000)
if meta_features_train:
    X_meta_train_vec = meta_vectorizer.fit_transform(meta_features_train)
    meta_model.fit(X_meta_train_vec, meta_labels_train)
else:
    print("[WARN] No disagreements found between base models. Meta-model was not trained.")
    meta_model = None

# ==============================================================================
# EVALUATE THE HIERARCHICAL ENSEMBLE
# ==============================================================================
print(f"\n{'=' * 20} EVALUATING THE HIERARCHICAL ENSEMBLE {'=' * 20}")

nlp = spacy.load("en_core_web_lg")
GAZETTEERS = {"PERSON": ["Arjuna", "Krishna", "Bhima", "Karna"], "GPE": ["Kurukshetra", "Hastinapura"],
              "WEAPON": ["Gandiva"], "DYNASTY": ["Pandavas", "Kauravas"]}
matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr="LOWER")
for label, terms in GAZETTEERS.items():
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(label, patterns)


def predict_final_ensemble(sentences, crf_model, svm_model, svm_vectorizer, meta_model, meta_vectorizer):
    all_final_tags = []

    # CRF + SVM predictions aligned with dataset tokens
    crf_preds = crf_model.predict([sent2features(s) for s in sentences])
    svm_features_text, _ = flatten_and_format_for_svm(sentences)
    svm_preds = svm_model.predict(svm_vectorizer.transform(svm_features_text))

    svm_idx = 0  # pointer into svm_preds (flat)

    for sent_idx, sent_tuples in enumerate(tqdm(sentences, desc="Predicting with Ensemble")):
        n_tokens = len(sent_tuples)
        final_tags = [None] * n_tokens

        # Gazetteer override via spaCy (map back to dataset tokens by exact string match)
        text = " ".join([tok for tok, _, _ in sent_tuples])
        doc = nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:
            label = doc.vocab.strings[match_id]
            span_tokens = [t.text.lower() for t in doc[start:end]]
            for i, (tok, _, _) in enumerate(sent_tuples):
                if tok.lower() in span_tokens:
                    final_tags[i] = f"B-{label}" if tok.lower() == span_tokens[0] else f"I-{label}"

        # Merge CRF + SVM + Meta-model
        for i, (tok, _, _) in enumerate(sent_tuples):
            if final_tags[i] is None:  # only if gazetteer didn't override
                crf_pred = crf_preds[sent_idx][i]
                svm_pred = svm_preds[svm_idx]

                if crf_pred == svm_pred:
                    final_tags[i] = crf_pred
                else:
                    if meta_model:
                        meta_feature = [{'token': tok.lower(), 'crf_pred': crf_pred, 'svm_pred': svm_pred}]
                        final_tags[i] = meta_model.predict(meta_vectorizer.transform(meta_feature))[0]
                    else:
                        final_tags[i] = crf_pred
                svm_idx += 1
            else:
                svm_idx += 1  # still advance pointer to stay aligned

        all_final_tags.extend(final_tags)

    return all_final_tags



# --- Evaluate Hierarchical Ensemble ---
y_test_flat = [label for sent in y_test_crf for label in sent]
ensemble_preds = predict_final_ensemble(test_sentences, crf, svm, vectorizer, meta_model, meta_vectorizer)
ensemble_preds = ensemble_preds[:len(y_test_flat)]  # Ensure length matches
ensemble_accuracy = accuracy_score(y_test_flat, ensemble_preds)
ensemble_f1 = f1_score(y_test_flat, ensemble_preds, average='weighted')

# --- Calculate and Display Improvement ---
crf_f1_improvement = ((KNOWN_CRF_SCORES["f1_score"] - KNOWN_BASELINE_SCORES["f1_score"]) / KNOWN_BASELINE_SCORES[
    "f1_score"]) * 100
ensemble_f1_improvement = ((ensemble_f1 - KNOWN_BASELINE_SCORES["f1_score"]) / KNOWN_BASELINE_SCORES["f1_score"]) * 100

print("\n" + "#" * 60)
print("### FINAL MODEL PERFORMANCE REPORT ###")
print("#" * 60 + "\n")

summary_data = {
    "Model": [KNOWN_BASELINE_SCORES["model_name"], KNOWN_CRF_SCORES["model_name"], "Proposed (Hierarchical Ensemble)"],
    "Accuracy": [KNOWN_BASELINE_SCORES["accuracy"], KNOWN_CRF_SCORES["accuracy"], ensemble_accuracy],
    "Weighted F1-Score": [KNOWN_BASELINE_SCORES["f1_score"], KNOWN_CRF_SCORES["f1_score"], ensemble_f1],
    "% F1 Improvement vs Baseline": ["-", f"{crf_f1_improvement:.2f}%", f"{ensemble_f1_improvement:.2f}%"]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n--- Detailed Classification Report for Proposed Ensemble Model ---")
print(classification_report(y_test_flat, ensemble_preds, digits=4, zero_division='warn'))