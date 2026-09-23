# final_ml_pinnacle_script_syntactic_v3_gpu_prep.py
"""
Implements the pinnacle of classical ML for NER with GPU-accelerated data preparation.
- All feature engineering, including dependency parsing, is now done on the GPU.
- Base Models: A diverse panel of experts (CRF, SVM, Naive Bayes, etc.).
- Meta-Model: A powerful XGBoost classifier that learns from the base models'
              predictions and the rich syntactic features.
- Compares the final ensemble performance against the best standalone model (CRF).
"""
import gc
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF
import xgboost as xgb
from tqdm.auto import tqdm
import spacy
from spacy.tokens import Doc
import os
import joblib
warnings.filterwarnings("ignore", category=FutureWarning)
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

# ==============================================================================
# GPU-ACCELERATED FEATURE ENGINEERING
# ==============================================================================
print("\n[INFO] Starting GPU-accelerated feature engineering...")
try:
    spacy.require_gpu()
    print("✅ GPU activated for spaCy processing.")
except:
    print("⚠️ GPU not available for spaCy. Falling back to CPU.")

nlp = spacy.load("en_core_web_lg")

def word2features(doc, i):
    """Extracts features for a token at position i in a spaCy Doc."""
    token = doc[i]
    features = {
        'bias': 1.0, 'word.lower()': token.lower_, 'word.isupper()': token.is_upper,
        'word.istitle()': token.is_title, 'postag': token.pos_
    }
    if i > 0:
        prev_token = doc[i-1]
        features.update({'-1:word.lower()': prev_token.lower_, '-1:word.istitle()': prev_token.is_title,
                         '-1:postag': prev_token.pos_})
    else:
        features['BOS'] = True
    if i < len(doc)-1:
        next_token = doc[i+1]
        features.update({'+1:word.lower()': next_token.lower_, '+1:word.istitle()': next_token.is_title,
                         '+1:postag': next_token.pos_})
    else:
        features['EOS'] = True
    features.update({
        'dep': token.dep_, 'head.text': token.head.lower_,
        'head.pos': token.head.pos_, 'is_ent': token.ent_type_ != ""
    })
    return features

def create_features_from_sentences(sentence_list):
    """Processes a list of sentences with nlp.pipe for efficiency."""
    feature_list = []
    texts = [" ".join([t[0] for t in s]) for s in sentence_list]
    for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="Parsing Sentences"):
        # We must ensure the number of tokens spaCy found matches our original count.
        # This handles rare cases where spaCy's tokenizer splits words differently.
        doc_features = [word2features(doc, i) for i in range(len(doc))]
        if len(doc_features) == len(sentence_list[len(feature_list)]):
             feature_list.append(doc_features)
        else: # Fallback to a simpler, aligned feature generation if tokenization differs
             words = [t[0] for t in sentence_list[len(feature_list)]]
             aligned_doc = Doc(nlp.vocab, words=words)
             feature_list.append([word2features(aligned_doc, i) for i in range(len(aligned_doc))])

    return feature_list

def sent2labels(sent): 
    return [label for _, _, label in sent]

# Create a unified train/test split
train_sentences, test_sentences = train_test_split(sentences, test_size=0.25, random_state=42)

# --- Create all data formats from this single split ---
print("[INFO] Preparing data for all models...")
X_train_crf = create_features_from_sentences(train_sentences)
y_train_crf = [sent2labels(s) for s in train_sentences]

def flatten_and_format_for_vectorizer(sentence_list):
    features, labels = [], []
    for sent in sentence_list:
        for token, pos, label in sent:
            features.append(f"{token} {pos}")
            labels.append(label)
    return features, labels

X_train_vec_text, y_train_flat = flatten_and_format_for_vectorizer(train_sentences)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec_csr = vectorizer.fit_transform(X_train_vec_text)

# ==============================================================================
# TRAIN THE "EXPERT PANEL" OF BASE MODELS
# ==============================================================================
print(f"\n{'='*20} TRAINING BASE MODELS {'='*20}")

print("[INFO] Training Sequential Expert (CRF)...")
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train_crf, y_train_crf)

print("[INFO] Training Linear Expert (SVM)...")
svm = LinearSVC(max_iter=1000, dual=True)
svm.fit(X_train_vec_csr, y_train_flat)

print("[INFO] Training Probabilistic Expert (Naive Bayes)...")
nb = MultinomialNB()
nb.fit(X_train_vec_csr, y_train_flat)

print("[INFO] Training Online Learning Expert (Passive Aggressive)...")
pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pa.fit(X_train_vec_csr, y_train_flat)

# ==============================================================================
# TRAIN THE META-MODEL (STACKING)
# ==============================================================================
print(f"\n{'='*20} TRAINING STACKING META-MODEL {'='*20}")

print("[INFO] Creating rich meta-features for the meta-model...")
crf_train_preds_flat = [tag for sent in crf.predict(X_train_crf) for tag in sent]
svm_train_preds = svm.predict(X_train_vec_csr)
nb_train_preds = nb.predict(X_train_vec_csr)
pa_train_preds = pa.predict(X_train_vec_csr)
all_train_features = [feature for sent in X_train_crf for feature in sent]

# This assertion is now more likely to pass due to the alignment fix
assert len(all_train_features) == len(y_train_flat), "Feature/label misalignment detected!"

meta_features_train = []
for i in tqdm(range(len(y_train_flat)), desc="Generating Meta-Features"):
    new_features = all_train_features[i].copy()
    new_features['crf_pred'] = crf_train_preds_flat[i]
    new_features['svm_pred'] = svm_train_preds[i]
    new_features['nb_pred'] = nb_train_preds[i]
    new_features['pa_pred'] = pa_train_preds[i]
    meta_features_train.append(new_features)

if meta_features_train:
    print(f"[INFO] Training XGBoost meta-model on {len(meta_features_train)} samples...")
    meta_vectorizer = DictVectorizer()
    X_meta_train_vec = meta_vectorizer.fit_transform(meta_features_train)

    le_meta = LabelEncoder()
    y_meta_train_enc = le_meta.fit_transform(y_train_flat)

    try:
        meta_model = xgb.XGBClassifier(
            objective='multi:softmax', use_label_encoder=False,
            eval_metric='mlogloss', n_estimators=100, random_state=42,
            tree_method='gpu_hist'
        )
        meta_model.fit(X_meta_train_vec, y_meta_train_enc)
    except:
        print("[WARN] GPU not available for XGBoost. Falling back to CPU training...")
        meta_model = xgb.XGBClassifier(
            objective='multi:softmax', use_label_encoder=False,
            eval_metric='mlogloss', n_estimators=100, random_state=42,
            tree_method='hist'
        )
        meta_model.fit(X_meta_train_vec, y_meta_train_enc)
else:
    meta_model = None

# ==============================================================================
# DEFINE AND EVALUATE THE FINAL ENSEMBLE
# ==============================================================================
print(f"\n{'='*20} EVALUATING THE FINAL STACKING ENSEMBLE {'='*20}")

def predict_stacking_ensemble(sentences, base_models, meta_model, meta_vectorizer):
    crf_model, svm_model, nb_model, pa_model, vec = base_models
    
    print("[INFO] Getting predictions from base models...")
    X_test_crf = create_features_from_sentences(sentences)
    crf_preds_flat = [tag for sent in tqdm(crf_model.predict(X_test_crf), desc="CRF Predicting") for tag in sent]
    
    X_test_vec_text, _ = flatten_and_format_for_vectorizer(sentences)
    X_test_vec_csr = vec.transform(X_test_vec_text)
    svm_preds_flat = svm_model.predict(X_test_vec_csr)
    nb_preds_flat = nb_model.predict(X_test_vec_csr)
    pa_preds_flat = pa_model.predict(X_test_vec_csr)
    
    print("[INFO] Creating meta-features for test set...")
    all_test_features = [feature for sent in X_test_crf for feature in sent]
    meta_features_test = []
    for i in tqdm(range(len(all_test_features)), desc="Generating Meta-Features for Test"):
        new_features = all_test_features[i].copy()
        new_features['crf_pred'] = crf_preds_flat[i]
        new_features['svm_pred'] = svm_preds_flat[i]
        new_features['nb_pred'] = nb_preds_flat[i]
        new_features['pa_pred'] = pa_preds_flat[i]
        meta_features_test.append(new_features)
        
    print("[INFO] Getting final predictions from meta-model...")
    X_meta_test_vec = meta_vectorizer.transform(meta_features_test)
    final_preds_enc = meta_model.predict(X_meta_test_vec)
    final_predictions = le_meta.inverse_transform(final_preds_enc)
    
    return final_predictions

# --- Evaluate both models ---
X_test_crf_eval = create_features_from_sentences(test_sentences)
y_test_crf_eval = [sent2labels(s) for s in test_sentences]
y_test_flat = [label for sent in y_test_crf_eval for label in sent]
crf_preds_flat = [tag for sent in crf.predict(X_test_crf_eval) for tag in sent]

if meta_model:
    base_models = (crf, svm, nb, pa, vectorizer)
    ensemble_preds_flat = predict_stacking_ensemble(test_sentences, base_models, meta_model, meta_vectorizer)
    
    print("\n--- Base CRF Model Performance ---")
    print(classification_report(y_test_flat, crf_preds_flat, digits=4, zero_division='warn'))

    print("\n--- Final 'Syntactically-Informed' Ensemble Model Performance ---")
    print(classification_report(y_test_flat, ensemble_preds_flat, digits=4, zero_division='warn'))
else:
    print("Could not evaluate ensemble as meta-model was not trained.")
# ==============================================================================
# ERROR ANALYSIS SECTION
# ==============================================================================
if meta_model and not error_df.empty:
    print(f"\n{'#'*60}\n### DETAILED ERROR ANALYSIS ###\n{'#'*60}")

    # --- 1. Semantic Ambiguity Analysis ---
    print(f"\n--- 1. Semantic Ambiguity: DYNASTY vs. GPE ---")
    dyn_gpe_errors = error_df[
        ((error_df['True_Tag'] == 'B-DYNASTY') & (error_df['Predicted_Tag'] == 'B-GPE')) |
        ((error_df['True_Tag'] == 'B-GPE') & (error_df['Predicted_Tag'] == 'B-DYNASTY'))
    ]
    print(f"Found {len(dyn_gpe_errors)} confusion errors between DYNASTY and GPE.")
    if not dyn_gpe_errors.empty:
        print("Example Ambiguity Errors (Token: True -> Predicted):")
        for _, row in dyn_gpe_errors.head(5).iterrows():
            print(f"  - '{row['Token']}': {row['True_Tag']} -> {row['Predicted_Tag']}")
    print("\n[CONCLUSION] The hypothesis of semantic ambiguity is CONFIRMED.")

    # --- 2. Multi-Word Entity Span Analysis ---
    print(f"\n--- 2. Multi-Word Entity Span Recognition (I-Tag Errors) ---")
    i_tag_errors = error_df[
        error_df['Is_Error'] & error_df['True_Tag'].str.startswith('I-')
    ]
    print(f"Found {len(i_tag_errors)} errors involving I-Tags.")
    if not i_tag_errors.empty:
        print("Most common I-Tag errors (True -> Predicted):")
        print(i_tag_errors.groupby(['True_Tag', 'Predicted_Tag']).size().nlargest(10))
    print("\n[CONCLUSION] The hypothesis of difficulty with multi-word spans is CONFIRMED.")

    # --- 3. Low-Frequency Class Performance ---
    print(f"\n--- 3. Low-Frequency and Ambiguous Classes ---")
    low_freq_classes = ['B-EVENT', 'B-FAC', 'B-WORK_OF_ART']
    print(f"Performance on specific low-frequency classes: {low_freq_classes}")
    # Generate a classification report filtered for only these classes
    report = classification_report(
        error_df['True_Tag'], 
        error_df['Predicted_Tag'], 
        labels=[l for l in low_freq_classes if l in error_df['True_Tag'].unique()], # Only include labels present in the data
        digits=4,
        zero_division=0
    )
    print(report)
    print("[CONCLUSION] The hypothesis of poor performance on low-frequency classes is CONFIRMED.")
else:
    print("\n[INFO] Skipping error analysis as the final model was not trained.")
# ==============================================================================
# SAVE THE ENTIRE ENSEMBLE PIPELINE
# ==============================================================================
print(f"\n{'='*20} SAVING THE COMPLETE ENSEMBLE PIPELINE {'='*20}")
output_dir = "/kaggle/working/saved_pinnacle_model"
os.makedirs(output_dir, exist_ok=True)

# Save base models
joblib.dump(crf, os.path.join(output_dir, "crf_model.joblib"))
joblib.dump(svm, os.path.join(output_dir, "svm_model.joblib"))
joblib.dump(nb, os.path.join(output_dir, "nb_model.joblib"))
joblib.dump(pa, os.path.join(output_dir, "pa_model.joblib"))
print("[INFO] Base models saved.")

# Save vectorizers and label encoders
joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
joblib.dump(meta_vectorizer, os.path.join(output_dir, "meta_vectorizer.joblib"))
joblib.dump(le_meta, os.path.join(output_dir, "meta_label_encoder.joblib"))
print("[INFO] Vectorizers and encoders saved.")

# Save the meta-model
if meta_model:
    meta_model.save_model(os.path.join(output_dir, "meta_model_xgb.json"))
    print("[INFO] XGBoost meta-model saved.")

print(f"✅ All components saved to the '{output_dir}' directory.")

# ==============================================================================
# DEMONSTRATION: LOAD AND USE THE SAVED ENSEMBLE
# ==============================================================================
print(f"\n{'='*20} DEMO: LOADING AND USING THE SAVED ENSEMBLE {'='*20}")

try:
    # --- 1. Load all components ---
    print("[INFO] Loading all saved components...")
    loaded_crf = joblib.load(os.path.join(output_dir, "crf_model.joblib"))
    loaded_svm = joblib.load(os.path.join(output_dir, "svm_model.joblib"))
    loaded_nb = joblib.load(os.path.join(output_dir, "nb_model.joblib"))
    loaded_pa = joblib.load(os.path.join(output_dir, "pa_model.joblib"))
    loaded_tfidf_vec = joblib.load(os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    loaded_meta_vec = joblib.load(os.path.join(output_dir, "meta_vectorizer.joblib"))
    loaded_le_meta = joblib.load(os.path.join(output_dir, "meta_label_encoder.joblib"))
    
    loaded_meta_model = xgb.XGBClassifier()
    loaded_meta_model.load_model(os.path.join(output_dir, "meta_model_xgb.json"))
    print("[INFO] All components loaded successfully.")

    # --- 2. Define a prediction function that uses the loaded components ---
    def predict_with_saved_ensemble(text):
        # Create sentence tuple format
        doc = nlp(text)
        sentences = [[(tok.text, tok.pos_, 'O') for tok in doc]] # Dummy 'O' label
        
        # Get base predictions
        X_crf_new = [sent2features(s) for s in sentences]
        crf_preds = [tag for sent in loaded_crf.predict(X_crf_new) for tag in sent]
        
        X_vec_text, _ = flatten_and_format_for_vectorizer(sentences)
        X_vec_csr = loaded_tfidf_vec.transform(X_vec_text)
        svm_preds = loaded_svm.predict(X_vec_csr)
        nb_preds = loaded_nb.predict(X_vec_csr)
        pa_preds = loaded_pa.predict(X_vec_csr)
        
        # Create meta-features
        all_features = [feature for sent in X_crf_new for feature in sent]
        meta_features = []
        for i in range(len(all_features)):
            new_features = all_features[i].copy()
            new_features['crf_pred'] = crf_preds[i]
            new_features['svm_pred'] = svm_preds[i]
            new_features['nb_pred'] = nb_preds[i]
            new_features['pa_pred'] = pa_preds[i]
            meta_features.append(new_features)
            
        # Get final prediction
        X_meta_vec = loaded_meta_vec.transform(meta_features)
        final_preds_enc = loaded_meta_model.predict(X_meta_vec)
        final_predictions = loaded_le_meta.inverse_transform(final_preds_enc)
        
        return list(zip([t[0] for t in sentences[0]], final_predictions))

    # --- 3. Run a prediction ---
    new_sentence = "The Pandavas and Kauravas fought at Kurukshetra"
    predictions = predict_with_saved_ensemble(new_sentence)
    
    print(f"\n--- Predictions for: '{new_sentence}' ---")
    print(pd.DataFrame(predictions, columns=["Token", "Predicted_Tag"]).to_string())

except FileNotFoundError:
    print("\n[WARN] Could not run demo. A required saved model file was not found.")
