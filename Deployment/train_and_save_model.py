import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import joblib

# === Load your dataset ===
df = pd.read_csv("raw_dataset.csv")

# === Encode labels ===
label_encoder = LabelEncoder()
y_bal_encoded = label_encoder.fit_transform(df['Category'])

# === TF-IDF Vectorizer ===
tfidf_vectorizer = TfidfVectorizer(
    max_features=1200,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Resume'])


# === Structural Feature Extractor ===
def extract_structural_features(text_series):
    features = []
    for text in text_series:
        tokens = text.split()
        num_words = len(tokens)
        num_chars = sum(len(word) for word in tokens)
        avg_word_length = num_chars / num_words if num_words else 0
        num_digits = sum(char.isdigit() for char in text)
        num_uppercase_words = sum(word.isupper() for word in tokens)
        num_unique_words = len(set(tokens))
        num_stopwords = sum(1 for word in tokens if word.lower() in ENGLISH_STOP_WORDS)

        features.append([
            num_words,
            avg_word_length,
            num_digits,
            num_uppercase_words,
            num_unique_words,
            num_stopwords
        ])
    return np.array(features)


structural_features = extract_structural_features(df['Resume'])
scaler_struct = StandardScaler()
struct_scaled = scaler_struct.fit_transform(structural_features)
struct_sparse = csr_matrix(struct_scaled)

# === Semantic (BERT) Features ===
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_vectors = bert_model.encode(df['Resume'].tolist(), show_progress_bar=True)
scaler_sem = StandardScaler()
semantic_scaled = scaler_sem.fit_transform(semantic_vectors)
semantic_sparse = csr_matrix(semantic_scaled)

# === Stack All Features ===
X_np = hstack([tfidf_matrix, struct_sparse, semantic_sparse])

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_np, y_bal_encoded)
print("After SMOTE class distribution:", Counter(y_smote))

# === Train-Test Split ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X_smote, y_smote, test_size=0.3, stratify=y_smote, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# === Train XGBoost Classifier ===
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
val_score = xgb_clf.score(X_val, y_val)
test_score = xgb_clf.score(X_test, y_test)
print(f"Validation Accuracy: {val_score*100:.2f}%")
print(f"Test Accuracy: {test_score*100:.2f}%")

# === Save All Required Objects ===
joblib.dump(xgb_clf, "models/xgb_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(tfidf_vectorizer, "models/vectorizer.pkl")
joblib.dump(scaler_struct, "models/scaler_struct.pkl")
joblib.dump(scaler_sem, "models/scaler_sem.pkl")

print("All components saved as .pkl files!")
