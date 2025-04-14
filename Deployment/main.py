from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.sparse import hstack, csr_matrix
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import uvicorn

# === Load Pretrained Assets ===
xgb_model = joblib.load("./models/xgb_model.pkl")
label_encoder = joblib.load("./models/label_encoder.pkl")
tfidf_vectorizer = joblib.load("./models/vectorizer.pkl")
scaler_struct = joblib.load("./models/scaler_struct.pkl")
scaler_sem = joblib.load("./models/scaler_sem.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

# === Allow Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Serve Static HTML ===
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# === Structural Feature Extractor ===
def extract_structural_features(text):
    tokens = text.split()
    num_words = len(tokens)
    num_chars = sum(len(word) for word in tokens)
    avg_word_length = num_chars / num_words if num_words else 0
    num_digits = sum(char.isdigit() for char in text)
    num_uppercase_words = sum(word.isupper() for word in tokens)
    num_unique_words = len(set(tokens))
    num_stopwords = sum(1 for word in tokens if word.lower() in ENGLISH_STOP_WORDS)
    return np.array([[num_words, avg_word_length, num_digits, num_uppercase_words, num_unique_words, num_stopwords]])


# === Prediction Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pdf_reader = PdfReader(file.file)
    resume_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    tfidf_features = tfidf_vectorizer.transform([resume_text])
    struct_features = extract_structural_features(resume_text)
    struct_scaled = scaler_struct.transform(struct_features)
    struct_sparse = csr_matrix(struct_scaled)
    semantic_embedding = bert_model.encode([resume_text])
    semantic_scaled = scaler_sem.transform(semantic_embedding)
    semantic_sparse = csr_matrix(semantic_scaled)

    combined = hstack([tfidf_features, struct_sparse, semantic_sparse])
    prediction = xgb_model.predict(combined)
    label = label_encoder.inverse_transform(prediction)[0]

    return {"category": label}

# === Local run ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
