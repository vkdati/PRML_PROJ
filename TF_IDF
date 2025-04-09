from sklearn.feature_extraction.text import TfidfVectorizer

# === Step 1: Initialize TF-IDF Vectorizer ===
tfidf_vectorizer = TfidfVectorizer(
    max_features=1200,      # Top 1200 most important words/phrases
    ngram_range=(1, 2),     # Unigrams + Bigrams
    min_df=5,               # Appear in at least 5 resumes
    max_df=0.8              # Appear in at most 80% of resumes
)

# === Step 2: Fit and Transform ===
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Resume'])

# === Step 3: Confirm Shape ===
print("✅ TF-IDF Matrix Shape:", tfidf_matrix.shape)
