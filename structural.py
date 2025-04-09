import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# Extract features
structural_features = extract_structural_features(df['Resume'])

# Wrap in a DataFrame for clarity
import pandas as pd
structural_df = pd.DataFrame(
    structural_features,
    columns=[
        'resume_length',
        'avg_word_length',
        'num_digits',
        'num_uppercase_words',
        'num_unique_words',
        'num_stopwords'
    ]
)

# Check output
print("✅ Structural Feature Matrix Shape:", structural_df.shape)
print("📊 Sample Structural Features:\n", structural_df.head())
