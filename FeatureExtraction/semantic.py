import numpy as np

def avg_glove_vector(text_series, model, dim=100):
    vectors = []

    for text in text_series:
        words = text.split()
        valid_words = [word for word in words if word in model]

        if valid_words:
            vec = np.mean([model[word] for word in valid_words], axis=0)
        else:
            vec = np.zeros(dim)

        vectors.append(vec)

    return np.vstack(vectors)

# Apply it to your resumes
semantic_vectors = avg_glove_vector(df['Resume'], glove_model)

# Check shape
print("✅ Semantic Feature Matrix Shape:", semantic_vectors.shape)
