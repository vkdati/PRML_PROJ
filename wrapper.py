from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# === 1. Scale Structural Features ===
scaler_struct = StandardScaler()
struct_scaled_balanced = scaler_struct.fit_transform(structural_df)

# === 2. Scale Semantic Features ===
scaler_sem = StandardScaler()
semantic_scaled_balanced = scaler_sem.fit_transform(semantic_vectors_balanced)

# === 3. Convert to Sparse Matrices ===
struct_sparse_balanced = csr_matrix(struct_scaled_balanced)
semantic_sparse_balanced = csr_matrix(semantic_scaled_balanced)

# === 4. Stack TF-IDF + Structural + Semantic ===
X_np = hstack([
    tfidf_matrix,
    struct_sparse_balanced,
    semantic_sparse_balanced
])

# Final shape check
print("Final Feature Matrix Shape:", X_np.shape)

