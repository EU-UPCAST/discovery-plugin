import numpy as np
from sklearn.neighbors import NearestNeighbors
import faiss

# Step 1: Generate random data for the index and query datasets
# Let's assume each vector has 128 dimensions, with 10,000 vectors in the index and 100 queries.
d = 128  # dimension of vectors
num_index = 100000  # number of vectors in the index
num_queries = 10000  # number of query vectors
k = 10  # number of nearest neighbors to search

# Create random data
np.random.seed(42)
index_data = np.random.random((num_index, d)).astype(np.float32)
query_data = np.random.random((num_queries, d)).astype(np.float32)

# Function to calculate FAISS accuracy
def calculate_faiss_accuracy(query_data, index_data, k=10):
    # Step 2: Exact nearest neighbors using scikit-learn
    exact_nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    exact_nn.fit(index_data)
    _, exact_indices = exact_nn.kneighbors(query_data)

    # Step 3: FAISS nearest neighbors search
    dimension = index_data.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance metric
    faiss_index.add(index_data)  # Add the index data to the FAISS index
    _, faiss_indices = faiss_index.search(query_data, k)

    # Step 4: Calculate recall-based accuracy
    correct = 0
    total = 0
    for i in range(query_data.shape[0]):
        # Compare FAISS results to exact results for each query
        correct += len(set(exact_indices[i]).intersection(faiss_indices[i]))
        total += k

    # Accuracy calculation
    accuracy = correct / total
    return accuracy

# Calculate accuracy and display the result
accuracy = calculate_faiss_accuracy(query_data, index_data, k)
print(f"FAISS accuracy: {accuracy:.4f}")
