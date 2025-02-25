from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def normalize_vectors(vectors):
    """
    Normalize a list of vectors using Min-Max scaling to the range [0, 1].

    Args:
    - vectors: A list of numeric vectors to normalize.

    Returns:
    - Normalized vectors.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(vectors)


def extract_numeric_fields(dict_list):
    """
    Extract numeric fields from a list of dictionaries.

    Args:
    - dict_list: A list of dictionaries with numeric values.

    Returns:
    - A list of numeric vectors corresponding to each dictionary.
    - A list of keys (fields) that correspond to the numeric values.
    """
    numeric_keys = set()

    # Identify all numeric fields
    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    # Convert each dictionary into a vector based on numeric fields
    numeric_vectors = []
    numeric_keys = list(numeric_keys)  # To ensure consistent order of keys

    for d in dict_list:
        vector = []
        for key in numeric_keys:
            # Append the value if it exists; otherwise append 0
            value = d.get(key, 0)
            if isinstance(value, (int, float)):
                vector.append(value)
            else:
                vector.append(0)  # Default to 0 for missing values
        numeric_vectors.append(vector)

    return numeric_vectors, numeric_keys


def compute_cosine_similarity(new_dict, dict_list):
    """
    Compute cosine similarity between a new dictionary and a list of dictionaries.

    Args:
    - new_dict: A dictionary containing numeric values.
    - dict_list: A list of dictionaries to compare against.

    Returns:
    - A list of cosine similarities between the new dictionary and each dictionary in the list.
    """
    # Combine new_dict with the dict_list
    all_dicts = [new_dict] + dict_list

    # Extract numeric vectors
    numeric_vectors, numeric_keys = extract_numeric_fields(all_dicts)

    # Normalize the numeric vectors
    normalized_vectors = normalize_vectors(numeric_vectors)

    # Convert to numpy arrays
    normalized_vectors = np.array(normalized_vectors)

    # Compute cosine similarity among all items
    similarity_matrix = cosine_similarity(normalized_vectors)

    # The first row will contain the similarities between the new_dict and each dictionary
    return similarity_matrix[0][1:], numeric_keys


# Example Usage
new_resource_metadata = {
    'price': 100.0,
    'size': 50.0,
    'weight': 10.5,
}

all_packages_metadata = [
    {'price': 99.9, 'size': 50.1, 'weight': 15.6, 'other_field': 'value1'},
    {'price': 105.0, 'size': 49.0, 'weight': 9.8, 'other_field': 'value2'},
    {'price': 200.0, 'size': 60.0, 'weight': 12.0, 'other_field': 'value3'},
]

similarities, numeric_fields = compute_cosine_similarity(new_resource_metadata, all_packages_metadata)

# Print results
for i, sim in enumerate(similarities):
    print(f"Similarity with item {i + 1}: {sim:.4f}")
