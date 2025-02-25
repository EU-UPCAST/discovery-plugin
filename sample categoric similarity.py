from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def exact_match_similarity(val1, val2):
    """
    Compute exact match similarity for single-label categorical data.
    Returns 1.0 for a match, 0.0 otherwise.
    """
    return 1.0 if val1 == val2 else 0.0

def jaccard_similarity(list1, list2):
    """
    Compute Jaccard similarity for multi-label (list) categorical data.
    """
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


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


def compute_combined_similarity(new_dict, dict_list, numeric_weight=0.7, categorical_weight=0.3):
    """
    Compute combined similarity between a new dictionary and a list of dictionaries
    based on both numeric and categorical fields.

    Args:
    - new_dict: The new dictionary containing numeric and categorical values.
    - dict_list: A list of dictionaries to compare against.
    - numeric_weight: Weight to assign to numeric similarity (default 0.7).
    - categorical_weight: Weight to assign to categorical similarity (default 0.3).

    Returns:
    - A list of combined similarity scores between the new_dict and each dictionary in dict_list.
    """
    all_dicts = [new_dict] + dict_list

    # Extract numeric fields and normalize
    numeric_vectors, numeric_keys = extract_numeric_fields(all_dicts)
    normalized_vectors = normalize_vectors(numeric_vectors)

    # Compute cosine similarity for numeric fields
    similarity_matrix = cosine_similarity(normalized_vectors)
    numeric_similarities = similarity_matrix[0][1:]  # Numeric similarities between new_dict and all others

    combined_similarities = []

    # Iterate through the dictionaries and compute similarity for categorical fields
    for i, existing_dict in enumerate(dict_list):
        categorical_sim = 0
        categorical_count = 0

        # Compute categorical similarity
        for key, value in new_dict.items():
            if key in existing_dict and isinstance(value, (str, list)):
                existing_value = existing_dict[key]
                if isinstance(value, str) and isinstance(existing_value, str):
                    # Exact match similarity for single-label categorical fields
                    categorical_sim += exact_match_similarity(value, existing_value)
                elif isinstance(value, list) and isinstance(existing_value, list):
                    # Jaccard similarity for multi-label categorical fields
                    categorical_sim += jaccard_similarity(value, existing_value)
                categorical_count += 1

        # Normalize categorical similarity (in case there are multiple fields)
        if categorical_count > 0:
            categorical_sim /= categorical_count

        # Combine numeric and categorical similarity using weights
        combined_similarity = (numeric_weight * numeric_similarities[i]) + (categorical_weight * categorical_sim)
        combined_similarities.append(combined_similarity)

    return combined_similarities


# Example Usage
new_resource_metadata = {
    'price': 100.0,
    'size': 50.0,
    'weight': 10.5,
    'category': 'electronics',  # Single-label categorical
    'tags': ['tech', 'gadget'],  # Multi-label categorical
}

all_packages_metadata = [
    {'price': 99.9, 'size': 50.1, 'weight': 10.6, 'category': 'electronics', 'tags': ['gadget', 'technology'],
     'other_field': 'value1'},
    {'price': 105.0, 'size': 49.0, 'weight': 9.8, 'category': 'home_appliances', 'tags': ['home', 'appliance'],
     'other_field': 'value2'},
    {'price': 200.0, 'size': 60.0, 'weight': 12.0, 'category': 'electronics', 'tags': ['gadget'],
     'other_field': 'value3'},
]

# Calculate combined similarity
combined_similarities = compute_combined_similarity(new_resource_metadata, all_packages_metadata)

# Print the combined similarities
for i, sim in enumerate(combined_similarities):
    print(f"Combined Similarity with item {i + 1}: {sim:.4f}")
