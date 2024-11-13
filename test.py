import nltk
import random
import requests
import time

# Download the Brown Corpus and necessary tokenizers
nltk.download('brown')
nltk.download('punkt')

from nltk.corpus import brown
from nltk import sent_tokenize

# Prepare the Brown Corpus sentences
brown_sentences = []
for category in brown.categories():
    words = brown.words(categories=category)
    text = " ".join(words)
    sentences = sent_tokenize(text)
    brown_sentences.extend(sentences)


# Function to generate random meaningful sentences
def generate_random_sentences(num_sentences=5):
    random_sentences = random.sample(brown_sentences, num_sentences)
    return "\n".join(random_sentences)


# Function to create a dataset with random package notes
def create_dataset(api_url, package_name, package_title, organization_name):
    package_notes = generate_random_sentences(5)

    form_data = {
        'package_name': package_name,
        'package_title': package_title,
        'organization_name': organization_name,
        'package_notes': package_notes
    }

    response = requests.post(api_url, data=form_data)
    return response.status_code, response.text


# Function to call the embedding API
def create_embeddings(api_url):
    start_time = time.time()
    response = requests.get(api_url, headers={'accept': 'application/json'})
    elapsed_time = time.time() - start_time
    return response.status_code, elapsed_time


# Function to call the data processing workflow API with recent package names
def call_data_processing_workflow(api_url, package_names):
    start_time = time.time()
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = package_names
    response = requests.post(api_url, headers=headers, json=data)
    elapsed_time = time.time() - start_time
    return response.status_code, elapsed_time


# Function to loop and create 1000 datasets per iteration, then call the embeddings and workflow APIs
def create_multiple_datasets(api_url, embedding_url, workflow_url, iterations=100, datasets_per_iteration=1000):
    all_results = []  # List to store time results for each iteration

    for iteration in range(1, iterations + 1):
        print(f"Starting iteration {iteration}/{iterations}...")

        # Time the dataset creation process
        start_time = time.time()
        recent_package_names = []  # To store the last 15 package names

        for i in range(datasets_per_iteration):
            package_name = f'sample_{iteration}_{i}'
            package_title = f'Sample {iteration} {i}'
            organization_name = 'upcast'

            status_code, response_text = create_dataset(api_url, package_name, package_title, organization_name)

            # Add the package name to the recent list (keeping only the last 15)
            if len(recent_package_names) >= 15:
                recent_package_names.pop(0)  # Remove the oldest entry if we have 15
            recent_package_names.append(package_name)

            if i % 100 == 0:
                print(f"Iteration {iteration}: Created dataset {i + 1}/{datasets_per_iteration}, Status: {status_code}")

        creation_time = time.time() - start_time
        print(f"Finished creating {datasets_per_iteration} datasets in {creation_time:.2f} seconds.")

        # Call the embedding creation API and time it
        embedding_status, embedding_time = create_embeddings(embedding_url)
        print(f"Embedding creation took {embedding_time:.2f} seconds, Status: {embedding_status}")

        # Call the data processing workflow API with the recent package names
        workflow_status, workflow_time = call_data_processing_workflow(workflow_url, recent_package_names)
        print(f"Data processing workflow took {workflow_time:.2f} seconds, Status: {workflow_status}")

        # Store the results in a dictionary
        results = {
            'iteration': iteration,
            'creation_time': creation_time,
            'embedding_time': embedding_time,
            'workflow_time': workflow_time
        }

        # Append the results to the all_results list
        all_results.append(results)

        time.sleep(1)  # Optional sleep to avoid overloading the server
        print(f"Finished iteration {iteration}/{iterations}.\n")

    return all_results  # Return all results after completion


# Example usage
api_url = 'http://62.171.168.208:10001/catalog/create_dataset/'
embedding_url = 'http://62.171.168.208:10000/discover/create_embeddings'
workflow_url = 'http://62.171.168.208:10000/discover/discover_data_processing_workflow'

results_list = create_multiple_datasets(api_url, embedding_url, workflow_url, iterations=10,
                                        datasets_per_iteration=10)

# You can print or save the results_list as needed
print(results_list)
