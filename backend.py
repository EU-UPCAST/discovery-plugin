import os
import pickle
from typing import List

import ckanapi
import numpy as np
import torch
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import config

class Backend:

    def __init__(self):
        # Initialize backend client
        self.backend = ckanapi.RemoteCKAN(config.backend_api_url, apikey=config.backend_api_key)

    # Retrieve metadata for all resources
    def get_all_resources_metadata(self):
        # Get a list of packages (datasets)
        packages = self.backend.action.package_list()

        all_resources_metadata = []

        for package_id in packages:
            # Get the resources for a package
            resources = self.backend.action.package_show(id=package_id)['resources']
            all_resources_metadata.extend([resource['description'] for resource in resources if 'description' in resource])

        return all_resources_metadata

    def get_all_packages_metadata(self):
        # Get a list of packages (datasets)
        packages = self.backend.action.package_list()

        all_resources_metadata = []

        for package_id in packages:
            # Get the resources for a package
            # if package_id != excluded_id:
                resources = self.backend.action.package_show(id=package_id)
                all_resources_metadata.append({'id': package_id,"text":resources['notes']})
            # all_resources_metadata.extend(resources['tags'])

        return all_resources_metadata

    # Create BERT embeddings for the metadata
    def create_embeddings(self,texts):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        embeddings = []

        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids)
                embeddings.append(outputs.pooler_output)

        return torch.cat(embeddings)


    def create_embedding_from_list(self,list_of_dicts):
        # Load pre-trained BERT model and tokenizer
        model_name = 'bert-base-uncased'  # Change to the specific BERT model you want to use
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        # # Sample list of dictionaries with text data
        # list_of_dicts = [
        #     {"id": 1, "text": "This is the first document"},
        #     {"id": 2, "text": "This document is the second document"},
        #     {"id": 3, "text": "And this is the third one"},
        #     {"id": 4, "text": "Is this the first document?"},
        # ]

        # Extract text data from the list of dictionaries
        text_data = [d['text'] for d in list_of_dicts]

        # Tokenize and encode the text using BERT tokenizer
        encoded_data = [tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True) for text in
                        text_data]

        # Pad the sequences to the same length
        max_len = max(len(seq) for seq in encoded_data)
        padded_data = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in encoded_data]

        # Convert to PyTorch tensors
        input_ids = torch.tensor(padded_data)

        # Pass the input through the BERT model to get embeddings
        with torch.no_grad():
            outputs = model(input_ids)

        # Extract the embeddings for the [CLS] tokens
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Convert embeddings to a list of NumPy arrays
        embeddings_list = [emb.numpy() for emb in embeddings]

        return embeddings_list

    def update_embedding_metadata(self):

        all_packages_metadata = self.get_all_packages_metadata()
        embeddings = self.create_embedding_from_list(all_packages_metadata)
        with open('./content/embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        with open('./content/all_packages_metadata.pickle', 'wb') as f:
            pickle.dump(all_packages_metadata, f)

    # Function to check similarity for a new resource
    def check_similarity(self,new_resource_metadata, threshold=0.7):

        all_packages_metadata = ""
        if os.path.exists("./content/embeddings.pickle"):
            with open("./content/embeddings.pickle", 'rb') as f:
                embeddings = pickle.load(f)
            with open("./content/all_packages_metadata.pickle", 'rb') as f:
                all_packages_metadata = pickle.load(f)
        else:
            self.update_embedding_metadata()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        encoded_data = tokenizer.encode(new_resource_metadata, add_special_tokens=True, max_length=128, truncation=True,return_tensors='pt')
        # input_ids = tokenizer.encode(new_resource_metadata, return_tensors='pt')
        # with torch.no_grad():
        #     new_embedding = model(input_ids).pooler_output

        # Pad the sequences to the same length
        max_len = max(len(seq) for seq in embeddings)
        # padded_data = [encoded_data + [tokenizer.pad_token_id] * (max_len - len(encoded_data)) ]

        # Convert to PyTorch tensors
        input_ids = torch.tensor(encoded_data)

        # Pass the input through the BERT model to get embeddings
        with torch.no_grad():
            outputs = model(input_ids)

        # Extract the embeddings for the [CLS] tokens
        new_embedding = outputs.last_hidden_state[:, 0, :]

        similarities = cosine_similarity(new_embedding, embeddings)
        max_similarity = max(similarities[0])
        threshold = sorted(similarities[0])[-2] - 0.03
        similar_resources = []

        for i in range(0,similarities[0].size):
            if similarities[0][i]>threshold: # and similarities[0][i]<=0.98:
                similar_resource = all_packages_metadata[i]
                similar_resource["score"] = str(similarities[0][i])
                similar_resources.append(similar_resource)
        if len(similar_resources)>0:
            sorted_data = sorted(similar_resources, key=lambda x: x['score'], reverse=True)
            return sorted_data
        else:
            return "No similar resource found."

    def upload_file_to_dataset(self,dataset_id, file):

        try:
            # Retrieve the dataset by ID
            dataset = self.backend.action.package_show(id=dataset_id)

            # Prepare the resource data
            resource_data = {
                'package_id': dataset_id,
                'name': file.filename,  # Use the file name as the resource name
                'format': file.filename.split('.')[-1],  # Use the file extension as the resource format
            }

            # Create the resource, which returns the resource ID
            resource_id = self.backend.action.resource_create(**resource_data)

            # Upload the file to the resource
            self.backend.action.resource_patch(id=resource_id['id'], upload=(file.filename, file.file))

            return f"File '{file.filename}' uploaded successfully to dataset '{dataset['name']}."

        except Exception as e:
            return e

    def create_backend_package(self,package_name, package_title, organization_name, package_notes):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            # Prepare the package data
            package_data = {
                'name': package_name,
                'title': package_title,
                'notes': package_notes,
                'owner_org': organization['id'],
            }

            # Create the package
            self.backend.action.package_create(**package_data)

            return f"Package '{package_name}' created successfully."
        except Exception as e:
            return e
    def create_backend_package_custom(self,organization_name, package_data):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            package_data["owner_org"] = organization['id']

            # Create the package
            self.backend.action.package_create(**package_data)

            return "Package {} created successfully.".format(package_data["name"])
        except Exception as e:
            return e

    def update_backend_package(self,package_id, package_name, package_title, organization_name, package_notes):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            # Prepare the package data
            package_data = {
                'id' : package_id,
                'name': package_name,
                'title': package_title,
                'notes': package_notes,
                'owner_org': organization['id'],
            }

            # Create the package
            self.backend.action.package_update(**package_data)

            return f"Package '{package_name}' created successfully."
        except Exception as e:
            return e
    def delete_package(self,package_id):
        try:
            self.backend.action.package_delete(id=package_id)
            print(f"Package {package_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting package: {e}")

import os
import pickle
import ckanapi
import torch
import faiss
from transformers import BertTokenizer, BertModel
import config


class FaissSimilarItem(BaseModel):
    id: str = Field(title="id", description="The unique id of the dataset", example="dataset-id")
    text: str = Field(title="text", description="The unique id of the dataset", example="The description text")
    distance: float = Field(title="distance", description="Distance", example="0")

class DatasetSimilarities(BaseModel):
    dataset_id: str = Field(title="Dataset ID", description="The ID of the main dataset", example="transportation-dataset")
    similar_datasets: List[FaissSimilarItem] = Field(title="Similar Datasets", description="List of similar datasets")

class Backend_Faiss:

    def __init__(self):
        # Initialize backend client
        self.backend = ckanapi.RemoteCKAN(config.backend_api_url, apikey=config.backend_api_key)
        # Initialize the tokenizer and model for BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    # Retrieve metadata for all resources
    def get_all_resources_metadata(self):
        packages = self.backend.action.package_list()
        all_resources_metadata = []

        for package_id in packages:
            resources = self.backend.action.package_show(id=package_id)['resources']
            all_resources_metadata.extend([resource['description'] for resource in resources if 'description' in resource])

        return all_resources_metadata

    def get_all_packages_metadata(self):
        packages = self.backend.action.package_list()
        all_packages_metadata = []

        for package_id in packages:
            resources = self.backend.action.package_show(id=package_id)
            all_packages_metadata.append({'id': package_id, "text": resources['notes']})

        return all_packages_metadata

    # Create BERT embeddings for the metadata
    def create_embeddings(self, texts):
        embeddings = []

        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(input_ids)
                embeddings.append(outputs.pooler_output)

        return torch.cat(embeddings).numpy()

    def create_all_embedding_from_list(self, list_of_dicts):
        # Concatenate all field values into a single string for each package
        text_data = []
        for d in list_of_dicts:
            # Join all values into a single string, separated by spaces or any desired delimiter
            concatenated_text = ' '.join(str(value) for value in d.values() if value is not None)
            text_data.append(concatenated_text)

        embeddings = self.create_embeddings(text_data)
        return embeddings


    def create_embedding_from_list(self, list_of_dicts):
        if "text" in list_of_dicts[0].keys():
            text_data = [d['text'] for d in list_of_dicts]
            embeddings = self.create_embeddings(text_data)
        else:
            text_data = [d['notes'] for d in list_of_dicts]
            embeddings = self.create_embeddings(text_data)

        return embeddings

    # Create BERT embeddings for the metadata
    def create_all_embeddings(self, texts):
        embeddings = []

        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(input_ids)
                embeddings.append(outputs.pooler_output)

        return torch.cat(embeddings).numpy()

    def separate_fields(self,input_dict):
        numeric_fields = {}
        text_fields = {}

        for key, value in input_dict.items():
            if key == "id":
                text_fields[key] = value
                #numeric_fields[key] = value
            if isinstance(value, (int, float)):  # Check if value is numeric
                numeric_fields[key] = value
            elif isinstance(value, str):  # Check if value is a string
                text_fields[key] = value

        return numeric_fields, text_fields

    def find_similar_datasets(self, dataset_id_list: List[str], numeric=False):
        backend = Backend_Faiss()
        similar_datasets = []

        for dataset_id in dataset_id_list:
            try:
                # Fetch dataset metadata
                resources = backend.backend.action.package_show(id=dataset_id)
                #numeric_dict, text_dict = self.separate_fields(resources)

                # Check similarity
                similarity_result = backend.check_similarity_all(resources, numeric=True)
                # Prepare result for each dataset
                similar_datasets.append({
                    "dataset_id": dataset_id,
                    "similar_datasets": [
                        FaissSimilarItem(id=item["name"], text=item["notes"], distance=item["distance"])
                        for item in similarity_result
                    ]
                })
            except BaseException as b:
                # If no similar datasets are found or an error occurs, continue to the next
                similar_datasets.append({
                    "dataset_id": dataset_id,
                    "similar_datasets": []
                })

        return similar_datasets


    def get_all_packages_all_metadata(self):
        # Get a list of packages (datasets)
        packages = self.backend.action.package_list()

        all_resources_metadata = []

        for package_id in packages:
            # Get the resources for a package
            resources = self.backend.action.package_show(id=package_id)

            # Prepare metadata dictionary to include all fields
            # Start with the package ID
            package_metadata = {'id': package_id}

            # Add all fields from the resources dictionary
            package_metadata.update(resources)  # This will include all available fields, including custom fields

            all_resources_metadata.append(package_metadata)

        return all_resources_metadata
    def update_embedding_metadata(self):
        all_packages_metadata = self.get_all_packages_all_metadata()

        embeddings = self.create_embedding_from_list(all_packages_metadata)

        with open('./content/embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        with open('./content/all_packages_metadata.pickle', 'wb') as f:
            pickle.dump(all_packages_metadata, f)

    def update_all_embedding_metadata(self):
        all_packages_metadata = self.get_all_packages_all_metadata()
        numeric_dict_list = []
        text_dict_list = []

        for package in all_packages_metadata:
            numeric_dict, text_dict = self.separate_fields(package)
            numeric_dict_list.append(numeric_dict)
            text_dict_list.append(text_dict)

        # Normalize numeric fields and create numeric vectors
        numeric_values = [list(numeric_dict.values()) for numeric_dict in numeric_dict_list]
        numeric_values_array = np.array(numeric_values)

        # Normalize numeric values (min-max normalization in this case)
        scaler = MinMaxScaler()
        normalized_numeric_values = scaler.fit_transform(numeric_values_array)

        embeddings = self.create_embedding_from_list(text_dict_list)
        with open('./content/embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        with open('./content/all_packages_metadata.pickle', 'wb') as f:
            pickle.dump(all_packages_metadata, f)
        with open('./content/normalized_numeric_values.pickle', 'wb') as f:
            pickle.dump(normalized_numeric_values, f)

    # Function to check similarity for a new resource using FAISS
    def check_similarity(self, new_resource_metadata, threshold=0.7):
        all_packages_metadata = ""
        if os.path.exists("./content/embeddings.pickle"):
            with open("./content/embeddings.pickle", 'rb') as f:
                embeddings = pickle.load(f)
            with open("./content/all_packages_metadata.pickle", 'rb') as f:
                all_packages_metadata = pickle.load(f)
        else:
            self.update_all_embedding_metadata()
            with open("./content/embeddings.pickle", 'rb') as f:
                embeddings = pickle.load(f)

        # Create embedding for new resource
        new_embedding = self.create_embedding_from_list([new_resource_metadata])

        # Set up FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        index.add(embeddings)  # Add existing embeddings to the index

        # Search for nearest neighbors
        distances, indices = index.search(new_embedding, k=4)#len(embeddings))  # k is the number of nearest neighbors

        similar_resources = []
        for i, distance in zip(indices[0], distances[0]):
#            if distance < threshold:
            similar_resource = all_packages_metadata[i]
            similar_resource["distance"] = distance  # Inverse of distance as score
            similar_resources.append(similar_resource)

        if similar_resources:
            sorted_data = sorted(similar_resources, key=lambda x: x['distance'], reverse=False)
            return sorted_data
        else:
            return "No similar resource found."

    def check_similarity_all(self, new_resource_metadata, threshold=0.7, numeric=False):
        all_packages_metadata = ""
        numeric_dict, text_dict = self.separate_fields(new_resource_metadata)
        # Load embeddings and metadata
        if os.path.exists("./content/embeddings.pickle"):
            with open("./content/embeddings.pickle", 'rb') as f:
                embeddings = pickle.load(f)
            with open("./content/all_packages_metadata.pickle", 'rb') as f:
                all_packages_metadata = pickle.load(f)
        else:
            self.update_all_embedding_metadata()
            with open("./content/embeddings.pickle", 'rb') as f:
                embeddings = pickle.load(f)
            with open("./content/all_packages_metadata.pickle", 'rb') as f:
                all_packages_metadata = pickle.load(f)

        # Load normalized numeric values if numeric=True
        if numeric and os.path.exists("./content/normalized_numeric_values.pickle"):
            with open("./content/normalized_numeric_values.pickle", 'rb') as f:
                normalized_numeric_values = pickle.load(f)
        else:
            normalized_numeric_values = None

        # Create embedding for new resource (text-based similarity)
        new_embedding = self.create_embedding_from_list([text_dict])

        # Set up FAISS index for text-based similarity
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        index.add(embeddings)  # Add existing embeddings to the index

        # Search for nearest neighbors in the text embedding space
        distances, indices = index.search(new_embedding, k=4)

        # Prepare the list of similar resources
        similar_resources = []
        for i, distance in zip(indices[0], distances[0]):
            similar_resource = all_packages_metadata[i]
            similar_resource["distance"] = distance  # Text similarity distance (lower is more similar)

            # If numeric similarity is requested, compute it
            if numeric and normalized_numeric_values is not None:
                new_numeric_vector = np.array(list(numeric_dict.values())).reshape(1, -1)
                numeric_similarity = cosine_similarity(new_numeric_vector, normalized_numeric_values)
                similar_resource["numeric_similarity"] = numeric_similarity[0][0]
            else:
                similar_resource["numeric_similarity"] = None  # If no numeric similarity is computed

            similar_resources.append(similar_resource)

        # Sort results by text similarity distance (ascending order, closest first)
        if similar_resources:
            sorted_data = sorted(similar_resources, key=lambda x: x['distance'], reverse=False)
            # Apply threshold for filtering results (based on text distance)
            filtered_results = [res for res in sorted_data if res['distance'] > 0]
            return filtered_results if filtered_results else "No similar resource found."
        else:
            return "No similar resource found."

    def upload_file_to_dataset(self, dataset_id, file):
        try:
            dataset = self.backend.action.package_show(id=dataset_id)
            resource_data = {
                'package_id': dataset_id,
                'name': file.filename,
                'format': file.filename.split('.')[-1],
            }
            resource_id = self.backend.action.resource_create(**resource_data)
            self.backend.action.resource_patch(id=resource_id['id'], upload=(file.filename, file.file))
            return f"File '{file.filename}' uploaded successfully to dataset '{dataset['name']}."
        except Exception as e:
            return e

    def create_backend_package(self, package_name, package_title, organization_name, package_notes):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            package_data = {
                'name': package_name,
                'title': package_title,
                'notes': package_notes,
                'owner_org': organization['id'],
            }
            self.backend.action.package_create(**package_data)
            return f"Package '{package_name}' created successfully."
        except Exception as e:
            return e

    def create_backend_package_custom(self, organization_name, package_data):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            package_data["owner_org"] = organization['id']
            self.backend.action.package_create(**package_data)
            return f"Package '{package_data['name']}' created successfully."
        except Exception as e:
            return e

    def update_backend_package(self, package_id, package_name, package_title, organization_name, package_notes):
        try:
            organization = self.backend.action.organization_show(id=organization_name)
            package_data = {
                'id': package_id,
                'name': package_name,
                'title': package_title,
                'notes': package_notes,
                'owner_org': organization['id'],
            }
            self.backend.action.package_update(**package_data)
            return f"Package '{package_name}' updated successfully."
        except Exception as e:
            return e

    def delete_package(self, package_id):
        try:
            self.backend.action.package_delete(id=package_id)
            print(f"Package {package_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting package: {e}")
