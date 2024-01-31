import ckanapi
import torch
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

    def get_all_packages_metadata(self, excluded_id):
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

    # Function to check similarity for a new resource
    def check_similarity(self,new_resource_metadata, embeddings, all_packages_metadata, threshold=0.7):
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
        threshold = sorted(similarities[0])[-2] - 0.02
        similar_resources = []
        for i in range(0,similarities[0].size):
            if i>=5:
                break
            if similarities[0][i]>threshold and similarities[0][i]!=1:
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

