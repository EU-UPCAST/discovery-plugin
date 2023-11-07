import ckanapi
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import config

class Similarity:

    def __init__(self):
        # Initialize CKAN client
        self.ckan = ckanapi.RemoteCKAN(config.ckan_api_url, apikey=config.ckan_api_key)

    # Retrieve metadata for all resources
    def get_all_resources_metadata(self):
        # Get a list of packages (datasets)
        packages = self.ckan.action.package_list()

        all_resources_metadata = []

        for package_id in packages:
            # Get the resources for a package
            resources = self.ckan.action.package_show(id=package_id)['resources']
            all_resources_metadata.extend([resource['description'] for resource in resources if 'description' in resource])

        return all_resources_metadata

    def get_all_packages_metadata(self, excluded_id):
        # Get a list of packages (datasets)
        packages = self.ckan.action.package_list()

        all_resources_metadata = []

        for package_id in packages:
            # Get the resources for a package
            if package_id != excluded_id:
                resources = self.ckan.action.package_show(id=package_id)
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
        # Display the embeddings
        # for emb in embeddings_list:
        #     print(emb)
    # Function to check similarity for a new resource
    def check_similarity(self,new_resource_metadata, embeddings, all_packages_metadata, threshold=0.8):
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

        similar_resources = []
        for i in range(0,similarities[0].size):
            if i>=5:
                break
            if similarities[0][i]>threshold:
                similar_resource = all_packages_metadata[i]
                similar_resource["score"] = similarities[0][i]
                similar_resources.append(similar_resource)
        if len(similar_resources)>0:
            sorted_data = sorted(similar_resources, key=lambda x: x['score'], reverse=True)
            return sorted_data
        else:
            return "No similar resource found."

# Get all resources metadata and create embeddings
# all_resources_metadata = get_all_resources_metadata()
# print(similarity_result)
