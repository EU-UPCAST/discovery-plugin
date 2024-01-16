import requests
import config

# backend API endpoint for dataset search
api_url = [config.backend_api_url] # Replace 'Backend_URL' with your actual API key

# backend API key
api_key = [config.backend_api_key]  # Replace 'YOUR_API_KEY_HERE' with your actual API key

# Parameters for the search query
params = {
    'q': 'health',  # Search term, change this to your desired keyword
    'rows': 10,      # Number of results to return
    'start': 0       # Index of the first result
    #'url':'api/3/action/package_search'
}

# Top 10 tags and vocabulary tags used by datasets
params = {
    'facet.field':["tags"],
    'facet.limit':10,
    'rows': 0

}

# All datasets that have tag ‘economy’
params = {
    'fq':'tags:economy'
}

# Headers with the API key
headers = {
    'Authorization': api_key
}

# Making the API request with headers
response = requests.get(api_url, json=params, headers=headers)

if response.status_code == 200:
    results = response.json()

    # Displaying information about datasets
    datasets = results['result']['results']
    for dataset in datasets:
        print(f"Dataset Name: {dataset['title']}")
        print(f"Organization: {dataset['organization']['title']}")
        print(f"URL: {dataset['url']}")
        print("------------------------------------")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
