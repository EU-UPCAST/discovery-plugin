import os
import pickle
from typing import List, Dict, Any

import httpx
import uvicorn
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException, Header, Depends
from pydantic import BaseModel, Field

import config
from backend import Backend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title = 'UPCAST Publish Plugin API v2', description="UPCAST Discovery Plugin API Endpoints to Publish Datasets to UPCAST Discovery Plugin repository",
              root_path="/publish-api")

# Allow all origins to access your API (you can configure this as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_token(apitoken: str = Header(None)):
    if apitoken != config.backend_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API token")

class DatasetItem(BaseModel):
    package_name: str = Form(...),
    package_title: str = Form(...),
    organization_name: str = Form(...),
    package_notes: str = Form(...),

class ResourceItem(BaseModel):
    dataset_id: str
    file: UploadFile = UploadFile(...)

class SimilarItem(BaseModel):
    id: str = Field(title="id", description="The unique id of the dataset", example="dataset-id")
    text: str = Field(title="text", description="The unique id of the dataset", example="The description text")
    score: str = Field(title="score", description="The similarity score", example="0.91")

# region Backend standard calls

@app.post("/catalog/create_dataset/", dependencies=[Depends(verify_api_token)])
async def create_dataset(
        package_name: str = Form(...),
        package_title: str = Form(...),
        organization_name: str = Form(...),
        package_notes: str = Form(...),
):
    backend = Backend()
    return backend.create_backend_package(package_name, package_title, organization_name, package_notes)

def publish_nokia(body):
    import requests

    # Step 1: Get the authentication token
    auth_url = config.integration_nokia['url']
    auth_payload = config.integration_nokia['auth']
    auth_headers = {"Content-Type": "application/json"}

    auth_response = requests.post(auth_url, json=auth_payload, headers=auth_headers)

    if auth_response.status_code == 200:
        auth_data = auth_response.json()
        token = auth_data.get("token")  # Update key name if different
        print("Authentication successful. Token:", token)
    else:
        print("Failed to authenticate:", auth_response.text)
        exit()

    # Step 2: Use the token to send data to the streams endpoint
    streams_url = "https://upcast.dataexchange.nokia.com//streams/streams"
    streams_payload = {
        "url": "http://68k.news/",
        "visibility": "public",
        "name": "upcastmetatest_semih",
        "type": "metatest",
        "description": "tetsing meta data 3",
        "snippet": "{}",
        "price": 1000000,
        "location": {
            "type": "Point",
            "coordinates": [20, 44]
        },
        "terms": "https://gdpr-info.eu/",
        "external": False,
        "subcategory": "66605a587b0d2a883f28cdfc",
        "metadata": {
            "tags": [
                {
                    "@context": {
                        "upcast": "https://www.upcast-project.eu/upcast-vocab/1.0/",
                        "dcat": "http://www.w3.org/ns/dcat#",
                        "foaf": "http://xmlns.com/foaf/0.1/",
                        "idsa-core": "https://w3id.org/idsa/core/",
                        "dct": "http://purl.org/dc/terms/",
                        "odrl": "http://www.w3.org/ns/odrl/"
                    },
                    "@graph": [
                        {
                            "@id": "http://upcast-project.eu/distribution/example-distribution-dataset-1",
                            "@type": "dcat:Distribution",
                            "dct:description": "Example Distribution of a Dataset",
                            "dct:format": "csv",
                            "dct:title": "CSV Distribution of Dataset 1",
                            "dcat:byteSize": 346,
                            "dcat:mediaType": "text/csv"
                        },
                        {
                            "@id": "http://upcast-project.eu/dataset/example-dataset-1",
                            "@type": "dcat:Dataset",
                            "dct:title": "Example Dataset",
                            "dct:description": "Example of a Dataset showing a minimal set of properties for Usage Constraints",
                            "dct:publisher": {
                                "@id": "https://upcast-project.eu/producer/example-data-provider"
                            },
                            "idsa-core:Provider": {
                                "@id": "https://upcast-project.eu/producer/example-data-provider"
                            },
                            "dcat:distribution": {
                                "@id": "http://upcast-project.eu/dataset/example-dataset-1"
                            },
                            "odrl:hasPolicy": {
                                "@id": "http://upcast-project.eu/policy/usage-constraint-example"
                            }
                        },
                        {
                            "@id": "https://upcast-project.eu/provider/example-data-provider",
                            "@type": [
                                "foaf:Agent",
                                "foaf:Organization"
                            ],
                            "foaf:name": "Data Provider Organization"
                        },
                        {
                            "@id": "http://upcast-project.eu/policy/usage-constraint-example",
                            "@type": "odrl:Offer",
                            "odrl:permission": {
                                "odrl:action": {
                                    "@id": "odrl:aggregate"
                                },
                                "odrl:assigner": {
                                    "@id": "https://upcast-project.eu/provider/example-data-provider"
                                },
                                "odrl:target": {
                                    "@id": "http://upcast-project.eu/dataset/example-dataset-1"
                                }
                            },
                            "odrl:prohibition": {
                                "odrl:action": {
                                    "@id": "odrl:use"
                                },
                                "odrl:assigner": {
                                    "@id": "https://upcast-project.eu/provider/example-data-provider"
                                },
                                "odrl:assignee": {
                                    "@id": "http://data-space-vocabulary/classes/AI-Agent"
                                },
                                "odrl:target": {
                                    "@id": "http://upcast-project.eu/dataset/example-dataset-1"
                                }
                            }
                        }
                    ]
                }
            ]
        }
    }
    streams_headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }

    streams_response = requests.post(streams_url, json=streams_payload, headers=streams_headers)

    if streams_response.status_code == 200:
        print("Stream data posted successfully:", streams_response.json())
    else:
        print("Failed to post stream data:", streams_response.text)

@app.post("/catalog/create_dataset_with_custom_fields/", dependencies=[Depends(verify_api_token)])
async def create_dataset_with_custom_fields(body: Dict[str, Any]):
    backend = Backend()
    try:
        marketplace = ''
        package_response = backend.create_backend_package_custom(body)
        for ex in body['extras']:
            if ex['key']=='marketplace':
                marketplace = ex['value']
        try:
            if marketplace == 'nokia':
                publish_nokia(body)
        except:
            pass

        return backend.create_backend_package_custom(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/catalog/update_dataset/", dependencies=[Depends(verify_api_token)])
async def update_dataset(
        package_id: str = Form(...),
        package_name: str = Form(...),
        package_title: str = Form(...),
        organization_name: str = Form(...),
        package_notes: str = Form(...),
):
    backend = Backend()
    return backend.update_backend_package(package_id, package_name, package_title, organization_name, package_notes)

@app.post("/catalog/delete_dataset/", dependencies=[Depends(verify_api_token)])
async def delete_dataset(
        package_id: str = Form(...)
):
    backend = Backend()
    resp = backend.delete_package(package_id)
    return "Package {} deleted successfully.".format(package_id)

@app.post("/catalog/upload_data/", dependencies=[Depends(verify_api_token)])
async def upload_file(dataset_id: str,
    file: UploadFile = UploadFile(...)):
    backend = Backend()
    return backend.upload_file_to_dataset(dataset_id, file)


if __name__ == "__main__":
    uvicorn.run("main-publish-secure:app", host="127.0.0.1", port=8000, reload=True)

