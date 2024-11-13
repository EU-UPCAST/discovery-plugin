import os
import pickle
from typing import List

import httpx
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException, Header, Depends
from pydantic import BaseModel, Field

import config
from backend import Backend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title = 'UPCAST Publish Plugin API')

# Allow all origins to access your API (you can configure this as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_token(api_token: str = Header(None)):
    if api_token != config.backend_api_key:
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

# @app.get("/")
# async def root():
#     return {"message": "This is the API service for UPCAST Publish Plugin"}

# @app.get("/discover/translational_search")
# async def translational_search(q: str, key: str = ""):
#     client = GPT(key)
#     text = "detect the language translate it to french, german, turkish, english: "
#
#     trans_text = client.ask_gpt("Translate", text)
#     q = {
#         "q": trans_text,
#         "fl": "title, description, tags, language",
#         "fq": "res_format:json",
#         "rows": 10
#     }
#     # TODO send the query to Backend, not sure if this works
#
#     # Construct the Backend API URL
#     url = f"{config.backend_api_url}resource_search"
#
#     return mirror(url, q)


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

@app.post("/catalog/create_dataset_from_resource_spec/", dependencies=[Depends(verify_api_token)])
async def create_dataset_from_resource_spec(organization_name: str,
    file: UploadFile = UploadFile(...)):
    backend = Backend()
    # TODO create a custom logic to convert a resource into a ckan package with custom fields
    package_data = {
        'name': "package_name",
        'title': "package_title",
        'notes': "package_notes"
    }
    return HTTPException(status_code=404, detail="Not yet implemented")
    # return backend.create_backend_package_custom(organization_name, package_data)

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
