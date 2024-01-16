import json
from fastapi import FastAPI, Query, Request, UploadFile, File, Form, HTTPException, Response
import httpx
import requests
from pydantic import BaseModel, Field
import config
from GPT import GPT
from similarity import Similarity
from typing import List

app = FastAPI(title = 'UPCAST Discovery Plugin API')


class DatasetItem(BaseModel):
    dataset_id: str = Query(None, description="Unique dataset ID")

class ResourceItem(BaseModel):
    resource_id: str = Query(None, description="Unique resource ID")


class SimilarItem(BaseModel):
    id: str = Field(title="id", description="The unique id of the dataset", example="dataset-id")
    text: str = Field(title="text", description="The unique id of the dataset", example="The description text")
    score: str = Field(title="score", description="The similarity score", example="0.91")


@app.get("/")
async def root():
    return {"message": "This is the API service for UPCAST Discovery Plugin"}

#
# @app.get("/discover/workflow_search")
# async def workflow_search(q: str = Query("*:*", description="The solr query"), gpt_key: str = ""):
#     client = GPT(config.gpt_key)
#     return (client.ask_gpt("Workflow", q))


@app.post("/discover/discover_similar_datasets", response_model=List[SimilarItem])
async def discover_similar_datasets(item: DatasetItem):
    sim = Similarity()

    all_packages_metadata = sim.get_all_packages_metadata(item.dataset_id)
    embeddings = sim.create_embedding_from_list(all_packages_metadata)

    # Example: Check similarity for a new resource
    resources = sim.backend.action.package_show(id=item.dataset_id)

    new_resource_metadata = resources['notes']
    similarity_result = sim.check_similarity(new_resource_metadata, embeddings, all_packages_metadata)
    res = []
    for i in range(len(similarity_result)):
        res.append(SimilarItem(id=similarity_result[i]["id"],text=similarity_result[i]["text"],score=similarity_result[i]["score"]))
    #     del similarity_result[i]["text"]
    return res

#
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

@app.get("/discover/dataset_search")
async def dataset_search(
        q: str = Query("*:*", description="The solr query"),
        fq: str = Query(None, description="Any filter queries to apply"),
        fq_list: list = Query(None, description="Additional filter queries to apply"),
        sort: str = Query("score desc, metadata_modified desc", description="Sorting of the search results"),
        rows: int = Query(10, description="The maximum number of matching rows to return"),
        start: int = Query(None,
                           description="The offset in the complete result for where the returned datasets should begin"),
        facet: bool = Query(True, description="Whether to enable faceted results"),
        facet_mincount: int = Query(None,
                                    description="The minimum counts for facet fields to be included in the results"),
        facet_limit: int = Query(50, description="The maximum number of values the facet fields return"),
        facet_field: list = Query(None, description="The fields to facet upon"),
        include_drafts: bool = Query(False, description="Include draft datasets"),
        include_private: bool = Query(False, description="Include private datasets"),
        use_default_schema: bool = Query(False, description="Use default dataset schema instead of a custom schema"),
):
    # Construct the Backend API URL
    url = f"{config.backend_api_url}package_search"

    # Construct the data_dict based on the provided parameters
    data_dict = {
        "q": q,
        "fq": fq,
        "fq_list": fq_list,
        "sort": sort,
        "rows": rows,
        "start": start,
        "facet": facet,
        "facet.mincount": facet_mincount,
        "facet.limit": facet_limit,
        "facet.field": facet_field,
        "include_drafts": include_drafts,
        "include_private": include_private,
        "use_default_schema": use_default_schema,
    }

    # Remove None values from data_dict
    data_dict = {key: value for key, value in data_dict.items() if value is not None}

    # Make a request to the Backend API
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data_dict)

    if response.status_code == 200:
        result = response.json()
        if result.get('success', False):
            return result['result']
        else:
            return {"error": "API request was not successful."}
    else:
        return {"error": "API request failed."}


@app.get("/discover/resource_search")
async def resource_search(
        query: str = Query(..., description="The search criteria (e.g., field:term)"),
        order_by: str = Query(None, description="A field to order the results"),
        offset: int = Query(None, description="Apply an offset to the query"),
        limit: int = Query(None, description="Apply a limit to the query"),
):
    # Construct the Backend API URL
    url = f"{config.backend_api_url}resource_search"

    # Construct the data_dict based on the provided parameters
    data_dict = {
        "query": query,
        "order_by": order_by,
        "offset": offset,
        "limit": limit,
    }
    return mirror(url, data_dict)
    # Remove None values from data_dict
    # data_dict = {key: value for key, value in data_dict.items() if value is not None}


@app.get("/discover/dataset_list")
async def dataset_list(limit: int = None, offset: int = None):
    # Construct the Backend API URL
    url = f"{config.backend_api_url}package_list"

    # Construct the data_dict based on the provided parameters
    data_dict = {}
    if limit is not None:
        data_dict['limit'] = limit
    if offset is not None:
        data_dict['offset'] = offset

    return mirror(url, data_dict)

#
# @app.get("/discover/current_dataset_list_with_resources")
# async def current_dataset_list_with_resources(limit: int = None, offset: int = None, page: int = None):
#     # Construct the Backend API URL
#     url = f"{config.backend_api_url}current_package_list_with_resources"
#
#     # Construct the data_dict based on the provided parameters
#     data_dict = {}
#     if limit is not None:
#         data_dict['limit'] = limit
#         if page is not None:
#             data_dict['offset'] = (page - 1) * limit  # Calculate offset based on page number
#     elif offset is not None:
#         data_dict['offset'] = offset
#
#     return mirror(url, data_dict)

#
# # external generic mirror is not working yet
# @app.get("/mirror")
# async def external_mirror(request: Request):
#     try:
#         # Extract and pass headers to the outgoing request
#         incoming_headers = request.headers.items()
#         headers = {key: value for key, value in incoming_headers}
#
#         # Extract request data
#         content = await request.body()
#         content_dict = content.decode("utf-8")
#         json_data = json.loads(content_dict)
#
#         url = f"{config.backend_api_url}{json_data['url']}"
#         del json_data['url']
#         # Forward request to Backend instance
#         response = requests.post(url, json=json_data, headers=headers)
#
#         return response.json()
#     except BaseException as b:
#         return {"error": str(b)}


# endregion

@app.post("catalog/upload_data/")
async def upload_resource(file: UploadFile = File(...), resource_name: str = Form(...), dataset_id: str = Form(...)):
    url = f"{config.backend_api_url}upload_data"

    files = {
        'upload': (file.filename, file.file)
    }

    headers = {
        'Authorization': config.backend_api_key
    }

    data = {
        'package_id': dataset_id,
        'name': resource_name
    }

    response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code == 200:
        return {"detail": "File uploaded successfully"}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


@app.post("catalog/create_dataset/")
async def create_dataset(name: str = Form(...), title: str = Form(...), notes: str = Form(...)):
    # Construct the Backend API URL
    url = f"{config.backend_api_url}package_create"

    headers = {
        'Authorization': config.backend_api_key,
        'Content-Type': 'application/json',
    }

    data = {
        'name': name,
        'title': title,
        'notes': notes,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return {"detail": "Dataset created successfully"}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


async def mirror(url, data_dict):
    # Make a request to the Backend API
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data_dict)

    if response.status_code == 200:
        result = response.json()
        if result.get('success', False):
            return result['result']
        else:
            return {"error": "Backend API request was not successful."}
    else:
        return {"error": "Backend API request failed."}
