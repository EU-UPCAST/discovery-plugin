import os
import pickle
from typing import List

import httpx
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field

import config
from backend import Backend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title = 'UPCAST Discovery Plugin API')

# Allow all origins to access your API (you can configure this as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
#     return {"message": "This is the API service for UPCAST Discovery Plugin"}
@app.get("/ui/discover_ui")
async def get_search_page():
    # You can also specify media type explicitly
    return FileResponse("ui/search.html", media_type="text/html")


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
    backend = Backend()
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

    response = backend.backend.action.package_search(**data_dict)

    return response

@app.get("/discover/resource_search")
async def resource_search(
        query: str = Query(..., description="The search criteria (e.g., field:term)"),
        order_by: str = Query(None, description="A field to order the results"),
        offset: int = Query(None, description="Apply an offset to the query"),
        limit: int = Query(None, description="Apply a limit to the query"),
):
    backend = Backend()

    # Construct the data_dict based on the provided parameters
    data_dict = {
        "query": query,
        "order_by": order_by,
        "offset": offset,
        "limit": limit,
    }
    response = backend.backend.action.resource_search(**data_dict)
    return response

@app.get("/discover/dataset_show_resources")
async def dataset_show(
        dataset_name: str
):
    backend = Backend()

    response = backend.backend.action.package_show(id=dataset_name)
    return response

# endregion


@app.post("/discover/discover_similar_datasets", response_model=List[SimilarItem])
async def discover_similar_datasets(dataset_id: str = Query(None, description="Unique dataset ID")):
    backend = Backend()

    # Example: Check similarity for a new resource
    resources = backend.backend.action.package_show(id=dataset_id)

    new_resource_metadata = resources['notes']
    similarity_result = backend.check_similarity(new_resource_metadata)
    res = []
    try:
        for i in range(len(similarity_result)):
            res.append(SimilarItem(id=similarity_result[i]["id"],text=similarity_result[i]["text"],score=similarity_result[i]["score"]))
        return res
    except:
        return {"result":"no similar resources found"}

@app.get("/discover/create_embeddings")
async def create_embeddings():
    backend = Backend()
    backend.update_embedding_metadata()
    return {"result":"embedding updated"}

@app.post("/discover/discover_similar_datasets_description", response_model=List[SimilarItem])
async def discover_similar_datasets_description(description: str = Query("sample description", description="Dataset description")):
    backend = Backend()

    similarity_result = backend.check_similarity(description)
    res = []
    try:
        for i in range(len(similarity_result)):
            res.append(SimilarItem(id=similarity_result[i]["id"],text=similarity_result[i]["text"],score=similarity_result[i]["score"]))
        return res
    except:
        return {"result":"no similar resources found"}
