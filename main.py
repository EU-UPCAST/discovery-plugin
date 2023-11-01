import json

from fastapi import FastAPI, Query, Request
import httpx
import requests

from GPT import GPT

key = ""
app = FastAPI()
CKAN_API_BASE_URL = 'http://62.171.168.208:5000/'

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/discover/workflow_search")
async def workflow_search(q: str = Query("*:*", description="The solr query"),key: str =""):
    client = GPT(key)
    return(client.ask_gpt("Workflow", q))


@app.get("/discover/package_search")
async def package_search(
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
        use_default_schema: bool = Query(False, description="Use default package schema instead of a custom schema"),
):
    # Construct the CKAN API URL
    url = f"{CKAN_API_BASE_URL}package_search"

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

    # Make a request to the CKAN API
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data_dict)

    if response.status_code == 200:
        result = response.json()
        if result.get('success', False):
            return result['result']
        else:
            return {"error": "CKAN API request was not successful."}
    else:
        return {"error": "CKAN API request failed."}

@app.get("/discover/resource_search")
async def resource_search(
        query: str = Query(..., description="The search criteria (e.g., field:term)"),
        order_by: str = Query(None, description="A field to order the results"),
        offset: int = Query(None, description="Apply an offset to the query"),
        limit: int = Query(None, description="Apply a limit to the query"),
):
    # Construct the CKAN API URL
    url = f"{CKAN_API_BASE_URL}resource_search"

    # Construct the data_dict based on the provided parameters
    data_dict = {
        "query": query,
        "order_by": order_by,
        "offset": offset,
        "limit": limit,
    }
    return mirror(url,data_dict)
    # Remove None values from data_dict
    data_dict = {key: value for key, value in data_dict.items() if value is not None}


@app.get("/package_list")
async def get_package_list(limit: int = None, offset: int = None):
    # Construct the CKAN API URL
    url = f"{CKAN_API_BASE_URL}package_list"

    # Construct the data_dict based on the provided parameters
    data_dict = {}
    if limit is not None:
        data_dict['limit'] = limit
    if offset is not None:
        data_dict['offset'] = offset

    return mirror(url,data_dict)

@app.get("/current_package_list_with_resources")
async def get_current_package_list_with_resources(limit: int = None, offset: int = None, page: int = None):
    # Construct the CKAN API URL
    url = f"{CKAN_API_BASE_URL}current_package_list_with_resources"

    # Construct the data_dict based on the provided parameters
    data_dict = {}
    if limit is not None:
        data_dict['limit'] = limit
        if page is not None:
            data_dict['offset'] = (page - 1) * limit  # Calculate offset based on page number
    elif offset is not None:
        data_dict['offset'] = offset

    return mirror(url,data_dict)

@app.get("/mirror")
async def mirror(request: Request):
    try:

        # Extract request data
        content = await request.body()
        content_dict = content.decode("utf-8")
        json_data = json.loads(content_dict)

        url = f"{CKAN_API_BASE_URL}{json_data['url']}"
        del json_data['url']
        # Forward request to CKAN instance
        response = requests.post(url, json=json_data)

        return response.json()
    except BaseException as b:
        return {"error": str(b)}