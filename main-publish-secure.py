import json
import os
import pickle
from typing import List, Dict, Any

import httpx
import uvicorn
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException, Header, Depends
import requests
from pydantic import BaseModel, Field

import config
from backend import Backend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from rdflib import Graph
from confluent_kafka import Producer, KafkaException
from datetime import datetime, timedelta, timezone


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
    if apitoken != config.service_api_key:
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

def publish_nokia(upcast_object):
    upcast_object_graph = Graph().parse(data=upcast_object.replace("'",'"'), format="json-ld")
    upcast_object_json = upcast_object
    try:
        # Step 1: Get the authentication token
        auth_url = config.integration_nokia['url']
        auth_payload = config.integration_nokia['auth']
        auth_headers = {"Content-Type": "application/json"}

        auth_response = requests.post(
            auth_url,
            json=auth_payload,
            headers=auth_headers,
            verify=False
        )
        if str(auth_response.status_code)[0] == "2":
            auth_data = auth_response.json()
            token = auth_data.get("token")  # Update key name if different
            print("Authentication successful. Token:", token)
        else:
            print("Failed to authenticate:", auth_response.text)
            exit()

        # Step 2: Use the token to send data to the streams endpoint
        streams_url = "https://upcast.dataexchange.nokia.com//streams/streams"
        streams_payload = {
            "url": "http://example.org/",
            "visibility": "public",
            "name": "UPCAST metatest",
            "type": "metatest",
            "description": "testing meta data 3",
            "snippet": "{}",
            "price": 0,
            "location": {
                "type": "Point",
                "coordinates": [20, 44]
            },
            "terms": "https://gdpr-info.eu/",
            "external": False,
            "subcategory": "66605a587b0d2a883f28cdfc",
            "metadata": {
                "tags": [
                    upcast_object_json
                ]
            }
        }
        query = """
        PREFIX dcat: <http://www.w3.org/ns/dcat#>

        SELECT ?id ?p ?o ?type
        WHERE {
          ?s a ?type ;
             ?p ?o .
          BIND(STR(?s) AS ?id)
        }
        """

        # Execute the query and assign the result
        results = upcast_object_graph.query(query)
#         = [(str(row.id),str(row.desc),str(row.type)) for row in results]
        url = ""
        desc = ""
        price = 0
        title = ""
        for row in results:
            pass
        for row in results:
            if "distribution" in str(row.type).lower():
                url = row.id
                if "description" in str(row.p).lower():
                    desc = row.o
                if "price" in str(row.p).lower():
                    price = row.o
                if "title" in str(row.p).lower():
                    title = row.o
            elif "dataset" in  str(row.type).lower():
                if url == "":
                    url = row.id
                if "description" in str(row.p).lower() and desc == "":
                    desc = row.o
                if "price" in str(row.p).lower() and price == 0:
                    price = row.o
                if "title" in str(row.p).lower() and title == 0:
                    title = row.o

        streams_payload["url"] = str(url)
        streams_payload["price"] = price.value
        streams_payload["name"] = str(title)
        streams_payload["description"] = str(desc)


        streams_headers = {
            "Content-Type": "application/json",
            "Authorization": token
        }

        streams_response = requests.post(streams_url, json=streams_payload, headers=streams_headers,
            verify=False)

        if str(auth_response.status_code)[0] == "2":
            print("Stream data posted successfully:", streams_response.json())
        else:
            print("Failed to post stream data:", streams_response.text)
    except BaseException as b:
        print(b)
@app.post("/catalog/create_dataset_with_custom_fields/", dependencies=[Depends(verify_api_token)])
async def create_dataset_with_custom_fields(body: Dict[str, Any]):
    backend = Backend()
    try:
        marketplace = ''
        package_response = backend.create_backend_package_custom(body)
        for ex in body['extras']:
            if ex['key']=='marketplace':
                marketplace = ex['value']
            if ex['key']=='upcast':
                upcast_object = ex['value']
                try:
                    upcast_object_graph = Graph().parse(data=upcast_object.replace("'",'"'), format="json-ld")
                except:
                    raise HTTPException(status_code=400, detail="UPCAST object could not be parsed")
        try:
            if 'created successfully' in package_response:
                if marketplace == 'nokia' and 'created successfully' in package_response:
                    publish_nokia(upcast_object)

                await push_kafka_message("publishing-plugin", "publish", marketplace, body)
        except BaseException as b:
            pass
        return package_response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e.detail))


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

# Define Kafka broker configuration
kafka_conf = {'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS")}
kafka_enabled = False
push_consumer_enabled = True
push_provider_enabled = True
# Create Producer instance
try:
    producer = Producer(kafka_conf)
    producer.list_topics(timeout=2)  # Set a timeout to avoid blocking indefinitely
    kafka_enabled = True
except KafkaException as e:
    print(f"Kafka Producer error: {e}")
    kafka_enabled = False
    producer = None  # Ensure the producer is not used further

async def push_kafka_message(topic, action, marketplace_id, message):
    try:
        if kafka_enabled:
            if action == "create":
                action = action
            else:
                action = message["negotiation_status"]
            message_json = {
                "source": topic,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "did": message["dataset_id"],
                "marketplace_id": marketplace_id,
                "update": message
            }
            producer.produce(topic, str(message_json).encode('utf-8'), callback=delivery_report)
            producer.flush()  # Ensure all messages are delivered before exiting
    except BaseException as b:
        print(b)
        pass

def delivery_report(err, msg):
    """ Callback for message delivery report """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")
if __name__ == "__main__":
    uvicorn.run("main-publish-secure:app", host="127.0.0.1", port=8000, reload=True)

