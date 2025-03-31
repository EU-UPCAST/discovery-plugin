import json
import os
import pickle
from typing import List, Dict, Any
import logging
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

from upcast_extractor import UpcastCkanMetadataExtractor

from marketplace_credentials import router as credentials_router, read_credentials, MarketPlace

# Then add this line with your other app.include_router() calls

logger = logging.getLogger(__name__)
app = FastAPI(title = 'UPCAST Publish Plugin API v2', description="UPCAST Discovery Plugin API Endpoints to Publish Datasets to UPCAST Discovery Plugin repository",
              root_path="/publish-api")
app.include_router(credentials_router)

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

def publish_nokia(upcast_object, extracted_fields, credentials):
    upcast_object_json = upcast_object
    try:
        # Step 1: Get the authentication token
        auth_url = credentials['url'] + "/auth/tokens"
        auth_payload = {"email":credentials['username'], "password":credentials['password']}
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
        streams_url = credentials['url'] + "/streams/streams"
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

        url = extracted_fields["upcast_dataset_uri"]
        desc = extracted_fields["upcast_description"]
        price = extracted_fields["upcast_price"]
        title = extracted_fields["upcast_title"]


        streams_payload["url"] = str(url)
        streams_payload["price"] = float(price)
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

def publish_okfn(upcast_object, extracted_fields, credentials):
    upcast_object_json = upcast_object
    try:
        # Step 1: Get the authentication token
        auth_url = credentials['url'] + "/upcast/resource"
        auth_headers = {"Content-Type": "application/json", "api-key" : credentials['password']}

        response = requests.post(
            auth_url,
            json=upcast_object_json,
            headers=auth_headers,
            verify=False
        )
        if str(response.status_code)[0] == "2":
            print("Stream data posted successfully:", response.json())
        else:
            print("Failed to post stream data:", response.text)

    except BaseException as b:
        print(b)

def ensure_string_values(data: Dict) -> Dict[str, str]:
    """Convert all values in a dictionary to strings."""
    return {k: str(v) if v is not None else "" for k, v in data.items()}

def create_policy_object_from_dataset(negotiation_provider_user_id, dataset, policy, distribution):
    natural_language_document = ""

    dataset["raw_object"] = dataset.copy()
    if "upcast:price" in dataset:
        dataset["price"] = dataset["upcast:price"]
    if "@id" in dataset:
        dataset["uri"] = dataset["@id"]
    if "dct:title" in dataset:
        dataset["title"] = dataset["dct:title"]
    if "upcast:priceUnit" in dataset:
        dataset["price_unit"] = dataset["upcast:priceUnit"]
    if "odrl:hasPolicy" in dataset and "odrl_policy" in dataset and "uid" in dataset["odrl_policy"]:
        dataset["policy_url"] = dataset["odrl_policy"]["uid"]
    if "dct:description" in dataset:
        dataset["description"] = dataset["dct:description"]
    if "@type" in dataset:
        dataset["type_of_data"] = dataset["@type"]
    if "dct:publisher" in dataset and "@id" in dataset["dct:publisher"]:
        dataset["publisher"] = dataset["dct:publisher"]["@id"]
    if distribution is not {}:
        dataset["distribution"] = ensure_string_values(distribution)

    # Request body
    payload_negotiation = {
        "title": dataset['dct:title'],
        "type": "offer",  # This will be overridden to "REQUEST" by the server
        "consumer_id": negotiation_provider_user_id,
        "producer_id": negotiation_provider_user_id,
        "data_processing_workflow_object": {
            # Include your data processing workflow details here
            "workflow_steps": []
        },
        "natural_language_document": natural_language_document,
        "resource_description_object": dataset,
        "odrl_policy": policy
    }

    # Query parameters (the master password is passed as a query parameter)
    params = {
        "master_password_input": config.MASTER_PASSWORD  # Replace with actual master password
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(config.negotiation_url, params=params, headers=headers,
                             data=json.dumps(payload_negotiation))

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        offer_id = result['offer_id']

        print("Offer created successfully!")
        print(f"Offer ID: {result['offer_id']}")
        print(f"Negotiation ID: {result['negotiation_id']}")
        return offer_id
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def create_negotiation_offer(negotiation_provider_user_id, dataset, policy, distribution):
    natural_language_document = ""

    dataset["raw_object"] = dataset.copy()
    if "upcast:price" in dataset:
        dataset["price"] = dataset["upcast:price"]
    if "@id" in dataset:
        dataset["uri"] = dataset["@id"]
    if "dct:title" in dataset:
        dataset["title"] = dataset["dct:title"]
    if "upcast:priceUnit" in dataset:
        dataset["price_unit"] = dataset["upcast:priceUnit"]
    if "odrl:hasPolicy" in dataset and "odrl_policy" in dataset and "uid" in dataset["odrl_policy"]:
        dataset["policy_url"] = dataset["odrl_policy"]["uid"]
    if "dct:description" in dataset:
        dataset["description"] = dataset["dct:description"]
    if "@type" in dataset:
        dataset["type_of_data"] = dataset["@type"]
    if "dct:publisher" in dataset and "@id" in dataset["dct:publisher"]:
        dataset["publisher"] = dataset["dct:publisher"]["@id"]
    if distribution is not {}:
        dataset["distribution"] = ensure_string_values(distribution)

    # Request body
    payload_negotiation = {
        "title": dataset['dct:title'],
        "type": "offer",  # This will be overridden to "REQUEST" by the server
        "consumer_id": negotiation_provider_user_id,
        "producer_id": negotiation_provider_user_id,
        "data_processing_workflow_object": {
            # Include your data processing workflow details here
            "workflow_steps": []
        },
        "natural_language_document": natural_language_document,
        "resource_description_object": dataset,
        "odrl_policy": policy
    }

    # Query parameters (the master password is passed as a query parameter)
    params = {
        "master_password_input": config.MASTER_PASSWORD  # Replace with actual master password
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(config.negotiation_url, params=params, headers=headers,
                             data=json.dumps(payload_negotiation))

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        offer_id = result['offer_id']

        print("Offer created successfully!")
        print(f"Offer ID: {result['offer_id']}")
        print(f"Negotiation ID: {result['negotiation_id']}")
        return offer_id
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


@app.post("/catalog/create_dataset_with_custom_fields_v0/", dependencies=[Depends(verify_api_token)])
async def create_dataset_with_custom_fields_v0(body: Dict[str, Any]):
    backend = Backend()
    try:
        marketplaces = []
        negotiation_provider_user_id = None
        policy = {}
        dataset = {}
        distribution = {}

        for ex in body['extras']:
            # if ex['key']=='marketplace':
            #     marketplaces.append(ex['value'])
            if ex['key']=='marketplace_ids':
                marketplaces = ex['value'].split(",")
            if ex['key']=='natural_language_document':
                natural_language_document = ex['value']
            if ex['key']=='negotiation_provider_user_id':
                negotiation_provider_user_id = ex['value']
            if ex['key']=='upcast':
                upcast_object = ex['value']
                try:
                    if type(upcast_object) is dict:
                        upcast_object_graph = Graph().parse(data=upcast_object, format="json-ld")
                    elif type(upcast_object) is str:
                        upcast_object = json.loads(upcast_object.replace("'",'"'))
                        upcast_object_graph = Graph().parse(data=upcast_object, format="json-ld")
                    for d in upcast_object["@graph"]:
                        if type(d["@type"]) is str:
                            if d["@type"] == "odrl:Offer" or "offer" in d["@type"].lower():
                                policy = d
                            if d["@type"] == "dcat:Dataset" or "dataset" in d["@type"].lower():
                                dataset = d
                            if d["@type"] == "dcat:Distribution" or "distribution" in d["@type"].lower():
                                distribution = d
                    original_context = upcast_object["@context"]
                    print("")
                except BaseException as b:
                    raise HTTPException(status_code=400, detail="UPCAST object could not be parsed")

        if negotiation_provider_user_id is not None:
            offer_id = create_negotiation_offer(negotiation_provider_user_id, dataset, policy, distribution)
            if offer_id is not None:
                body['extras'].append({"key": "negotiation_offer_id", "value": offer_id})

        package_response = backend.create_backend_package_custom(body)

        try:
            cr = read_credentials()
            if 'created successfully' in package_response:
                for marketplace in marketplaces:
                    if marketplace in cr:
                        if cr[marketplace]["marketplace"] == MarketPlace.Nokia:
                            publish_nokia(upcast_object,cr[marketplace])

                kafka_res = push_kafka_message("publishing-plugin", "publish", marketplace, body)
                if kafka_res != True:
                    message = str(kafka_res)
        except BaseException as b:
            message = " publish failed."
            pass
        return str(package_response) + message
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/catalog/create_dataset_with_custom_fields/", dependencies=[Depends(verify_api_token)])
async def create_upcast_dataset(body: Dict[str, Any]):
    backend = Backend()
    try:
        marketplaces = []
        negotiation_provider_user_id = None
        policy = {}
        dataset = {}
        distribution = {}
        message = ""
        upcast_object = None
        upcast_extras = {}

        # First pass to extract basic fields and the UPCAST object
        for ex in body['extras']:
            if ex['key']=='marketplace_account_ids':
                marketplaces = ex['value'].split(",")
            if ex['key'] == 'natural_language_document':
                natural_language_document = ex['value']
            if ex['key'] == 'negotiation_provider_user_id':
                negotiation_provider_user_id = ex['value']
            if ex['key'] == 'upcast':
                upcast_object = ex['value']

        # Process UPCAST data if present
        if upcast_object:
            try:
                # Parse the UPCAST object
                if type(upcast_object) is dict:
                    upcast_data = upcast_object
                elif type(upcast_object) is str:
                    upcast_data = json.loads(upcast_object.replace("'", '"'))

                # Use the simplified extractor to get CKAN extras
                extractor = UpcastCkanMetadataExtractor()
                upcast_extras = extractor.extract_ckan_extras(upcast_data)

                # Add the extracted extras to the body
                body['extras'].extend(upcast_extras)


            except Exception as e:
                logger.error(f"Error processing UPCAST data: {str(e)}")
                raise HTTPException(status_code=400, detail=f"UPCAST object could not be processed: {str(e)}")

        # Create the package in CKAN
        package_response = backend.create_backend_package_custom(body)

        # Handle marketplace publishing
        try:
            if 'created successfully' in package_response:
                cr = read_credentials()

                for marketplace in marketplaces:
                    if marketplace in cr:
                        if cr[marketplace]["type"] == MarketPlace.Nokia.value:
                            publish_nokia(upcast_object, upcast_extras, cr[marketplace])

                        if cr[marketplace]["type"] == MarketPlace.OKFN.value:
                            publish_okfn(upcast_object, upcast_extras, cr[marketplace])

                kafka_res = push_kafka_message("publishing-plugin", "publish", marketplaces, body)
                if kafka_res != True:
                    message = str(kafka_res)
        except Exception as b:
            message = " publish failed."
            logger.error(f"Publishing failed: {str(b)}")
            pass

        return str(package_response) + message
    except Exception as e:
        logger.error(f"Error in create_dataset_with_custom_fields: {str(e)}")
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

# Define Kafka broker configuration
# kafka_conf = {'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS")}

kafka_conf = config.KAFKA_CONFIG
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
    print(str(kafka_conf))
    kafka_enabled = False
    producer = None  # Ensure the producer is not used further

def push_kafka_message(topic, action, marketplace_id, message):
    try:
        if kafka_enabled:
            message_json = {
                "source": topic,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "did": message["name"],
                "marketplace_id": marketplace_id,
                "update": message
            }
            producer.produce(topic, json.dumps(message_json).encode('utf-8'), callback=delivery_report)
            producer.flush()  # Ensure all messages are delivered before exiting
            return True
    except BaseException as b:
        print(b)
        return b

def delivery_report(err, msg):
    """ Callback for message delivery report """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")
if __name__ == "__main__":
    uvicorn.run("main-publish-secure:app", host="127.0.0.1", port=8000, reload=True)

