import os
import json
from typing import Dict
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from fastapi import FastAPI, Query, UploadFile, Form, HTTPException, Header, Depends
import config

def verify_api_token(apitoken: str = Header(None)):
    if apitoken != config.service_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API token")

# Define the router for marketplace credentials management
router = APIRouter(
    prefix="/marketplace_credentials",
    tags=["marketplace_credentials"],
    dependencies=[Depends(verify_api_token)]
)

# Path to the credentials file
CREDENTIALS_FILE = "marketplace_credentials.json"


# Pydantic model for credentials
class MarketplaceCredentials(BaseModel):
    marketplace_id: str
    url: str
    username: str
    password: str

# Helper functions for credential management
def read_credentials() -> Dict:
    """Read credentials from file"""
    if not os.path.exists(CREDENTIALS_FILE):
        return {}

    try:
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def write_credentials(credentials: Dict) -> None:
    """Write credentials to file"""
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=2)


# Endpoints
@router.post("/")
async def add_or_update_credentials(credentials: MarketplaceCredentials):
    """
    Add or update marketplace credentials
    """
    creds_data = read_credentials()

    # Store by marketplace URL as the key
    marketplace_id = credentials.url.replace("https://", "").replace("http://", "").split("/")[0]
    marketplace_id = credentials.marketplace_id
    creds_data[marketplace_id] = credentials.dict()

    write_credentials(creds_data)
    return {"message": f"Credentials for {marketplace_id} updated successfully"}


@router.get("/")
async def get_all_credentials():
    """
    Get all marketplace credentials
    """
    return read_credentials()


@router.get("/{marketplace_id}")
async def get_credentials(marketplace_id: str):
    """
    Get credentials for a specific marketplace
    """
    creds_data = read_credentials()

    if marketplace_id not in creds_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for {marketplace_id}"
        )

    return creds_data[marketplace_id]


@router.delete("/{marketplace_id}")
async def delete_credentials(marketplace_id: str):
    """
    Delete credentials for a specific marketplace
    """
    creds_data = read_credentials()

    if marketplace_id not in creds_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for {marketplace_id}"
        )

    del creds_data[marketplace_id]
    write_credentials(creds_data)

    return {"message": f"Credentials for {marketplace_id} deleted successfully"}