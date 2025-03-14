import os
import json
from enum import Enum
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

class MarketPlace(str, Enum):
    Nokia = "nokia"
    Dawex = "dawex"
    OKFN = "okfn"

# Pydantic model for credentials
class MarketplaceCredentials(BaseModel):
    account_id: str
    marketplace: MarketPlace
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

    ## Store by marketplace URL as the key
    #account_id = credentials.url.replace("https://", "").replace("http://", "").split("/")[0]
    account_id = credentials.account_id
    creds_data[account_id] = credentials.dict()

    write_credentials(creds_data)
    return {"message": f"Credentials for {account_id} updated successfully"}


@router.get("/")
async def get_all_credentials():
    """
    Get all marketplace credentials
    """
    return read_credentials()


@router.get("/{account_id}")
async def get_credentials(account_id: str):
    """
    Get credentials for a specific marketplace
    """
    creds_data = read_credentials()

    if account_id not in creds_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for {account_id}"
        )

    return creds_data[account_id]


@router.delete("/{account_id}")
async def delete_credentials(account_id: str):
    """
    Delete credentials for a specific marketplace
    """
    creds_data = read_credentials()

    if account_id not in creds_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for {account_id}"
        )

    del creds_data[account_id]
    write_credentials(creds_data)

    return {"message": f"Credentials for {account_id} deleted successfully"}