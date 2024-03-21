# Discovery & Publish Plugin

Welcome to the Discovery Plugin

This service is powered by FastAPI and relies on a robust CKAN backend, datapusher, PostgreSQL database, Solr search, Redis caching, and more.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Docker Compose](#docker-compose)
- [Contributing](#contributing)
- [License](#license)

## Features

- High-performance FastAPI for seamless API handling
- Robust backend with PostgreSQL, Solr, and Redis support
- Datapusher for efficient data pushing
- Easy setup with Docker Compose
- Configuration flexibility through environment variables

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Docker
- Docker Compose

### Installation

1. Clone the repository:

   ```bash
   git clone --recurse-submodules https://github.com/EU-UPCAST/discovery-plugin.git

2. Navigate to the project directory:

    ```bash
    cd discovery-plugin

3. Make sure docker-ckan submodule is cloned, if not, you may force it again by pulling:
    ```bash
    git pull --recurse-submodules
   
3. There is an example environment file in the repository (example.env). Customize your .env file with necessary configurations. Sample .env:

    ```bash
    TZ=UTC
    Backend_PORT=5000
    DATASTORE_READONLY_PASSWORD=myreadonlypassword
    POSTGRES_PASSWORD=mypostgrespassword

## Usage
To run the Discovery Plugin service, execute the following command:

    docker-compose up -d

This will start the service, backend, datapusher, PostgreSQL, Solr, and Redis containers.


## Post-install Configuration
Visit CKAN instance at http://localhost:5001

Create an integration token in CKAN (/user/[user_name]/api-tokens)
=======
## Configuration
Create an integration token in CKAN (/user/[user_name]/api-tokens), and 

Update the config.py with the new token and ckan url:
backend_api_key=[new token]
backend_api_url=[ckan_api_url]

Customize the CKAN related services further by updating the environment variables in the .env file. Modify the configurations as needed for your specific use case.


Update config.py:

    backend_api_key=[new token]
    backend_api_url=[ckan_api_url]


## Docker-compose

The Docker Compose configuration is ready to orchestrate the entire service stack. It includes containers for Publish and Discovery APIs, ckan backend, datapusher, PostgreSQL, Solr, and Redis.

    docker-compose restart
    
    Visit http://localhost:8000/redoc to explore the Discovery Plugin and http://localhost:8001/redoc to explore the Publish Plugin!

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


## License

This project is licensed under the MIT License.
