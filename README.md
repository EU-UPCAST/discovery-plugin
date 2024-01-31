# Discovery Plugin

Welcome to the Discovery Plugin

This service is powered by FastAPI and relies on a robust backend, datapusher, PostgreSQL database, Solr search, Redis caching, and more.

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
   git clone https://github.com/EU-UPCAST/discovery-plugin.git
   
2. Navigate to the project directory:

    ```bash
    cd discovery-plugin

3. Customize your .env file with necessary configurations. Sample .env:

    ```bash
    TZ=UTC
    Backend_PORT=5000
    DATASTORE_READONLY_PASSWORD=myreadonlypassword
    POSTGRES_PASSWORD=mypostgrespassword

## Usage
To run the Discovery Plugin service, execute the following command:


    docker-compose up -d

This will start the service, backend, datapusher, PostgreSQL, Solr, and Redis containers.

Visit http://localhost:8000 to explore the Discovery Plugin!

## Configuration

Customize the service further by updating the environment variables in the .env file. Modify the configurations as needed for your specific use case.

# Example .env file

 
    TZ=UTC
    Backend_PORT=5000
    DATASTORE_READONLY_PASSWORD=myreadonlypassword
    POSTGRES_PASSWORD=mypostgrespassword

## Docker-compose

The Docker Compose configuration is ready to orchestrate the entire service stack. It includes containers for FastAPI, backend, datapusher, PostgreSQL, Solr, and Redis.


    docker-compose up -d

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


## License

This project is licensed under the MIT License.
