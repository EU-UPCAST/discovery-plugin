# UPCAST Discovery & Publish Plugin

Welcome to the UPCAST Discovery & Publish Plugin - a comprehensive data management system that enables dataset discovery, publishing, and marketplace integration through a robust FastAPI-based architecture.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Services](#services)
  - [Discovery API](#discovery-api)
  - [Publish API](#publish-api)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Marketplace Integration](#marketplace-integration)
- [Docker Compose](#docker-compose)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Discovery API**: Advanced dataset search and similarity detection using FAISS embeddings
- **Publish API**: Dataset publishing with custom metadata and marketplace integration
- **CKAN Integration**: Full integration with CKAN data management platform
- **Marketplace Support**: Direct publishing to Nokia and OKFN marketplaces
- **Semantic Search**: AI-powered dataset discovery and similarity matching
- **Negotiation System**: Automated offer creation and policy management
- **Kafka Integration**: Real-time messaging for distributed operations
- **Security**: Token-based authentication and authorization
- **Docker Support**: Complete containerized deployment

## Architecture

The system consists of several interconnected services:

- **Discovery API** (`discoveryapi`): Dataset search and similarity detection
- **Publish API** (`publishapi`): Dataset publishing and marketplace integration
- **CKAN Backend** (`ckan`): Core data management system
- **PostgreSQL** (`db`): Primary database
- **Solr** (`solr`): Search indexing and querying
- **Redis** (`redis`): Caching and session management
- **Datapusher** (`datapusher`): Data processing and transformation

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- Git with submodule support

### Installation

1. **Clone the repository with submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/EU-UPCAST/discovery-plugin.git
   cd discovery-plugin
   ```

2. **Ensure submodules are properly initialized**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Create your environment configuration**:
   ```bash
   cp example.env .env
   ```

4. **Configure your environment variables** in `.env`:
   ```bash
   TZ=UTC
   CKAN_PORT=5001
   DATASTORE_READONLY_PASSWORD=your_readonly_password
   POSTGRES_PASSWORD=your_postgres_password
   # Add other required configurations
   ```

### Configuration

#### Essential Configuration Files

1. **Environment Variables** (`.env`):
   - Database passwords and connection settings
   - CKAN port and configuration
   - Timezone settings

2. **API Configuration** (`config.py`):
   ```python
   backend_api_key = "your_ckan_api_token"
   backend_api_url = "http://ckan:5000/api/3/action/"
   service_api_key = "your_service_api_key"
   negotiation_url = "your_negotiation_service_url"
   MASTER_PASSWORD = "your_master_password"
   ```

## Services

### Discovery API

The Discovery API provides advanced dataset search and similarity detection capabilities.

**Key Features:**
- Dataset and resource search using Solr
- AI-powered similarity detection using FAISS embeddings
- Data processing workflow discovery
- Semantic search with description matching

**Main Endpoints:**
- `GET /discover/dataset_search` - Search datasets with advanced filtering
- `GET /discover/resource_search` - Search resources within datasets
- `POST /discover/discover_similar_datasets` - Find similar datasets
- `GET /discover/create_embeddings` - Update embedding metadata
- `GET /ui/search_ui` - Web interface for search
- `GET /ui/discover_ui` - Web interface for discovery

### Publish API

The Publish API enables dataset publishing with custom metadata and marketplace integration.

**Key Features:**
- Dataset creation with custom fields
- UPCAST metadata extraction and processing
- Multi-marketplace publishing (Nokia, OKFN)
- Automated negotiation offer creation
- File upload and resource management

**Main Endpoints:**
- `POST /catalog/create_dataset/` - Create basic dataset
- `POST /catalog/create_dataset_with_custom_fields/` - Create dataset with UPCAST metadata
- `POST /catalog/update_dataset/` - Update existing dataset
- `POST /catalog/delete_dataset/` - Delete dataset
- `POST /catalog/upload_data/` - Upload data files

## Usage

### Starting the System

1. **Start all services**:
   ```bash
   docker compose up -d
   ```

2. **Verify services are running**:
   ```bash
   docker compose ps
   ```

3. **Access the services**:
   - CKAN Backend: http://localhost:5001
   - Discovery API: http://localhost:8003
   - Publish API: http://localhost:8004
   - Datapusher: http://localhost:8800

### Post-Installation Setup

1. **Create CKAN API Token**:
   - Visit http://localhost:5001
   - Create an admin user account
   - Generate an API token at `/user/[username]/api-tokens`

2. **Update Configuration**:
   - Edit `config.py` with your CKAN API token
   - Set the correct CKAN URL if different from default
   - Configure marketplace credentials if needed

3. **Initialize Embeddings** (for Discovery API):
   ```bash
   curl -X GET "http://localhost:8003/discover/create_embeddings" \
        -H "apitoken: your_service_api_key"
   ```

### Example Usage

#### Search for Datasets
```bash
curl -X GET "http://localhost:8003/discover/dataset_search?q=climate&rows=10" \
     -H "apitoken: your_service_api_key"
```

#### Create a Dataset
```bash
curl -X POST "http://localhost:8004/catalog/create_dataset/" \
     -H "apitoken: your_service_api_key" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "package_name=my-dataset&package_title=My Dataset&organization_name=my-org&package_notes=Dataset description"
```

#### Find Similar Datasets
```bash
curl -X POST "http://localhost:8003/discover/discover_similar_datasets?dataset_id=my-dataset-id" \
     -H "apitoken: your_service_api_key"
```

## API Documentation

Once the services are running, you can access the interactive API documentation:

- **Discovery API**: http://localhost:8003/docs
- **Publish API**: http://localhost:8004/docs

The APIs also provide ReDoc documentation:
- **Discovery API**: http://localhost:8003/redoc
- **Publish API**: http://localhost:8004/redoc

## Marketplace Integration

The system supports integration with multiple data marketplaces:

### Supported Marketplaces

1. **Nokia Marketplace**
   - Requires authentication token
   - Supports pricing and location metadata
   - Automatic stream creation

2. **OKFN Marketplace**
   - API key-based authentication
   - UPCAST metadata publishing
   - Resource-based publishing

### Configuration

Configure marketplace credentials using the credentials management endpoints:
- `POST /marketplace-credentials/` - Add marketplace credentials
- `GET /marketplace-credentials/` - List configured marketplaces
- `DELETE /marketplace-credentials/{marketplace_id}` - Remove credentials

## Docker Compose

The system uses Docker Compose for orchestration. Key services:

```yaml
services:
  discoveryapi:    # Discovery API (port 8003)
  publishapi:      # Publish API (port 8004)
  ckan:           # CKAN backend (port 5001)
  db:             # PostgreSQL database
  solr:           # Solr search engine
  redis:          # Redis cache
  datapusher:     # Data processing (port 8800)
```

### Useful Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f [service_name]

# Restart specific service
docker compose restart [service_name]

# Stop all services
docker compose down

# Rebuild and restart
docker compose up -d --build
```

## Security

### Authentication

Both APIs use token-based authentication:
- Set `service_api_key` in `config.py`
- Include `apitoken` header in all requests
- Tokens are validated on each request

### Best Practices

- Use strong, unique API tokens
- Regularly rotate API keys
- Implement HTTPS in production
- Configure firewall rules appropriately
- Use environment variables for sensitive data

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Clone the repository with submodules
2. Set up your development environment
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue in the GitHub repository
- Check the API documentation for detailed endpoint information
- Review the configuration files for setup guidance

---

**Note**: This is part of the EU UPCAST project for federated data management and marketplace integration.