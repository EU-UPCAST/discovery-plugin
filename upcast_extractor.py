from typing import Dict, List, Any, Optional
from rdflib import Graph
import json
import logging

logger = logging.getLogger(__name__)


class UpcastCkanMetadataExtractor:
    """
    Simplified extractor for converting UPCAST JSON-LD data to CKAN metadata fields.

    This class focuses specifically on extracting relevant metadata from UPCAST
    JSON-LD data and formatting it for insertion into CKAN as extras fields.
    """

    def __init__(self, graph: Optional[Graph] = None):
        """Initialize the extractor with an optional RDF graph."""
        self.graph = graph

    def set_graph(self, graph: Graph) -> None:
        """Set the RDF graph to use for extraction."""
        self.graph = graph

    def extract_ckan_extras(self, json_ld_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract CKAN extras fields from UPCAST JSON-LD data.

        Args:
            json_ld_data: The JSON-LD data as a dictionary

        Returns:
            A list of dictionaries with 'key' and 'value' for CKAN extras
        """
        # Create a new graph and parse the JSON-LD data
        g = Graph()
        try:
            g.parse(data=json.dumps(json_ld_data), format='json-ld')
            self.set_graph(g)
        except Exception as e:
            logger.error(f"Failed to parse JSON-LD data: {str(e)}")
            raise ValueError(f"Invalid JSON-LD data: {str(e)}")

        # Extract all relevant metadata
        extras = []

        # Define CKAN reserved field names to avoid conflicts
        ckan_reserved_fields = {
            "title", "name", "author", "author_email", "maintainer", "maintainer_email",
            "license_id", "notes", "url", "version", "state", "type", "owner_org",
            "private", "tags", "resources"
        }

        # Extract dataset information
        dataset_info = self._extract_dataset_info()
        if dataset_info:
            for key, value in dataset_info.items():
                if value is not None:
                    if key == "id":
                        safe_key = "upcast_dataset_uri"
                    elif key in ckan_reserved_fields:
                        safe_key = f"upcast_{key}"
                    else:
                        safe_key = f"upcast_{key}"
                    extras.append({"key": safe_key, "value": value})

        # Extract distribution information
        distribution_info = self._extract_distribution_info()
        if distribution_info:
            for key, value in distribution_info.items():
                if value is not None:
                    if key == "id":
                        safe_key = "upcast_distribution_uri"
                    else:
                        safe_key = f"upcast_distribution_{key}"
                    extras.append({"key": safe_key, "value": value})

        # Extract themes
        if dataset_info and dataset_info.get("id"):
            themes = self._extract_themes(dataset_info["id"])
            if themes:
                extras.append({"key": "upcast_themes", "value": ",".join(themes)})

        # Extract policy information
        if dataset_info and dataset_info.get("id"):
            policy_uri = self._get_policy_uri(dataset_info["id"])
            if policy_uri:
                extras.append({"key": "upcast_policy_uri", "value": policy_uri})

                # Add simplified policy details
                policy_details = self._get_simplified_policy(policy_uri)
                if policy_details.get("allowed"):
                    extras.append({"key": "upcast_policy_allowed", "value": ",".join(policy_details["allowed"])})
                if policy_details.get("forbidden"):
                    extras.append({"key": "upcast_policy_forbidden", "value": ",".join(policy_details["forbidden"])})
                if policy_details.get("required"):
                    extras.append({"key": "upcast_policy_required", "value": ",".join(policy_details["required"])})
                if policy_details.get("summary"):
                    extras.append({"key": "upcast_policy_summary", "value": policy_details["summary"]})

        return extras

    def _extract_dataset_info(self) -> Dict[str, Any]:
        """Extract core dataset information from the RDF graph."""
        dataset_info = {}

        query = """
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX upcast: <https://www.upcast-project.eu/upcast-vocab/1.0/>

            SELECT ?dataset ?title ?description ?spatial ?publisher ?creator ?price ?priceUnit ?contactPoint
            WHERE {
                ?dataset a dcat:Dataset .
                OPTIONAL { ?dataset dct:title ?title }
                OPTIONAL { ?dataset dct:description ?description }
                OPTIONAL { ?dataset dct:spatial ?spatial }
                OPTIONAL { ?dataset dct:publisher ?publisher }
                OPTIONAL { ?dataset dct:creator ?creator }
                OPTIONAL { ?dataset upcast:price ?price }
                OPTIONAL { ?dataset upcast:priceUnit ?priceUnit }
                OPTIONAL { ?dataset dcat:contactPoint ?contactPoint }
            }
            LIMIT 1
        """

        results = self.graph.query(query)
        for row in results:
            dataset_info = {
                "id": str(row.dataset) if row.dataset else None,
                "title": str(row.title) if row.title else None,
                "description": str(row.description) if row.description else None,
                "spatial": str(row.spatial) if row.spatial else None,
                "publisher": str(row.publisher) if row.publisher else None,
                "contact_point": str(row.contactPoint) if row.contactPoint else None,
                "creator": str(row.creator) if row.creator else None,
                "price": float(row.price) if row.price else 0.0,
                "price_unit": str(row.priceUnit) if row.priceUnit else "EUR"
            }

        return dataset_info

    def _extract_distribution_info(self) -> Optional[Dict[str, Any]]:
        """Extract distribution information from the RDF graph."""
        distribution_info = None

        query = """
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            PREFIX dct: <http://purl.org/dc/terms/>

            SELECT ?dist ?title ?description ?format ?byteSize ?mediaType ?downloadURL
            WHERE {
                ?dist a dcat:Distribution .
                OPTIONAL { ?dist dct:title ?title }
                OPTIONAL { ?dist dct:description ?description }
                OPTIONAL { ?dist dct:format ?format }
                OPTIONAL { ?dist dcat:byteSize ?byteSize }
                OPTIONAL { ?dist dcat:mediaType ?mediaType }
                OPTIONAL { ?dist dcat:downloadURL ?downloadURL }
            }
            LIMIT 1
        """

        results = self.graph.query(query)
        for row in results:
            distribution_info = {
                "id": str(row.dist) if row.dist else None,
                "title": str(row.title) if row.title else None,
                "description": str(row.description) if row.description else None,
                "format": str(row.format) if row.format else None,
                "byte_size": int(row.byteSize) if row.byteSize else None,
                "media_type": str(row.mediaType) if row.mediaType else None,
                "download_url": str(row.downloadURL) if row.downloadURL else None
            }

        return distribution_info

    def _extract_themes(self, dataset_uri: str) -> List[str]:
        """Extract themes for a dataset."""
        themes = []

        query = f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>

            SELECT ?theme
            WHERE {{
                <{dataset_uri}> dcat:theme ?theme .
            }}
        """

        results = self.graph.query(query)
        for row in results:
            theme_uri = str(row.theme)
            theme_name = theme_uri.split('/')[-1]
            if ':' in theme_name:
                theme_name = theme_name.split(':')[-1]
            themes.append(theme_name)

        return themes

    def _get_policy_uri(self, dataset_uri: str) -> Optional[str]:
        """Get the policy URI associated with a dataset."""
        query = f"""
            PREFIX odrl: <http://www.w3.org/ns/odrl/>

            SELECT ?policy
            WHERE {{
                <{dataset_uri}> odrl:hasPolicy ?policy .
            }}
            LIMIT 1
        """

        results = self.graph.query(query)
        for row in results:
            return str(row.policy) if row.policy else None

        return None

    def _extract_permissions(self, policy_uri: str) -> List[Dict[str, Any]]:
        """Extract permissions from an ODRL policy."""
        permissions = []

        query = f"""
            PREFIX odrl: <http://www.w3.org/ns/odrl/>

            SELECT ?permission ?action
            WHERE {{
                <{policy_uri}> odrl:permission ?permission .
                OPTIONAL {{ ?permission odrl:action ?action }}
            }}
        """

        results = self.graph.query(query)
        for row in results:
            action = None
            if row.action:
                action = str(row.action)
                if "/odrl/" in action or "odrl:" in action:
                    action = action.split('/')[-1]
                    if ":" in action:
                        action = action.split(':')[-1]

            if action:
                permissions.append({"action": action})

        return permissions

    def _extract_prohibitions(self, policy_uri: str) -> List[Dict[str, Any]]:
        """Extract prohibitions from an ODRL policy."""
        prohibitions = []

        query = f"""
            PREFIX odrl: <http://www.w3.org/ns/odrl/>

            SELECT ?prohibition ?action
            WHERE {{
                <{policy_uri}> odrl:prohibition ?prohibition .
                OPTIONAL {{ ?prohibition odrl:action ?action }}
            }}
        """

        results = self.graph.query(query)
        for row in results:
            action = None
            if row.action:
                action = str(row.action)
                if "/odrl/" in action or "odrl:" in action:
                    action = action.split('/')[-1]
                    if ":" in action:
                        action = action.split(':')[-1]

            if action:
                prohibitions.append({"action": action})

        return prohibitions

    def _extract_obligations(self, policy_uri: str) -> List[Dict[str, Any]]:
        """Extract obligations from an ODRL policy."""
        obligations = []

        query = f"""
            PREFIX odrl: <http://www.w3.org/ns/odrl/>

            SELECT ?obligation ?action
            WHERE {{
                <{policy_uri}> odrl:obligation ?obligation .
                OPTIONAL {{ ?obligation odrl:action ?action }}
            }}
        """

        results = self.graph.query(query)
        for row in results:
            action = None
            if row.action:
                action = str(row.action)
                if "/odrl/" in action or "odrl:" in action:
                    action = action.split('/')[-1]
                    if ":" in action:
                        action = action.split(':')[-1]

            if action:
                obligations.append({"action": action})

        return obligations

    def _get_simplified_policy(self, policy_uri: str) -> Dict[str, Any]:
        """Get a simplified representation of a policy."""
        permissions = self._extract_permissions(policy_uri)
        prohibitions = self._extract_prohibitions(policy_uri)
        obligations = self._extract_obligations(policy_uri)

        permission_actions = [p.get("action") for p in permissions if p.get("action")]
        prohibition_actions = [p.get("action") for p in prohibitions if p.get("action")]
        obligation_actions = [p.get("action") for p in obligations if p.get("action")]

        return {
            "allowed": permission_actions,
            "forbidden": prohibition_actions,
            "required": obligation_actions,
            "summary": self._generate_policy_summary(permission_actions, prohibition_actions, obligation_actions)
        }

    def _generate_policy_summary(self, permissions: List[str], prohibitions: List[str], obligations: List[str]) -> str:
        """Generate a human-readable summary of a policy."""
        parts = []

        if permissions:
            parts.append(f"You may {', '.join(permissions)}")

        if prohibitions:
            parts.append(f"You may not {', '.join(prohibitions)}")

        if obligations:
            parts.append(f"You must {', '.join(obligations)}")

        if not parts:
            return "No specific actions defined in this policy."

        return ". ".join(parts) + "."