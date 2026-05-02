"""Collibra data-catalog importer.

Fetches dataset / column definitions from the Collibra REST API v2 and
converts them into the FDL YAML config format so they can drive data
generation directly — no manual Excel/YAML authoring needed.

Credentials (set as environment variables, never hardcoded):
    COLLIBRA_BASE_URL    https://your-org.collibra.com
    COLLIBRA_USERNAME    service account / personal user
    COLLIBRA_PASSWORD    password (or API token if Basic Auth token supported)

Usage (CLI):
    python main.py collibra-import \\
        --dataset "Account Booking" \\
        --output config/from_collibra.yaml

    python main.py collibra-import \\
        --domain "Finance" \\
        --asset-type "Data Set" \\
        --output config/finance_datasets.yaml

    python main.py generate \\
        --config config/from_collibra.yaml \\
        --output output/snapshot_v1
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collibra → FDL data-type mapping
# (Extend this map to match your organisation's custom Collibra type vocabulary)
# ---------------------------------------------------------------------------
_COLLIBRA_TYPE_MAP: Dict[str, str] = {
    # Collibra physical/logical type → FDL data_type
    "varchar":          "VA256",
    "varchar2":         "VA256",
    "nvarchar":         "VA256",
    "char":             "A",
    "nchar":            "A",
    "text":             "VA256",
    "clob":             "VA256",
    "integer":          "N19",
    "int":              "N19",
    "bigint":           "N38",
    "smallint":         "N6",
    "number":           "N19",
    "numeric":          "DC",
    "decimal":          "DC",
    "float":            "DC",
    "double":           "DC",
    "real":             "DC",
    "date":             "D",
    "datetime":         "DT",
    "timestamp":        "TS",
    "timestamp with time zone": "TS",
    "boolean":          "A1",
    "bit":              "A1",
}


def _map_data_type(collibra_type: str) -> str:
    return _COLLIBRA_TYPE_MAP.get((collibra_type or "").lower().strip(), "VA256")


# ---------------------------------------------------------------------------
# HTTP helper (uses only stdlib requests — already in deps)
# ---------------------------------------------------------------------------
class _CollibraSession:
    def __init__(self, base_url: str, username: str, password: str):
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required. Install: pip install requests")
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self._session = requests.Session()
        self._session.auth = self.auth
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}/rest/2.0/{path.lstrip('/')}"
        resp = self._session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def find_assets(self, name: str, asset_type_name: Optional[str] = None,
                    domain_name: Optional[str] = None) -> List[Dict]:
        params: Dict[str, Any] = {"name": name, "nameMatchMode": "ANYWHERE", "limit": 50}
        if asset_type_name:
            # Resolve type UUID first
            type_resp = self.get("assetTypes", params={"name": asset_type_name, "limit": 5})
            type_results = type_resp.get("results", [])
            if type_results:
                params["typeId"] = type_results[0]["id"]
        if domain_name:
            domain_resp = self.get("domains", params={"name": domain_name, "limit": 5})
            domain_results = domain_resp.get("results", [])
            if domain_results:
                params["domainId"] = domain_results[0]["id"]
        resp = self.get("assets", params=params)
        return resp.get("results", [])

    def get_asset_attributes(self, asset_id: str) -> List[Dict]:
        resp = self.get("attributes", params={"assetId": asset_id, "limit": 200})
        return resp.get("results", [])

    def get_relations(self, asset_id: str, relation_type: str = "00000000-0000-0000-0000-000000007042") -> List[Dict]:
        """Fetch child assets related to asset_id (e.g. columns of a table)."""
        resp = self.get("relations", params={
            "targetId": asset_id,
            "relationTypeId": relation_type,
            "limit": 500,
        })
        return resp.get("results", [])


# ---------------------------------------------------------------------------
# Attribute extraction helpers
# ---------------------------------------------------------------------------
def _attr_value(attrs: List[Dict], label: str) -> Optional[str]:
    """Return the first string value of an attribute with the given label."""
    label_lower = label.lower()
    for attr in attrs:
        attr_label = (attr.get("type", {}).get("name") or "").lower()
        if attr_label == label_lower:
            val = attr.get("value", {})
            if isinstance(val, dict):
                return val.get("value") or val.get("stringValue")
            return str(val) if val is not None else None
    return None


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------
class CollibraImporter:
    """
    Imports a dataset definition from Collibra and produces a YAML config
    compatible with the FDL Synthetic Data Platform.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("COLLIBRA_BASE_URL", "")
        self.username = username or os.getenv("COLLIBRA_USERNAME", "")
        self.password = password or os.getenv("COLLIBRA_PASSWORD", "")
        if not self.base_url:
            raise EnvironmentError(
                "COLLIBRA_BASE_URL is not set. "
                "Export it before running: set COLLIBRA_BASE_URL=https://your-org.collibra.com"
            )
        if not self.username or not self.password:
            raise EnvironmentError(
                "COLLIBRA_USERNAME and COLLIBRA_PASSWORD must be set."
            )
        self._session = _CollibraSession(self.base_url, self.username, self.password)

    # ------------------------------------------------------------------
    # Fetch and convert a single dataset asset
    # ------------------------------------------------------------------
    def fetch_dataset(self, asset_id: str) -> Dict:
        """Fetch asset + attributes + child columns for a Collibra dataset asset."""
        asset = self._session.get(f"assets/{asset_id}")
        attrs = self._session.get_asset_attributes(asset_id)
        # Try standard Collibra "Table" → "Column" relation (type UUID varies by instance)
        columns_raw = self._fetch_columns(asset_id)
        return {"asset": asset, "attrs": attrs, "columns": columns_raw}

    def _fetch_columns(self, dataset_id: str) -> List[Dict]:
        """
        Columns in Collibra are child assets related to the parent table/dataset.
        Standard relation type IDs differ between Collibra instances — we try
        the most common ones and fall back to a name-based search.
        """
        # Common Collibra "contains column" relation type UUIDs
        relation_type_ids = [
            "00000000-0000-0000-0000-000000007042",  # Data Element is part of Data Set
            "00000000-0000-0000-0000-000000007062",  # Column is part of Table
        ]
        for rel_id in relation_type_ids:
            rels = self._session.get_relations(dataset_id, rel_id)
            if rels:
                # Each relation has a 'source' (child column) and 'target' (parent table)
                col_ids = [r["source"]["id"] for r in rels if "source" in r]
                columns = []
                for col_id in col_ids:
                    try:
                        col_asset = self._session.get(f"assets/{col_id}")
                        col_attrs = self._session.get_asset_attributes(col_id)
                        columns.append({"asset": col_asset, "attrs": col_attrs})
                    except Exception as exc:
                        logger.warning(f"Could not fetch column {col_id}: {exc}")
                if columns:
                    return columns
        logger.warning(f"No columns found for dataset {dataset_id} via standard relation types. "
                       "Check relation type UUIDs for your Collibra instance.")
        return []

    # ------------------------------------------------------------------
    # Convert Collibra response → FDL YAML dict
    # ------------------------------------------------------------------
    def to_fdl_config(self, fetched: Dict, table_name: Optional[str] = None) -> Dict:
        asset = fetched["asset"]
        table_name = table_name or asset.get("displayName") or asset.get("name") or "table"
        # Sanitize for use as table identifier
        table_id = table_name.lower().replace(" ", "_").replace("-", "_")

        columns = []
        for col_data in fetched.get("columns", []):
            col_asset = col_data["asset"]
            col_attrs = col_data["attrs"]
            col_name = (col_asset.get("displayName") or col_asset.get("name") or "col").upper()
            raw_type = _attr_value(col_attrs, "Data Type") or _attr_value(col_attrs, "Physical Data Type") or "varchar"
            data_type = _map_data_type(raw_type)
            description = _attr_value(col_attrs, "Description") or _attr_value(col_attrs, "Technical Description")
            nullable_raw = _attr_value(col_attrs, "Nullable") or _attr_value(col_attrs, "Is Nullable")
            nullable = str(nullable_raw).lower() not in ("false", "no", "0") if nullable_raw else True
            is_pk_raw = _attr_value(col_attrs, "Is Primary Key") or _attr_value(col_attrs, "Primary Key")
            is_pk = str(is_pk_raw).lower() in ("true", "yes", "1") if is_pk_raw else False
            col_entry: Dict[str, Any] = {
                "name": col_name,
                "type": data_type,
                "nullable": nullable,
            }
            if is_pk:
                col_entry["pk"] = True
            if description:
                col_entry["description"] = description
            columns.append(col_entry)

        return {
            "config_format": "fdl-yaml-v1",
            "run_settings": {
                "default_records_per_table": 1000,
            },
            "tables": [
                {
                    "name": table_id,
                    "description": (
                        fetched["asset"].get("displayName") or fetched["asset"].get("name") or table_id
                    ),
                    "rows": 1000,
                    "columns": columns,
                }
            ],
        }

    # ------------------------------------------------------------------
    # High-level: find by name and export YAML
    # ------------------------------------------------------------------
    def import_dataset(
        self,
        dataset_name: str,
        output_path: str,
        asset_type: str = "Data Set",
        domain: Optional[str] = None,
    ) -> str:
        """
        Search Collibra for dataset_name, convert to FDL YAML, write to output_path.
        Returns the resolved output path.
        """
        import yaml  # PyYAML is in deps

        logger.info(f"Searching Collibra for dataset: {dataset_name!r} ...")
        assets = self._session.find_assets(dataset_name, asset_type_name=asset_type, domain_name=domain)
        if not assets:
            raise ValueError(
                f"No Collibra asset found matching '{dataset_name}' "
                f"(type={asset_type!r}, domain={domain!r})"
            )
        if len(assets) > 1:
            names = [a.get("displayName") or a.get("name") for a in assets]
            logger.warning(f"Multiple matches found — using first: {names}")
        asset_id = assets[0]["id"]
        logger.info(f"Found asset id={asset_id}. Fetching columns...")
        fetched = self.fetch_dataset(asset_id)
        config = self.to_fdl_config(fetched, table_name=dataset_name)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Collibra config written to {out} ({len(config['tables'][0]['columns'])} columns)")
        return str(out)
