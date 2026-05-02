"""Cloud upload helpers for Azure Blob Storage and AWS S3.

Usage (CLI):
    python main.py generate --config ... --output out/ \
        --upload-to azure://my-container/prefix/path
    python main.py generate --config ... --output out/ \
        --upload-to s3://my-bucket/prefix/path

URI format:
    azure://<container>[/<prefix>]
    s3://<bucket>[/<prefix>]

Credentials (read from environment — never hardcoded):
    Azure:
        AZURE_STORAGE_CONNECTION_STRING   (preferred)
        AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY
        AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_SAS_TOKEN
    AWS:
        AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY [+ AWS_DEFAULT_REGION]
        AWS_PROFILE                               (named profile)
        IAM role / instance-profile               (automatic on EC2 / ECS)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URI parser
# ---------------------------------------------------------------------------
def parse_upload_uri(uri: str):
    """
    Returns (scheme, bucket_or_container, prefix).
    Scheme is 'azure' or 's3'.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if scheme not in ("azure", "s3"):
        raise ValueError(f"Unsupported upload URI scheme '{scheme}'. Use azure:// or s3://")
    container_or_bucket = parsed.netloc or parsed.path.lstrip("/").split("/")[0]
    # prefix is everything after the bucket/container
    raw_path = parsed.path.lstrip("/")
    # for s3://<bucket>/prefix the bucket is in netloc
    if parsed.netloc:
        prefix = raw_path  # path after netloc = prefix
    else:
        parts = raw_path.split("/", 1)
        container_or_bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    prefix = prefix.rstrip("/")
    return scheme, container_or_bucket, prefix


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
class AzureUploader:
    def __init__(self, container: str, prefix: str = ""):
        self.container = container
        self.prefix = prefix.rstrip("/")
        self._client = self._build_client()

    def _build_client(self):
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure upload. "
                "Install it: pip install azure-storage-blob"
            )
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            return BlobServiceClient.from_connection_string(conn_str)
        account = os.getenv("AZURE_STORAGE_ACCOUNT")
        if not account:
            raise EnvironmentError(
                "Azure credentials not found. Set AZURE_STORAGE_CONNECTION_STRING "
                "or AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY / AZURE_STORAGE_SAS_TOKEN"
            )
        key = os.getenv("AZURE_STORAGE_KEY")
        sas = os.getenv("AZURE_STORAGE_SAS_TOKEN")
        if key:
            return BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net",
                credential=key,
            )
        if sas:
            return BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net?{sas.lstrip('?')}"
            )
        raise EnvironmentError(
            "Set AZURE_STORAGE_KEY or AZURE_STORAGE_SAS_TOKEN alongside AZURE_STORAGE_ACCOUNT"
        )

    def upload_directory(self, local_dir: str) -> List[str]:
        local_path = Path(local_dir)
        files = list(local_path.rglob("*"))
        uploaded: List[str] = []
        container_client = self._client.get_container_client(self.container)
        for f in files:
            if not f.is_file():
                continue
            relative = f.relative_to(local_path).as_posix()
            blob_name = f"{self.prefix}/{relative}" if self.prefix else relative
            with open(f, "rb") as fh:
                container_client.upload_blob(name=blob_name, data=fh, overwrite=True)
            uploaded.append(f"azure://{self.container}/{blob_name}")
            logger.info(f"  Uploaded → azure://{self.container}/{blob_name}")
        return uploaded


# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------
class S3Uploader:
    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = self._build_client()

    def _build_client(self):
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 upload. Install it: pip install boto3"
            )
        return boto3.client("s3", region_name=self.region)

    def upload_directory(self, local_dir: str) -> List[str]:
        local_path = Path(local_dir)
        files = list(local_path.rglob("*"))
        uploaded: List[str] = []
        for f in files:
            if not f.is_file():
                continue
            relative = f.relative_to(local_path).as_posix()
            key = f"{self.prefix}/{relative}" if self.prefix else relative
            self._client.upload_file(str(f), self.bucket, key)
            uploaded.append(f"s3://{self.bucket}/{key}")
            logger.info(f"  Uploaded → s3://{self.bucket}/{key}")
        return uploaded


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def upload_output(local_dir: str, uri: str) -> List[str]:
    """
    Upload all files under local_dir to the destination specified by uri.

    uri examples:
        azure://my-container/snapshots/2026-05-01
        s3://my-data-bucket/synthetic/run_01
    """
    scheme, target, prefix = parse_upload_uri(uri)
    logger.info(f"Uploading {local_dir!r} → {uri} ...")
    if scheme == "azure":
        uploader = AzureUploader(container=target, prefix=prefix)
    else:
        uploader = S3Uploader(bucket=target, prefix=prefix)
    paths = uploader.upload_directory(local_dir)
    logger.info(f"Upload complete: {len(paths)} file(s) uploaded to {uri}")
    return paths
