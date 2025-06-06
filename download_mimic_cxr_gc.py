#!./.venv/bin/python3

import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account


def download_data(bucket_name: str, data_dir: Path, key_path: str):
    """download data from gc bucket"""

    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket)
    for blob in blobs:
        print(blob.name)
        # Create local path
        local_path = data_dir / blob.name
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)


if __name__ == "__main__":
    BUCKET_NAME = "mimic-cxr-jpg-2.1.0.physionet.org"
    KEY_PATH = os.getenv("GC_SERVICE_KEY_FILE")
    if not KEY_PATH:
        print(
            "Set the GC_SERVICE_KEY_FILE env variable to the path "
            "of service account json credential file"
        )
        os._exit(1)
    if not Path(KEY_PATH).exists():
        print(f"{KEY_PATH} Key file not found")
        os._exit(1)

    DATA_DIR = Path("mimic-cxr-jpg-2.1.0")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_data(BUCKET_NAME, DATA_DIR, KEY_PATH)
