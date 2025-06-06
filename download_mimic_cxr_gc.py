#!./.venv/bin/python3

import os
from pathlib import Path

from google.cloud import storage


def download_data(bucket_name: str, project: str, data_dir: Path):
    """download data from gc bucket"""

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket)
    for blob in blobs:
        print(blob.name)
        return
        # Create local path
        local_path = data_dir / blob.name
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)


if __name__ == "__main__":
    BUCKET_NAME = "mimic-cxr-jpg-2.1.0.physionet.org"
    PROJECT = os.getenv("GC_PROJECT")
    if not PROJECT:
        print("specify GC project to use for client by setting GC_PROJECT env variable")
        os._exit(1)
    # KEY_PATH = os.getenv("GC_SERVICE_KEY_FILE")
    # if not KEY_PATH:
    #    print(
    #        "Set the GC_SERVICE_KEY_FILE env variable to the path "
    #        "of service account json credential file"
    #    )
    #    os._exit(1)
    # if not Path(KEY_PATH).exists():
    #    print(f"{KEY_PATH} Key file not found")
    #    os._exit(1)

    DATA_DIR = Path("data/mimic-cxr-jpg-2.1.0")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_data(BUCKET_NAME, PROJECT, DATA_DIR)
