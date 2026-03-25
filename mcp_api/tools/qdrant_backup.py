import requests
import os
import boto3
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant.event_agents.mcp_api.core.connection import COLLECTION_NAME

# Local temp dir
BACKUP_DIR = "/home/ubuntu/rj/qdrant_backup"
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.31.11.78:6333") 


load_dotenv()

BUFFER_LIMIT = 10
S3_BUCKET = os.getenv("S3_BUCKET_EVENT")
s3_session = boto3.Session(
    region_name="ap-south-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

S3_CLIENT = s3_session.client('s3')
S3_KEY = f"qdrant_backups/{COLLECTION_NAME}.snapshot"  


def create_snapshot(collection: str):
    """Create snapshot in Qdrant and download locally."""
    # Ask Qdrant to create snapshot
    response = requests.post(f"{QDRANT_URL}/collections/{collection}/snapshots")
    response.raise_for_status()
    snapshot = response.json()["result"]
    snapshot_name = snapshot["name"]

    print(f"✅ Snapshot created: {snapshot_name}")

    # Download snapshot
    snapshot_url = f"{QDRANT_URL}/collections/{collection}/snapshots/{snapshot_name}"
    backup_path = os.path.join(BACKUP_DIR, f"{collection}.snapshot")

    os.makedirs(BACKUP_DIR, exist_ok=True)
    with requests.get(snapshot_url, stream=True) as r:
        r.raise_for_status()
        with open(backup_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

    print(f"📦 Snapshot saved locally at {backup_path}")
    return backup_path

def upload_to_s3(local_path: str, s3_key: str):
    """Upload snapshot to S3 (overwrite previous)."""
    S3_CLIENT.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"☁️ Snapshot uploaded to s3://{S3_BUCKET}/{s3_key}")

if __name__ == "__main__":
    backup_path = create_snapshot(COLLECTION_NAME)

    # Upload and overwrite
    upload_to_s3(backup_path, S3_KEY)

    # Clean up local temp file
    # os.remove(backup_path)
    print(f"🧹 Cleaned up local file {backup_path}")
