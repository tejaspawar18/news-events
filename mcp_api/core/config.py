import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()
# Load appropriate .env file based on ENV variable
APP_ENV = os.getenv("APP_ENV", "production")
env_file = f".env.{APP_ENV}"
if os.path.exists(env_file):
    load_dotenv(env_file)
    logger.info(f"Loaded environment variables from {env_file}")
else:
    logger.warning(f"{env_file} not found. Using system environment variables only.")

class Config:
    ENV = APP_ENV
    PORT = int(os.getenv("API_PORT", 5000))
    logger.info(f"API will run on port: {PORT}")
    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "mcp_db_staging")
    FIELD_VALIDATORS_COLLECTION = os.getenv("FIELD_VALIDATORS_COLLECTION", "field_validators")
    MONGO_DOCUMENTS_COLLECTION = os.getenv("MONGO_DOCUMENTS_COLLECTION", "documents")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mcp_docs_staging")
    
    # Scylla
    SCYLLA_HOST = os.getenv("SCYLLA_HOST")
    SCYLLA_USER = os.getenv("SCYLLA_USER")
    SCYLLA_PASSWORD = os.getenv("SCYLLA_PASSWORD")
    SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE")

    # Monitoring
    MONITORING = os.getenv("API_MONITORING", "mcp_api_staging")
    LOG_DIR = os.getenv("LOG_DIR")
    
    @staticmethod
    def get_scylla_credentials():
        return {
            "host": Config.SCYLLA_HOST,
            "user": Config.SCYLLA_USER,
            "password": Config.SCYLLA_PASSWORD,
            "keyspace": Config.SCYLLA_KEYSPACE
        }
