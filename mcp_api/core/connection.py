
from pymongo import MongoClient
from qdrant_client import QdrantClient
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import ConsistencyLevel
from mcp_api.core.config import Config

# MongoDB
mongo_client = MongoClient(Config.MONGO_URI)
db = mongo_client[Config.DB_NAME]
collection = db[Config.MONGO_DOCUMENTS_COLLECTION]

# Qdrant
qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)


# ScyllaDB
class ScyllaDB:
    def __init__(self, credentials):
        auth_provider = None
        if credentials['user'] and credentials['password']:
            auth_provider = PlainTextAuthProvider(
                credentials['user'], credentials['password']
            )

        # Conditional load balancing for production
        if Config.ENV == "production":
            cluster_args = {
                "contact_points": [ip.strip() for ip in credentials["host"].split(',')],
                "auth_provider": auth_provider,
                "load_balancing_policy": DCAwareRoundRobinPolicy(local_dc="31"),
                "protocol_version": 4,
            }
        else:
            # Staging: No custom load balancing
            cluster_args = {
                "contact_points": [ip.strip() for ip in credentials["host"].split(',')],
                "auth_provider": auth_provider,
                "protocol_version": 4,
            }

        self.cluster = Cluster(**cluster_args)

        try:
            self.session = self.cluster.connect()
            
            # Apply consistency level only for production
            if Config.ENV == "production":
                self.session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM

            self.session.set_keyspace(credentials['keyspace'])

        except Exception as e:
            print(f"Error connecting to ScyllaDB: {e}")
            self.session = None
    
    def get_session(self):
        return self.session

    def close(self):
        self.cluster.shutdown()

# Global connection
SCYLLA_OBJ = ScyllaDB(Config.get_scylla_credentials())
