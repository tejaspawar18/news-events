from mcp_api.core.connection import collection, qdrant_client
from mcp_api.core.config import Config

def delete_all_documents():
    collection.delete_many({})
    qdrant_client.delete_collection(collection_name=Config.COLLECTION_NAME)

def reindex_documents(embed_func):
    documents = list(collection.find({}))
    for doc in documents:
        embed_func(doc)