from pymongo import MongoClient
from mcp_api.core.config import Config
from mcp_api.models.dynamic_model_builder import event_table_fields

def setup_field_validators(uri=Config.MONGO_URI, db_name=Config.DB_NAME):
    client = MongoClient(uri)
    db = client[db_name]
    db.drop_collection(Config.FIELD_VALIDATORS_COLLECTION)
    if Config.FIELD_VALIDATORS_COLLECTION not in db.list_collection_names():
        db.create_collection(Config.FIELD_VALIDATORS_COLLECTION)
        print(f"Created '{Config.FIELD_VALIDATORS_COLLECTION}' collection.")
    else:
        print(f"'{Config.FIELD_VALIDATORS_COLLECTION}' collection already exists.")

    collection = db[Config.FIELD_VALIDATORS_COLLECTION]

    model_definitions = [
        {
            "model_name": "events_qdrant",
            "fields": {
                "scylla_id": "UUID",
                "root_id": "UUID",
                'ac':'str',
                'pc': 'str',
                'district': 'str',
                'state': 'str',
                # "bm25_vector": "Optional[Dict[int, float]]",
                "vectors_array": "list[VectorEntry]"
            }
        },
        {
            "model_name": "events",
            "fields": {
                "scylla_id": "UUID",
                "root_id": "UUID",
                "publish_time": "str",
                "updated_time": "str",
                "state": "str",
                "category": "str",
                "topic": "str"
            }
        },
        {
            "model_name": "events_scylla",
            "fields": event_table_fields
        }
    ]

    for model in model_definitions:
        if not collection.find_one({"model_name": model["model_name"]}):
            collection.insert_one(model)
            print(f"Inserted model: {model['model_name']}")
        else:
            print(f"Model '{model['model_name']}' already exists.")
