from pymongo import MongoClient
from uuid import UUID
from pydantic import BaseModel, create_model, model_validator
from datetime import datetime
from typing import Any, List, Optional, Union, Dict
from mcp_api.core.config import Config
from mcp_api.models.schema import EventGeoLocation, EventParty, EventPartySentiment

# --- Cache for generated models ---
_model_cache: dict[str, type] = {}

# --- Custom embedded submodel ---
class VectorEntry(BaseModel):
    vector: Union[List[float], List[List[float]]]
    root_id: UUID
    qdrant_id: UUID
    vector_name: str

    @model_validator(mode="after")
    def check_at_least_one_key(self) -> "VectorEntry":
        if not self.qdrant_id and self.rank is None:
            raise ValueError("Either 'qdrant_id' or 'rank' must be provided.")
        return self

# --- Supported types map ---
PY_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list[float]": List[float],
    "list[int]": List[int],
    "list[str]": List[str],
    "VectorEntry": VectorEntry,
    "list[VectorEntry]": List[VectorEntry],
    "dict": dict,
    "Any": Any,
    "UUID": UUID,
    "datetime": datetime,
    "Optional[datetime]": Optional[datetime],
    "List[str]": List[str],
    "List[int]": List[int],
    "List[float]": List[float],
    "List[VectorEntry]": List[VectorEntry],
    "Optional[str]": Optional[str],
    "Optional[int]": Optional[int],
    "Optional[float]": Optional[float],
    "Optional[List[str]]": Optional[List[str]],
    "Optional[List[int]]": Optional[List[int]],
    "Optional[List[float]]": Optional[List[float]],
    "Optional[List[VectorEntry]]": Optional[List[VectorEntry]],
    "Optional[dict]": Optional[dict],
    "Optional[Dict[int, float]]": Optional[Dict[int, float]],
    "Optional[Any]": Optional[Any],
    "Optional[VectorEntry]": Optional[VectorEntry],
    "Optional[List[Any]]": Optional[List[Any]],
    "Optional[Union[str, int, float]]": Optional[Union[str, int, float]],
    "Optional[UUID]": Optional[UUID],
    "Optional[List[UUID]]": Optional[List[UUID]],
    "List[EventGeoLocation]": List[EventGeoLocation],
    "List[EventParty]":List[EventParty],
    "EventPartySentiment": EventPartySentiment,
}



event_table_fields = {
    "id": "UUID",
    "root_id": "UUID",
    "title": "str",
    "category": "str",
    "summary": "str",
    "summary_details": "str",
    "summary_hindi": "str",
    "summary_details_hindi": "str",
    "regional_language": "str",
    "summary_regional": "str",
    "summary_details_regional": "str",
    "geography": "str",
    "state": "str",
    "ac": "List[EventGeoLocation]",
    "pc": "List[EventGeoLocation]",
    "district": "List[EventGeoLocation]",
    "parties": "List[EventParty]",
    "candidates": "List[EventGeoLocation]",
    "tags": "str",
    "political_inclination_party": "EventPartySentiment",
    "sources": "List[str]",
    "s3_article_url": "List[str]",
    "source_domain": "List[str]",
    "parent_id": "Optional[UUID]",
    "parent_semantic_score": "float",
    "top_images": "List[str]",
    "media_links": "List[str]",
    "opinion": "str",
    "research": "str",
    "article_count": "int",
    "ancestor_count": "int",
    "published_time": "datetime",
    "updated_at": "datetime",
    "is_leaf": "bool",
}

# --- Type resolver ---
def resolve_type(type_str: str) -> Any:
    if type_str in PY_TYPES:
        return PY_TYPES[type_str]
    raise ValueError(f"Unsupported type: {type_str}")

# --- Model generator from Mongo ---
def get_model(model_name: str, uri=Config.MONGO_URI, db_name=Config.DB_NAME) -> type:
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    client = MongoClient(uri)
    collection = client[db_name][Config.FIELD_VALIDATORS_COLLECTION]
    doc = collection.find_one({"model_name": model_name})

    if not doc:
        raise ValueError(f"No schema found for model '{model_name}'")

    typed_fields = {
        field: (resolve_type(ftype), ...)
        for field, ftype in doc["fields"].items()
    }

    model_cls = create_model(f"{model_name.capitalize()}Model", **typed_fields)
    _model_cache[model_name] = model_cls
    return model_cls
