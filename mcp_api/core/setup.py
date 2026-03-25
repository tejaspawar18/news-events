from qdrant_client.http.models import (  # type: ignore
    VectorParams,
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    OptimizersConfigDiff,
)
from mcp_api.core.connection import qdrant_client
from mcp_api.core.config import Config

def initialize_qdrant_collection():
    """
    Create a Qdrant collection with multivector fields: 'title' and 'summary'.
    Uses MAX_SIM for chunk similarity and indexes useful filters.
    """
    if qdrant_client.collection_exists(collection_name=Config.COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=Config.COLLECTION_NAME)
    
    is_staging = Config.ENV == "staging"
    
    vectors_config = {
        "title": VectorParams(
            size=1024,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        ),
        "summary": VectorParams(
            size=1024,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    }

    # Common kwargs for both environments
    kwargs = {
        "collection_name": Config.COLLECTION_NAME,
        "vectors_config": vectors_config,
        "on_disk_payload": not is_staging
    }

        # Add optimizers_config ONLY for staging (in-memory)
    if is_staging:
        kwargs["optimizers_config"] = OptimizersConfigDiff(
            default_segment_number=1,
            indexing_threshold=0  # Forces in-memory storage
        )

    qdrant_client.create_collection(**kwargs)

    # Index filters for search
    for field in ["scylla_id", "state", "district", "ac", "pc"]:
        qdrant_client.create_payload_index(
            collection_name=Config.COLLECTION_NAME,
            field_name=field,
            field_schema="keyword"
        )
