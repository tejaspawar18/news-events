import time
from collections import defaultdict
import prometheus_client
from starlette.requests import Request
from fastapi import APIRouter, HTTPException, Response
from typing import List, Literal, Union, Dict
from cassandra.concurrent import execute_concurrent_with_args

from mcp_api.models.schema import (
    InsertRequestScylla,
    BatchSearchRequest,
    BatchSearchResponse,
    FetchRequest,
    HybridSearchRequest,
    HybridSearchResponse,
)
from mcp_api.core.setup import initialize_qdrant_collection
from mcp_api.core.connection import (
    SCYLLA_OBJ,
    qdrant_client,
)
from mcp_api.core.config import Config
from mcp_api.models.field_validator_setup import setup_field_validators
from mcp_api.services.embedding import get_embedding
from mcp_api.services.qdrant_service import (
    embed_documents,
    search_multivector_field,
    batch_search_field_multivector,
    batch_hybrid_search_qdrant,
)
from mcp_api.services.logging_config import get_logger
from mcp_api.monitoring.metric_v2 import (
    metrics_class,
    track_operation,
)

logger = get_logger("qdrant_api")

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request = None):
    """Expose Prometheus metrics for the FastAPI application.

    Args:
        request (Request, optional): The incoming request object. Defaults to None.

    Returns:
        _type_: _description_
    """
    return Response(
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
        content=prometheus_client.generate_latest(),
    )


@router.post("/initialize/")
def initialize():
    """
    Initialize the Qdrant collection and set up field-level validators.

    This endpoint sets up the required collection in Qdrant (if not already created)
    and configures any necessary validators for incoming data models.
    """
    try:
        initialize_qdrant_collection()
        setup_field_validators()
        logger.info("Initialization complete", extra={"event": "initialize_success"})
        return {"status": "initialized"}
    except Exception as e:
        metrics_class.record_error("initialization_error", "error", "initialize")
        logger.error("Initialization failed", extra={"event": "initialize_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/")
@track_operation("insert_documents")
def insert_documents(payloads: List[InsertRequestScylla], request: Request):
    """
    Inserts documents into MongoDB, ScyllaDB, and Qdrant.
    Each document stores two Qdrant points:
      - One for title (with multivector chunks)
      - One for summary_details (with multivector chunks)
    """

    logger.info(
        "Insert documents called",
        extra={"event": "insert_documents_start", "payload_count": len(payloads)},
    )
    try:
        start_time = time.time()
        metrics_class.record_document_processing("insert", len(payloads), "processing")
        # events_mongo_model = get_model("events")
        scylla_docs = []

        # for payload in payloads:
        #     payload_mongo = InsertRequestMongo(
        #         scylla_id=payload.id,
        #         root_id=payload.root_id,
        #         publish_time=str(payload.published_time),
        #         updated_time=str(payload.updated_at),
        #         state=payload.state,
        #         category=payload.category,
        #         topic=payload.title,
        #     )
        #     mongo_docs.append(events_mongo_model(**payload_mongo.dict()))
        #     scylla_docs.append(events_scylla_model(**payload.dict()))
        scylla_docs = payloads
        titles = [doc.title + "_" + doc.summary or "" for doc in scylla_docs]
        summaries = [doc.summary_details or "" for doc in scylla_docs]

        embedding_start = time.time()
        title_embeddings_batch = get_embedding(titles)
        summary_embeddings_batch = get_embedding(summaries)

        embedding_duration = time.time() - embedding_start

        metrics_class.record_embedding_generation(embedding_duration, len(payloads))

        # Track document embedding and insertion
        with metrics_class.observe_database_operation("multi", "embed_and_insert"):
            doc_ids = embed_documents(
                # mongo_docs,
                scylla_docs,
                title_embeddings_batch,
                summary_embeddings_batch,
            )

        # Record successful processing
        metrics_class.record_document_processing("insert", len(payloads), "success")
        total_duration = time.time() - start_time
        logger.info(
            "Documents embedded and inserted",
            extra={
                "event": "insert_success",
                "doc_count": len(doc_ids),
                "total_duration": total_duration,
                "embedding_duration": embedding_duration,
            },
        )

        return {"status": "inserted", "doc_ids": doc_ids, "count": len(doc_ids)}

    except Exception as e:
        metrics_class.record_document_processing("insert", len(payloads), "error")
        metrics_class.record_error("document_insert_failed", "error", "documents")
        logger.error(
            "Insert failed",
            exc_info=True,
            extra={"event": "insert_error", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/")
@track_operation("search_field")
def search_by_field(
    query: str,
    field: Literal["title+summary", "summary_details"] = "title+summary",
    limit: int = 3,
    threshold: float = 0.3,
    include_score: bool = True
    ):
    """
    Search multivector Qdrant by field using MAX_SIM.

    Args:
        query (str): The search query string.
        field (Literal): The field to search ("title+summary" or "summary_details").
        limit (int): Maximum number of results to return.
        threshold (float): Similarity threshold for filtering results.
        include_score (bool): Whether to include similarity scores in the results.

    Returns:
        dict: Dictionary containing the search results.

    Raises:
        HTTPException: If an error occurs during search.
    """
    logger.info(
        "Field search called",
        extra={
            "event": "search_start",
            "query": query,
            "field": field,
            "limit": limit,
            "threshold": threshold
        }
    )

    try:
        start_time = time.time()
        results = search_multivector_field(query, field, limit, threshold, include_score)

        # Extract similarity scores for metrics
        similarity_scores = []
        result_count = 0
        if isinstance(results, list):
            result_count = len(results)
            if include_score and results:
                similarity_scores = [r.get('score', 0) for r in results if isinstance(r, dict)]

        # Record search metrics
        metrics_class.record_search_metrics(
            search_type=field,
            query=query,
            results_count=result_count,
            threshold=threshold,
            similarity_scores=similarity_scores
        )

        duration = time.time() - start_time
        logger.info(
            "Field search completed",
            extra={
                "event": "search_success",
                "result_count": result_count,
                "duration": duration
            }
        )

        return {"results": results}

    except Exception as e:
        metrics_class.record_error("search_failed", "error", "search")
        logger.error("Field search failed", extra={"event": "search_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/batch/")
@track_operation("batch_search")
def batch_search_field_api(payload: BatchSearchRequest, request: Request) -> BatchSearchResponse:
    """
    Perform batch search over a specific field using multivector MAX_SIM aggregation.

    Args:
        payload (BatchSearchRequest): Batch search request containing queries and parameters.
        request (Request): FastAPI request object.

    Returns:
        BatchSearchResponse: Response containing batch search results and metrics.

    Raises:
        HTTPException: If an error occurs during batch search.
    """
    logger.info(
        "Batch field search called",
        extra={
            "event": "batch_search_field_start",
            "num_queries": len(payload.queries),
            "field": payload.where
        }
    )

    try:
        start_time = time.time()

        # Record batch size
        metrics_class.batch_size.labels(
            operation="batch_search",
            service="mcp_api"
        ).observe(len(payload.queries))
        field_dict = {"summary": "summary_details", "title+summary": "title"}
        results = batch_search_field_multivector(
            queries=payload.queries,
            field=field_dict.get(payload.where, 'title'),
            threshold=payload.threshold,
            include_score=payload.include_score,
            limit=payload.limit
        )

        # Calculate success/failure counts
        successful_queries = sum(1 for r in results.values() if r)
        failed_queries = sum(1 for r in results.values() if not r)

        # Record metrics for each query
        for i, query in enumerate(payload.queries):
            query_results = results.get(i, [])
            result_count = len(query_results) if query_results else 0

            # Extract similarity scores
            similarity_scores = []
            if payload.include_score and query_results:
                similarity_scores = [r.get('score', 0) for r in query_results if isinstance(r, dict)]

            metrics_class.record_search_metrics(
                search_type=f"batch_{payload.where}",
                query=query,
                results_count=result_count,
                threshold=payload.threshold,
                similarity_scores=similarity_scores
            )

        execution_time = time.time() - start_time

        logger.info(
            "Batch field search completed",
            extra={
                "event": "batch_search_field_success",
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "execution_time": execution_time
            }
        )

        return BatchSearchResponse(
            total_queries=len(payload.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            execution_time=execution_time,
            results=results
        )

    except Exception as e:
        metrics_class.record_error("batch_search_failed", "error", "batch_search")
        logger.error("Batch search field failed", extra={"event": "batch_search_field_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch field search failed: {e}")


@router.post("/hybrid_search/batch/", response_model=HybridSearchResponse)
@track_operation("hybrid_search")
def hybrid_search(payload: HybridSearchRequest, request: Request) -> HybridSearchResponse:
    """
    Hybrid Search API endpoint combining vector search with metadata filtering.

    Performs a semantic search over either the "title" or "summary" fields using Qdrant vector similarity,
    restricted to documents filtered by metadata (state, category, and date range) from MongoDB.
    
    Workflow:
    1. Filters documents from MongoDB based on provided metadata filters.
    2. Obtains vector embeddings for the batch of queries.
    3. Performs Qdrant vector search restricted to Mongo-filtered document IDs.
    4. Aggregates, ranks, and returns the most relevant documents per query, including similarity scores.

    Payload Body:
    - queries: List of search strings to be semantically matched.
    - field: "title" or "summary" — the document field to search vectors within.
    - state: Optional list of states to filter documents.
    - category: Optional list of categories to filter documents.
    - threshold: 
    - date_from and date_to: Optional ISO8601 datetime strings to filter publish times.
    - limit: Number of results per query (default 5).
    - include_score: Whether to include similarity score in the results.

    Responses:
    - total_queries: Number of queries processed.
    - successful_queries: Number of queries for which results were found.
    - failed_queries: Number of queries with no results.
    - results: Dictionary mapping query index to list of matched documents with metadata and optional scores.

    Errors:
    - Returns HTTP 500 with details if embedding generation or vector search fails.
    
    Example:
    ```
    {
      "queries": ["mental health support in Delhi University"],
      "field": "title+summary",
      "state": ["Delhi"],
      "threshold" : 0,
      "date_from": "2025-08-01T00:00:00+05:30",
      "date_to": "2025-08-05T23:59:59+05:30",
      "limit": 5,
      "include_score": true
    }
    ```
    """
    logger.info("Hybrid search initiated", extra={
        "event": "hybrid_search_start",
        "query_count": len(payload.queries),
        "filters": {
            "state": payload.state,
            "date_from": payload.date_from,
            "date_to": payload.date_to,
        }
    })
    try:
        start_time = time.monotonic()

        metrics_class.batch_size.labels(
            operation="hybrid_search",
            service="mcp_api"
        ).observe(len(payload.queries))
        
        field_dict = {"summary_details": "summary", "title+summary": "title"}
        # 2. Qdrant hybrid search (call service)
        qdrant_start = time.time()
        out_results = batch_hybrid_search_qdrant(
            queries=payload.queries,
            field=field_dict.get(payload.field),
            state=payload.state,
            date_from=payload.date_from,
            date_to=payload.date_to,
            threshold=payload.threshold,
            limit=payload.limit,
            include_score=payload.include_score,
            # inject dependencies:
            get_embedding=get_embedding,
            qdrant_client=qdrant_client,
        )
        qdrant_duration = time.time() - qdrant_start
        successful_queries = sum(1 for v in out_results.values() if v)
        failed_queries = sum(1 for v in out_results.values() if not v)
        
        # Record detailed metrics for each query
        for i, query in enumerate(payload.queries):
            query_results = out_results.get(i, [])
            result_count = len(query_results) if query_results else 0
            
            # Extract similarity scores
            similarity_scores = []
            if payload.include_score and query_results:
                similarity_scores = [r.get('score', 0) for r in query_results if isinstance(r, dict)]

        metrics_class.record_search_metrics(
                search_type=f"hybrid_{payload.field}",
                query=query,
                results_count=result_count,
                threshold=payload.threshold,
                similarity_scores=similarity_scores
            )
        
        total_duration = time.time() - start_time
        
        # Check for slow queries with more granular thresholds
        if total_duration > 5.0:
            metrics_class.record_slow_query("hybrid_search", "5s")
        if total_duration > 10.0:
            metrics_class.record_slow_query("hybrid_search", "10s")
        if total_duration > 15.0:
            metrics_class.record_slow_query("hybrid_search", "15s")

        # Record component timing
        logger.info("Hybrid search completed", extra={
            "event": "hybrid_search_complete",
            "total_queries": len(payload.queries),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "total_duration": round(total_duration, 3),
            "qdrant_duration": round(qdrant_duration, 3),
        })
        return HybridSearchResponse(
            total_queries=len(payload.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            results=out_results
        )
    except Exception as e:
        metrics_class.record_error("hybrid_search_failed", "error", "hybrid_search")
        logger.error("Hybrid search failed", exc_info=True, extra={
            "event": "hybrid_search_error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")

@router.post("/scylla/fetch-Topics")
@track_operation("fetch_documents")
def fetch_documents(payload: FetchRequest, request: Request):
    """
    Fetch documents from the ScyllaDB `events` table using a batch of (root_id, id) keys.

    This endpoint accepts a list of UUID key pairs representing the partition key (`root_id`) and 
    clustering key (`id`) for each document. It performs a concurrent batched query using 
    `execute_concurrent_with_args` and returns all matching rows as a dictionary, 
    keyed by the stringified `id`.

    Parameters:
    -----------
    request : FetchRequest
        A Pydantic model containing a list of document key pairs (root_id, id) to query.

    Returns:
    --------
    dict
        A dictionary with the key `"documents"` mapping to another dictionary of document records, 
        where each key is the document `id` as a string and each value is the corresponding row from ScyllaDB.

    Raises:
    -------
    HTTPException
        If any error occurs during the query execution or data fetching.
    """
    

    logger.info(
        "Fetch documents called",
        extra={
            "event": "fetch_documents_start",
            "num_keys": len(payload.keys)
        }
    )
    
    try:
        start_time = time.time()
        
        # Separate keys by type
        ids = []
        only_root_ids = []
        for key in payload.keys:
            if key.id is not None:
                ids.append((key.root_id, key.id))
            else:
                only_root_ids.append((key.root_id,))

        # Record batch sizes
        if ids:
            metrics_class.batch_size.labels(
                operation="fetch_by_id",
                service="mcp_api"
            ).observe(len(ids))
            
        if only_root_ids:
            metrics_class.batch_size.labels(
                operation="fetch_by_root_id",
                service="mcp_api"
            ).observe(len(only_root_ids))
        
        results = defaultdict(list)
        session = SCYLLA_OBJ.get_session()
        
        stmt = session.prepare("SELECT * FROM events WHERE root_id=? AND id=?")
        stmt_tree = session.prepare("SELECT * FROM events WHERE root_id=?")
        
        if ids:
            with metrics_class.observe_database_operation("scylladb", "fetch_by_id"):
                results['node'] = execute_concurrent_with_args(session, stmt, ids)
                
        if only_root_ids:
            with metrics_class.observe_database_operation("scylladb",
                                                          "fetch_by_root_id"):
                results['tree'] = execute_concurrent_with_args(session, stmt_tree, only_root_ids)
            
        
        documents: Dict[str, Union[Dict, List[Dict]]] = defaultdict(list)
        successful_fetches = 0
        failed_fetches = 0
        for result_set in results.get("node", []):
            success, result = result_set
            if success:
                for row in result:
                    key = str(row.id)
                    documents[key] = dict(row._asdict())
                    successful_fetches += 1
            else:
                failed_fetches += 1
                    

        for result_set in results.get("tree", []):
            success, result = result_set
            if success:
                for row in result.all():
                    key = str(row.root_id)
                    documents[key].append(dict(row._asdict()))
                    successful_fetches += 1
            else:
                failed_fetches += 1

        metrics_class.record_document_processing("fetch", successful_fetches, "success")
        if failed_fetches > 0:
            metrics_class.record_document_processing("fetch", failed_fetches, "failed")

        total_duration = time.time() - start_time
        
        logger.info(
            "Fetch documents completed",
            extra={
                "event": "fetch_documents_success",
                "document_count": len(documents),
                "successful_fetches": successful_fetches,
                "failed_fetches": failed_fetches,
                "duration": total_duration
            }
        )

        return {"documents": documents}

    except Exception as e:
        metrics_class.record_error("fetch_documents_failed", "error", "fetch_documents")
        logger.error("Fetch documents failed", extra={"event": "fetch_documents_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/admin/update-collection-size")
# async def update_collection_size():
#     """Admin endpoint to update Qdrant collection size metric"""
#     try:
#         # Get actual collection size from Qdrant
#         collection_info = qdrant_client.get_collection(COLLECTION_NAME)
#         if collection_info:
#             size = collection_info.points_count
#             metrics_class.update_qdrant_collection_size(COLLECTION_NAME, size)
#             return {"collection": COLLECTION_NAME, "size": size}
#         else:
#             return {"error": "Collection not found"}
#     except Exception as e:
#         metrics_class.record_error("collection_size_update_failed", "error", "admin")
#         raise HTTPException(status_code=500, detail=str(e))