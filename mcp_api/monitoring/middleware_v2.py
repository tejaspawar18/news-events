# mcp_api/middleware/metrics_middleware.py

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from mcp_api.monitoring.metric_v2 import metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically collect HTTP metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            try:
                body = await request.body()
                request_size = len(body)
            except:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        endpoint = self._get_endpoint_name(request.url.path)
        
        # Get response size
        response_size = 0
        if hasattr(response, 'headers') and 'content-length' in response.headers:
            response_size = int(response.headers['content-length'])
        
        # Record metrics
        metrics.record_http_request(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size
        )
        
        # Check for slow queries
        if duration > 2.0:
            metrics.record_slow_query(endpoint, "2s")
        if duration > 5.0:
            metrics.record_slow_query(endpoint, "5s")
        if duration > 10.0:
            metrics.record_slow_query(endpoint, "10s")
        
        return response
    
    def _get_endpoint_name(self, path: str) -> str:
        """Extract meaningful endpoint name from path"""
        # Remove query parameters and normalize path
        path = path.split('?')[0]
        
        # Map specific paths to readable names
        path_mapping = {
            '/initialize/': 'initialize',
            '/documents/': 'insert_documents',
            '/search/': 'search',
            '/search/batch/': 'batch_search',
            '/hybrid_search/batch/': 'hybrid_search',
            '/scylla/fetch-Topics': 'fetch_documents',
            '/metrics': 'metrics'
        }
        
        return path_mapping.get(path, path.replace('/', '_').strip('_') or 'root')


# Updated router with enhanced metrics integration
# mcp_api/api/routes/qdrant_router.py

from fastapi import APIRouter, Query, HTTPException, Response
from typing import List, Literal, Union, Dict, List
import time
from collections import defaultdict
import prometheus_client
from starlette.requests import Request

from cassandra.concurrent import execute_concurrent_with_args
from mcp_api.api.schema import (InsertRequestMongo,
                                InsertRequestScylla,
                                BatchSearchRequest,
                                BatchSearchResponse,
                                FetchRequest,
                                HybridSearchRequest,
                                HybridSearchResponse)    
from mcp_api.core.setup import initialize_qdrant_collection
from qdrant.event_agents.mcp_api.core.connection import (SCYLLA_PROD_OBJ,
                                 COLLECTION_NAME,
                                 collection, qdrant_client)
from mcp_api.models.field_validator_setup import setup_field_validators
from mcp_api.models.dynamic_model_builder import get_model
from mcp_api.services.embedding import get_embedding
from mcp_api.services.qdrant_service import (embed_documents,
                                             search_multivector_field,
                                             batch_search_field_multivector,
                                             batch_hybrid_search_qdrant,
                                             build_mongo_time_query)
from mcp_api.services.logging_config import get_logger
from mcp_api.monitoring.enhanced_metrics import metrics, track_operation, track_database_operation

logger = get_logger("qdrant_api")
router = APIRouter()

@router.get("/metrics")
async def get_metrics(request: Request = None):
    """Expose Prometheus metrics"""
    return Response(
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
        content=prometheus_client.generate_latest()
    )

@router.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    try:
        # You can add actual health checks here
        # e.g., check database connections, external services, etc.
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "checks": {
                "database": "ok",
                "qdrant": "ok",
                "embedding_service": "ok"
            }
        }
        
        metrics.set_application_state("running")
        return health_status
        
    except Exception as e:
        metrics.set_application_state("degraded")
        metrics.record_error("health_check_failed", "error", "health")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.post("/initialize/")
@track_operation("initialize")
def initialize():
    """
    Initialize the Qdrant collection and set up field-level validators.
    """
    try:
        initialize_qdrant_collection()
        setup_field_validators()
        logger.info("Initialization complete", extra={"event": "initialize_success"})
        return {"status": "initialized"}
    except Exception as e:
        metrics.record_error("initialization_error", "error", "initialize")
        logger.error("Initialization failed", extra={"event": "initialize_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/")
@track_operation("insert_documents")
def insert_documents(payloads: List[InsertRequestScylla], request: Request):
    """
    Inserts documents into MongoDB, ScyllaDB, and Qdrant.
    """
    logger.info(
        "Insert documents called",
        extra={"event": "insert_documents_start", "payload_count": len(payloads)}
    )
    
    try:
        start_time = time.time()
        
        # Record batch size
        metrics.record_document_processing("insert", len(payloads), "processing")
        
        events_mongo_model = get_model("events")
        events_scylla_model = InsertRequestScylla

        mongo_docs = []
        scylla_docs = []

        for payload in payloads:
            payload_mongo = InsertRequestMongo(
                scylla_id=payload.id,
                root_id=payload.root_id,
                publish_time=str(payload.published_time),
                updated_time=str(payload.updated_at),
                state=payload.state,
                category=payload.category,
                topic=payload.title,
            )
            mongo_docs.append(events_mongo_model(**payload_mongo.dict()))
            scylla_docs.append(events_scylla_model(**payload.dict()))

        titles = [doc.title + "_" + doc.summary or "" for doc in scylla_docs]
        summaries = [doc.summary_details or "" for doc in scylla_docs]
        
        # Track embedding generation
        embedding_start = time.time()
        title_embeddings_batch = get_embedding(titles)
        summary_embeddings_batch = get_embedding(summaries)
        embedding_duration = time.time() - embedding_start
        
        metrics.record_embedding_generation(embedding_duration, len(payloads))
        
        # Track document embedding and insertion
        with metrics.observe_database_operation("multi", "embed_and_insert"):
            doc_ids = embed_documents(mongo_docs, scylla_docs, title_embeddings_batch, summary_embeddings_batch)
        
        # Record successful processing
        metrics.record_document_processing("insert", len(payloads), "success")
        
        total_duration = time.time() - start_time
        logger.info(
            "Documents embedded and inserted",
            extra={
                "event": "insert_success",
                "doc_count": len(doc_ids),
                "total_duration": total_duration,
                "embedding_duration": embedding_duration
            }
        )
        
        return {"status": "inserted", "doc_ids": doc_ids, "count": len(doc_ids)}
        
    except Exception as e:
        metrics.record_document_processing("insert", len(payloads), "error")
        metrics.record_error("document_insert_failed", "error", "documents")
        logger.error("Insert failed", exc_info=True, extra={"event": "insert_error", "error": str(e)})
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
        metrics.record_search_metrics(
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
        metrics.record_error("search_failed", "error", "search")
        logger.error("Field search failed", extra={"event": "search_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/batch/")
@track_operation("batch_search")
def batch_search_field_api(payload: BatchSearchRequest, request: Request) -> BatchSearchResponse:
    """
    Perform batch search over a specific field using multivector MAX_SIM aggregation.
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
        metrics.batch_size.labels(
            operation="batch_search",
            service="mcp_api"
        ).observe(len(payload.queries))
        
        results = batch_search_field_multivector(
            queries=payload.queries,
            field=payload.where,
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
            
            metrics.record_search_metrics(
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
        metrics.record_error("batch_search_failed", "error", "batch_search")
        logger.error("Batch search field failed", extra={"event": "batch_search_field_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch field search failed: {e}")

@router.post("/hybrid_search/batch/", response_model=HybridSearchResponse)
@track_operation("hybrid_search")
def hybrid_search(payload: HybridSearchRequest, request: Request) -> HybridSearchResponse:
    """
    Hybrid Search API endpoint combining vector search with metadata filtering.
    """
    logger.info("Hybrid search initiated", extra={
        "event": "hybrid_search_start",
        "query_count": len(payload.queries),
        "filters": {
            "state": payload.state,
            "category": payload.category,
            "date_from": payload.date_from,
            "date_to": payload.date_to,
        }
    })
    
    try:
        start_time = time.time()
        
        # Record batch size
        metrics.batch_size.labels(
            operation="hybrid_search",
            service="mcp_api"
        ).observe(len(payload.queries))
        
        # 1. MongoDB filter phase
        mongo_start = time.time()
        mongo_query = {}
        if payload.state:
            mongo_query["state"] = {"$in": payload.state}
        if payload.category:
            mongo_query["category"] = {"$in": payload.category}
        if payload.date_from or payload.date_to:
            date_filter = build_mongo_time_query(payload.date_from, payload.date_to)
            if date_filter:
                mongo_query["publish_time"] = date_filter

        projection = {"_id": 1, "root_id": 1, "publish_time": 1}
        
        with metrics.observe_database_operation("mongodb", "filter_query"):
            mongo_docs = list(collection.find(mongo_query, projection))
        
        allowed_ids = [doc["_id"] for doc in mongo_docs]
        mongo_doc_map = {doc["_id"]: doc for doc in mongo_docs}
        mongo_duration = time.time() - mongo_start
        
        # Record MongoDB filtering metrics
        logger.info(f"MongoDB filter returned {len(allowed_ids)} documents in {mongo_duration:.3f}s")

        if not allowed_ids:
            # No documents match the filter
            empty_results = {i: [] for i in range(len(payload.queries))}
            return HybridSearchResponse(
                total_queries=len(payload.queries),
                successful_queries=0,
                failed_queries=len(payload.queries),
                results=empty_results
            )

        field_dict = {"summary": "summary_details", "title+summary": "title"}
        
        # 2. Qdrant hybrid search phase
        qdrant_start = time.time()
        out_results = batch_hybrid_search_qdrant(
            queries=payload.queries,
            field=field_dict.get(payload.field),
            allowed_ids=allowed_ids,
            mongo_doc_map=mongo_doc_map,
            threshold=payload.threshold,
            limit=payload.limit,
            include_score=payload.include_score,
            # inject dependencies:
            get_embedding=get_embedding,
            qdrant_client=qdrant_client,
            COLLECTION_NAME=COLLECTION_NAME
        )
        qdrant_duration = time.time() - qdrant_start
        
        # Calculate success/failure counts
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
            
            metrics.record_search_metrics(
                search_type=f"hybrid_{payload.field}",
                query=query,
                results_count=result_count,
                threshold=payload.threshold,
                similarity_scores=similarity_scores
            )
        
        total_duration = time.time() - start_time
        
        # Check for slow queries with more granular thresholds
        if total_duration > 5.0:
            metrics.record_slow_query("hybrid_search", "5s")
        if total_duration > 10.0:
            metrics.record_slow_query("hybrid_search", "10s")
        if total_duration > 15.0:
            metrics.record_slow_query("hybrid_search", "15s")
            
        # Record component timing
        logger.info("Hybrid search completed", extra={
            "event": "hybrid_search_complete",
            "total_queries": len(payload.queries),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "total_duration": round(total_duration, 3),
            "mongo_duration": round(mongo_duration, 3),
            "qdrant_duration": round(qdrant_duration, 3),
            "filtered_docs": len(allowed_ids)
        })
        
        return HybridSearchResponse(
            total_queries=len(payload.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            results=out_results
        )
        
    except Exception as e:
        metrics.record_error("hybrid_search_failed", "error", "hybrid_search")
        logger.error("Hybrid search failed", exc_info=True, extra={
            "event": "hybrid_search_error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")

@router.post("/scylla/fetch-Topics")
@track_operation("fetch_documents")
def fetch_documents(payload: FetchRequest, request: Request) -> Dict[str, Union[Dict, List[Dict]]]:
    """
    Fetch documents from the ScyllaDB `events` table using a batch of (root_id, id) keys.
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
            metrics.batch_size.labels(
                operation="fetch_by_id",
                service="mcp_api"
            ).observe(len(ids))
            
        if only_root_ids:
            metrics.batch_size.labels(
                operation="fetch_by_root_id",
                service="mcp_api"
            ).observe(len(only_root_ids))

        results = defaultdict(list)
        session = SCYLLA_PROD_OBJ.get_session()
        
        # Prepare statements
        stmt = session.prepare("SELECT * FROM events WHERE root_id=? AND id=?")
        stmt_tree = session.prepare("SELECT * FROM events WHERE root_id=?")
        
        # Execute queries with metrics tracking
        if ids:
            with metrics.observe_database_operation("scylladb", "fetch_by_id"):
                results['node'] = execute_concurrent_with_args(session, stmt, ids)
                
        if only_root_ids:
            with metrics.observe_database_operation("scylladb", "fetch_by_root_id"):
                results['tree'] = execute_concurrent_with_args(session, stmt_tree, only_root_ids)

        # Process results
        documents: Dict[str, Union[Dict, List[Dict]]] = defaultdict(list)
        successful_fetches = 0
        failed_fetches = 0

        # Process node results
        for result_set in results.get("node", []):
            success, result = result_set
            if success:
                row = result.one()
                if row:
                    key = f"{row.id}"
                    documents[key] = dict(row._asdict())
                    successful_fetches += 1
                else:
                    failed_fetches += 1
            else:
                failed_fetches += 1

        # Process tree results
        for result_set in results.get("tree", []):
            success, result = result_set
            if success:
                rows = result.all()
                if rows:
                    key = str(rows[0].root_id)  # Use root_id as key
                    documents[key] = [dict(row._asdict()) for row in rows]
                    successful_fetches += 1
                else:
                    failed_fetches += 1
            else:
                failed_fetches += 1

        # Record fetch metrics
        metrics.record_document_processing("fetch", successful_fetches, "success")
        if failed_fetches > 0:
            metrics.record_document_processing("fetch", failed_fetches, "failed")

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
        metrics.record_error("fetch_documents_failed", "error", "fetch_documents")
        logger.error("Fetch documents failed", extra={"event": "fetch_documents_error", "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints for monitoring

@router.get("/metrics/custom")
async def get_custom_metrics():
    """Get custom application metrics summary"""
    try:
        # You can add custom logic here to return application-specific metrics
        # This is useful for debugging and monitoring dashboards
        
        return {
            "application_metrics": {
                "status": "healthy",
                "timestamp": time.time(),
                # Add any custom metrics you want to expose via JSON
            }
        }
    except Exception as e:
        metrics.record_error("custom_metrics_failed", "error", "metrics")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/update-collection-size")
async def update_collection_size():
    """Admin endpoint to update Qdrant collection size metric"""
    try:
        # Get actual collection size from Qdrant
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        if collection_info:
            size = collection_info.points_count
            metrics.update_qdrant_collection_size(COLLECTION_NAME, size)
            return {"collection": COLLECTION_NAME, "size": size}
        else:
            return {"error": "Collection not found"}
    except Exception as e:
        metrics.record_error("collection_size_update_failed", "error", "admin")
        raise HTTPException(status_code=500, detail=str(e))