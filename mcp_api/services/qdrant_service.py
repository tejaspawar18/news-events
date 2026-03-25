from typing import List, Any, Literal, Dict, Optional
import uuid
import json
from collections import defaultdict
from qdrant_client.http.models import ( # pyright: ignore[reportMissingImports]
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    SearchRequest,
    NamedVector,
)
from cassandra.query import BatchStatement #pyright: ignore[reportMissingImports]

from mcp_api.core.connection import (
    qdrant_client,
    collection,
    SCYLLA_OBJ,
)
from mcp_api.models.udts import (
    EventPartyUDT,
    EventGeoLocationUDT,
    EventPartySentimentUDT,
)
from mcp_api.core.config import Config
from mcp_api.services.embedding import get_embedding
from mcp_api.models.schema import InsertRequestScylla


def embed_documents(
    # docs: List,
    scylla_docs: List,
    title_embeddings_batch: List[List[List[float]]],
    summary_embeddings_batch: List[List[List[float]]]
) -> List[str]:
    doc_ids = []
    qdrant_points = []

    for i, scylla_doc in enumerate(scylla_docs):
        scylla_id = str(scylla_doc.id)
        root_id = str(scylla_doc.root_id)
        state = str(scylla_doc.state)

        doc_ids.append(scylla_id)

        # Convert single to multivector
        title_vectors = title_embeddings_batch[i]
        if title_vectors and isinstance(title_vectors[0], (float, int)):
            title_vectors = [title_vectors]

        summary_vectors = summary_embeddings_batch[i]
        if summary_vectors and isinstance(summary_vectors[0], (float, int)):
            summary_vectors = [summary_vectors]

        # Qdrant point: title
        qdrant_points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{scylla_id}")),
                vector={"title": title_vectors, "summary": summary_vectors},
                payload={
                    "scylla_id": scylla_id,
                    "root_id": root_id,
                    "state": state,
                }
            )
        )


    # Insert Qdrant
    if qdrant_points:
        try:
            qdrant_client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=qdrant_points
            )
        except Exception as e:
            print(f"Qdrant upsert failed: {e}")

    # Insert Scylla
    try:
        batch_upsert_scylla(scylla_docs)
    except Exception as e:
        print(f"Scylla batch upsert failed: {e}")

    return doc_ids


def search_multivector_field(
    query: str,
    field: Literal["title+summary", "summary_details"],
    limit: int = 5,
    threshold: float = 0.3,
    include_score: bool = True
) -> List[Dict]:
    vectors = get_embedding([query])[0]

    if not vectors:
        return []

    # Ensure it's List[List[float]]
    if isinstance(vectors[0], float):
        vectors = [vectors]

    field_dict = {"summary_details": "summary", "title+summary": "title"}
    field = field_dict.get(field)

    search_requests = [
        SearchRequest(
            vector=NamedVector(name=field, vector=vec),
            limit=limit,
            with_payload=True,
            with_vector=True,
            score_threshold=threshold,
            # filter=Filter(must=[
            #     FieldCondition(key="field", match=MatchValue(value=field))
            # ])
        )
        for vec in vectors
    ]
    

    responses = qdrant_client.search_batch(
        collection_name=Config.COLLECTION_NAME,
        requests=search_requests
    )
    

    # Aggregate MAX_SIM by scylla_id
    score_map = defaultdict(float)
    for result in responses:
        for pt in result:
            sid = pt.payload.get("scylla_id")
            if sid:
                score_map[sid] = max(score_map[sid], pt.score)

    if not score_map:
        return []

    # Fetch Mongo docs
    mongo_docs = collection.find({"_id": {"$in": list(score_map.keys())}})
    doc_map = {doc["_id"]: doc for doc in mongo_docs}

    sorted_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
    results = []

    for sid, score in sorted_docs:
        doc = doc_map.get(sid)
        if not doc:
            continue
        if include_score:
            doc["similarity_score"] = score
        results.append(doc)

    return results


def batch_search_field_multivector(
    queries: List[str],
    field: Literal['title', 'summary'] = 'title',
    threshold: float = 0.0,
    include_score: bool = True,
    limit: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    if not queries:
        return {}

    embeddings = get_embedding(queries)

    if len(embeddings) != len(queries):
        raise ValueError("Mismatch between query and embedding counts")

    search_requests = []
    query_index_map = []

    for i, embedding in enumerate(embeddings):
        if not embedding:
            continue
        chunks = [embedding] if isinstance(embedding[0], (float, int)) else embedding
        for chunk in chunks:
            search_requests.append(
                SearchRequest(
                    vector=NamedVector(name=field, vector=chunk),
                    limit=limit,
                    with_payload=True,
                    score_threshold=threshold,
                )
            )
            query_index_map.append(i)

    search_results = qdrant_client.search_batch(
        collection_name=Config.COLLECTION_NAME,
        requests=search_requests
    )

    per_query_scores = defaultdict(lambda: defaultdict(float))
    all_scylla_ids = set()
    root_ids_map = {}
    for i, result in enumerate(search_results):
        query_idx = query_index_map[i]
        for point in result:
            sid = point.payload.get("scylla_id")
            sroot_id = point.payload.get("root_id")
            root_ids_map[sid] = sroot_id
            if sid:
                all_scylla_ids.add(sid)
                per_query_scores[query_idx][sid] = max(per_query_scores[query_idx][sid], point.score)

    final_results = {}

    for query_idx, sid_score_map in per_query_scores.items():
        doc_list = []
        sorted_docs = sorted(sid_score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        for sid, score in sorted_docs:
            doc = {
                "id": sid,
                "root_id": root_ids_map[sid]
            }
            if include_score:
                doc["similarity_score"] = score
            doc_list.append(doc)
        final_results[query_idx] = doc_list

    for i, query in enumerate(queries):
        if i not in final_results:
            final_results[i] = []

    return final_results


def batch_hybrid_search_qdrant(
    queries: List[str],
    field: str,
    state: List[str],
    date_from: Optional[str],
    date_to: Optional[str],
    limit: int = 5,
    threshold: float = 0,
    include_score: bool = True,
    get_embedding=None,
    qdrant_client=None,
) -> Dict[int, List[Dict[str, Any]]]:
    embeddings = get_embedding(queries)
    if len(embeddings) != len(queries):
        raise ValueError("Embedding count mismatch")

    search_requests = []
    query_index_map = []
    
    for qidx, embedding in enumerate(embeddings):
        emb_chunks = [embedding] if isinstance(embedding[0], (float, int)) else embedding
        for chunk in emb_chunks:
            must_clause = [
                FieldCondition(key="state", match=MatchAny(any=state)),
            ]
            search_requests.append(
                SearchRequest(
                    vector=NamedVector(name=field, vector=chunk),
                    limit=limit,
                    with_payload=True,
                    score_threshold=threshold,
                    filter=Filter(must=must_clause),
                )
            )
            query_index_map.append(qidx)

    search_results = qdrant_client.search_batch(
        collection_name=Config.COLLECTION_NAME, requests=search_requests
    )

    per_query_docs = defaultdict(list)
    for i, result_points in enumerate(search_results):
        qidx = query_index_map[i]
        for point in result_points:
            per_query_docs[qidx].append((point.score, point))

    out_results = {}
    for idx in range(len(queries)):
        all_points = per_query_docs.get(idx, [])
        seen_ids = set()
        ranked = []
        for score, pt in sorted(all_points, key=lambda x: x[0], reverse=True):
            sid = pt.payload.get("scylla_id")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            doc = {
                "id": sid,
                "qdrant_point_id": pt.id,
                "root_id": pt.payload.get("root_id"),
            }
            if include_score:
                doc["similarity_score"] = score
            ranked.append(doc)
            if len(ranked) >= limit:
                break
        out_results[idx] = ranked
    return out_results


def batch_upsert_scylla(docs: List[InsertRequestScylla]):
    if not docs:
        return

    session = SCYLLA_OBJ.get_session()

    def serialize_doc(doc):
        raw = doc.dict(exclude_none=True)

        for k, v in raw.items():
            if k in ['ac', 'pc', 'district', 'candidates'] and isinstance(v, list):
                raw[k] = [EventGeoLocationUDT(**item) for item in v]
            elif k == 'parties' and isinstance(v, list):
                raw[k] = [EventPartyUDT(**item) for item in v]
            elif k == 'political_inclination_party' and isinstance(v, dict):
                raw[k] = EventPartySentimentUDT(**v)
            elif k in ['sources', 's3_article_url', 'source_domain', 'top_images'] and isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        raw[k] = parsed
                except Exception as e:
                    print(f"Warning: Could not parse field {k} as JSON list: {e}")
        return raw

    try:
        doc_dicts = [serialize_doc(doc) for doc in docs]
    except Exception as e:
        return f"Error during document serialization: {e}"

    prepared_cache = {}

    query_backpropagate = """
        UPDATE events SET is_leaf = ?
        WHERE root_id = ? and id = ?
    """
    try:
        prepared_backpropagate = session.prepare(query_backpropagate)
    except Exception as e:
        print(f"Error preparing backpropagation statement: {e}")
        return f"Error preparing backpropagation statement: {e}"

    batch = BatchStatement()

    for i, doc in enumerate(doc_dicts):
        try:

            fields = sorted(doc.keys())
            key = tuple(fields)

            if key not in prepared_cache:
                placeholders = ', '.join(['?'] * len(fields))
                columns = ', '.join(fields)
                query = f"INSERT INTO events ({columns}) VALUES ({placeholders})"
                prepared_cache[key] = session.prepare(query)

            prepared = prepared_cache[key]

            values = [doc[field] for field in fields]
            batch.add(prepared, values)

            if doc.get("parent_id") and doc.get("id") != doc.get("parent_id"):
                batch.add(prepared_backpropagate, [False, doc.get("root_id"), doc.get("parent_id")])

        except Exception as doc_error:
            print(f"Error processing document {i + 1}: {doc_error}")
            print(f"Document data: {doc}")
            return f"Error processing document {i + 1}: {doc_error}"

    try:
        session.execute(batch)
        print(f"Batch of {len(docs)} documents upserted to ScyllaDB.")
        return f"Successfully upserted {len(docs)} documents"
    except Exception as e:
        print(f"Error executing batch: {e}")
        return f"Error executing batch: {e}"
