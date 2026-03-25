from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import ValidationError
from fastapi.responses import JSONResponse
import traceback
from typing import List, Any, Union
import asyncio
import uuid
import numpy as np
import logging
import torch
import time
import psutil

from api.model_loader import load_model_and_tokenizer
# from api.logger import get_loki_logger
from api.metric import *
from api.models import *
logger = logging.getLogger(__name__)
# logger = get_loki_logger('embed_server','/home/ubuntu/embed_server/api/embed_apis' )

MAX_TOKENS = 512
BATCH_SIZE = 16
BATCH_INTERVAL = 0.03  # 30ms
MAX_TOKENS_PER_BATCH = 20000

app = FastAPI(
    title="Multilingual E5-Large Embedding Service",
    description="High-performance embedding service for multilingual-e5-large model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

request_queue: asyncio.Queue = asyncio.Queue()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    field_errors = []
    model_name = None

    for error in exc.errors():
        loc = error.get("loc", [])
        msg = error.get("msg", "Invalid input")
        error_type = error.get("type", "unknown_error")

        # Try to extract the model name if available (usually at index 0 of loc)
        if len(loc) > 0 and isinstance(loc[0], str) and loc[0] != "body":
            model_name = loc[0]

        # Get the field name (typically last in `loc` list)
        field_name = loc[-1] if loc else "unknown"

        field_errors.append(f"Field `{field_name}` - {msg} ({error_type})")

    # Avoid duplicates
    simplified_errors = list(set(field_errors))

    # Log compactly
    logger.error(
        f"{request.method} {request.url.path} - Validation Error in {model_name}: {simplified_errors} | Body: [Truncated: {len(str(exc.body))} chars]"
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": f"Validation failed in {model_name}",
            "details": simplified_errors
        },
    )
    

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Get memory usage
        memory_info = {}
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        memory_info["cpu_percent"] = psutil.cpu_percent()
        memory_info["ram_usage_mb"] = psutil.virtual_memory().used / (1024 * 1024)
        
        return HealthResponse(
            status="healthy",
            model_loaded=getattr(app.state, "session", None) is not None,
            gpu_available=torch.cuda.is_available(),
            memory_usage=memory_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# @app.get("/model-info")
# async def get_model_info():
#     """Get model information"""
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     return {
#         "model_name": os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large"),
#         "device": str(model.device),
#         "cuda_available": torch.cuda.is_available(),
#         "max_workers": int(os.getenv("MAX_WORKERS", "4")),
#         "batch_size": int(os.getenv("BATCH_SIZE", "32"))
#     }


@app.on_event("startup")
async def startup_event():
    session, tokenizer = load_model_and_tokenizer()
    app.state.session = session
    app.state.tokenizer = tokenizer
    print("✅ Model and tokenizer loaded.")

    # 🔥 Warm-up embedding
    # dummy_text = ["This is a warm-up pass."]
    # inputs = tokenizer(dummy_text, return_tensors="np", padding=True, truncation=True, max_length=512)
    # ort_inputs = {
    #     "input_ids": inputs["input_ids"].astype(np.int64),
    #     "attention_mask": inputs["attention_mask"].astype(np.int64),
    # }

    # outputs = session.run(None, ort_inputs)
    # print(f"🔥 Warm-up completed. Output shape: {outputs[0].shape}")
    dummy_text = ["warmup"] * BATCH_SIZE

    inputs = tokenizer(
        dummy_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS
    )

    session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    })


    # 🚀 Start background batch worker
    asyncio.create_task(batch_worker(app))


@app.post("/embed")
async def embed(req: Union[EmbedRequest, List[str]], request: Request):
    REQUEST_COUNT.inc()
    req_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # Handle both formats: {"texts": [...]} and [...]
    if isinstance(req, list):
        texts = req
        chunk = True # Default or extract from another source if needed
    else:
        texts = req.texts
        chunk = req.chunk

    queued_req = QueuedRequest(req_id, texts, chunk)
    await request_queue.put(queued_req)
    result = await queued_req.future  # Wait for result
    
    duration = time.perf_counter() - start_time
    REQUEST_LATENCY.observe(duration)
    logger.info(f"[{req_id}] Request served in {duration:.3f}s | chunk={chunk} | texts={len(texts)}")
    return {"embeddings": result}


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    try:
        mask_expanded = attention_mask[..., None]
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    except Exception as e:
        logger.error(f"Mean pooling failed: {e}")
        raise

def chunk_text(text, tokenizer, max_length):
    # tokens = tokenizer.encode(text, add_special_tokens=False)
    tokenized = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=MAX_TOKENS
                    )["input_ids"]

    return [tokenized[i:i+max_length] for i in range(0, len(tokenized), max_length)]

async def batch_worker(app: FastAPI):
    print("🚀 Batch worker started.")
    
    tokenizer = app.state.tokenizer
    session = app.state.session
    
    while True:
        batch: List[QueuedRequest] = []

        try:
            first_item = await asyncio.wait_for(request_queue.get(), timeout=1)
            batch.append(first_item)
        except asyncio.TimeoutError:
            continue

        start_time = asyncio.get_event_loop().time()
        while len(batch) < BATCH_SIZE:
            wait_time = BATCH_INTERVAL - (asyncio.get_event_loop().time() - start_time)
            if wait_time <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(request_queue.get(), timeout=wait_time))
            except asyncio.TimeoutError:
                break

        try:
            flat_texts = []
            metadata = []
            total_tokens = 0

            for req in batch:
                for text in req.texts:
                    tokenized = tokenizer.encode(text, add_special_tokens=False)
                    if len(tokenized) > MAX_TOKENS and req.chunk:
                        chunks = chunk_text(text, tokenizer, MAX_TOKENS)
                        chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
                    else:
                        chunk_texts = [text]
                    
                    for ct in chunk_texts:
                        ct_tokens = tokenizer.encode(ct, add_special_tokens=False)
                        if total_tokens + len(ct_tokens) > MAX_TOKENS_PER_BATCH:
                            break  # Stop batching
                        flat_texts.append(ct)
                        metadata.append(req)
                        total_tokens += len(ct_tokens)

            if not flat_texts:
                logger.warning("Empty batch after processing. Continuing.")
                continue

            inputs = tokenizer(flat_texts, return_tensors="np", padding=True, truncation=True, max_length=MAX_TOKENS)
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            token_count = int(np.sum(attention_mask))
            batch_size = len(flat_texts)

            BATCH_SIZE_GAUGE.set(batch_size)
            BATCH_TOKENS_GAUGE.set(token_count)

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            inference_start = time.perf_counter()
            with BATCH_DURATION.time():
                # 🔥 Offload inference to a thread to avoid blocking the event loop
                outputs = await asyncio.to_thread(session.run, None, ort_inputs)
            inference_time = time.perf_counter() - inference_start
            
            logger.info(f"🔁 Inference complete: {len(flat_texts)} texts in {inference_time:.3f}s")

            pooled = mean_pool(outputs[0], attention_mask)

            # Map back to futures
            idx = 0
            for req in batch:
                text_count = sum(1 for m in metadata if m.id == req.id)
                embeddings = pooled[idx:idx + text_count].tolist()
                idx += text_count
                if not req.future.done():
                    req.future.set_result(embeddings)
        except Exception as e:
            logger.error(f"Error in batch worker: {e}")
            logger.error(traceback.format_exc())
            # Notify all futures in the failed batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
