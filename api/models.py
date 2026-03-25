from pydantic import BaseModel
from typing import List, Any, Dict
import asyncio


class EmbedRequest(BaseModel):
    texts: List[str]
    chunk: bool = False


class QueuedRequest:
    def __init__(self, req_id: str, texts: List[str], chunk: bool):
        self.id = req_id
        self.texts = texts
        self.chunk = chunk
        self.future = asyncio.get_event_loop().create_future()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
