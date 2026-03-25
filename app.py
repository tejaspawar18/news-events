from fastapi import FastAPI
from mcp_api.services.logging_config import get_logger, setup_request_logging
from mcp_api.api.routes import router as mcp_router

# Setup logging
logger = get_logger(__name__)
setup_request_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Qdrant API Service",
    description="High-performance APIs for document embedding, semantic search using Qdrant vector database",
    version="1.0.0"
)

app.include_router(mcp_router)