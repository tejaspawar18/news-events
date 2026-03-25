import json
from datetime import datetime
from typing import List, Optional, Any, Literal, Dict
from uuid import UUID
from pydantic import BaseModel, Field


class InsertRequestMongo(BaseModel):
    scylla_id: UUID
    root_id: UUID
    publish_time: str
    updated_time: str
    state: str
    category: str
    topic: str
    

class EventGeoLocation(BaseModel):
    # Define your actual fields here
    name: str
    id: Optional[str]


class EventParty(BaseModel):
    # Define your actual fields here
    party: str
    party_id: Optional[str]


class EventPartySentiment(BaseModel):
    sentiment: str  # e.g., "Pro-BJP"
    reason: Optional[str]
    
    
class InsertRequestScylla(BaseModel):
    ac: Optional[List[EventGeoLocation]]
    ancestor_count: Optional[int]
    article_count: Optional[int]
    candidates: Optional[List[EventGeoLocation]]
    category: str
    district: Optional[List[EventGeoLocation]]
    geography: Optional[str]
    id: UUID
    is_leaf: Optional[bool]
    media_links: Optional[List[str]]
    opinion: Optional[str]
    parent_id: Optional[UUID] = None
    parent_semantic_score: Optional[float]
    parties: Optional[List[EventParty]]
    pc: Optional[List[EventGeoLocation]]
    political_inclination_party: Optional[EventPartySentiment]
    published_time: datetime
    regional_language: Optional[str]
    research: Optional[str]
    root_id: UUID
    s3_article_url: Optional[List[str]]
    source_domain: Optional[List[str]]
    sources: Optional[List[str]]
    state: str
    summary: Any#Optional[str]
    summary_details: Any#Optional[str]
    summary_details_hindi: Optional[str]
    summary_details_regional: Optional[str]
    summary_hindi: Optional[str]
    summary_regional: Optional[str]
    tags: Optional[str]
    title: Optional[str]
    top_images: Optional[List[str]]
    updated_at: datetime
    
    def scylla_dict(self):
        data = self.model_dump()
        for k in ["ac", "pc", "district", "parties", "candidates"]:
            if hasattr(self, k):
                data[k] = json.dumps([x.model_dump() for x in getattr(self, k)])
        if hasattr(self, "political_inclination_party") and self.political_inclination_party:
            data["political_inclination_party"] = json.dumps(self.political_inclination_party.model_dump())
        return data

 
class BatchSearchRequest(BaseModel):
    """Simplified batch search request with uniform parameters"""
    queries: List[str] = Field(..., min_items=1, max_items=100, description="List of search query texts")
    where: Literal['title+summary', 'summary_details'] = Field(default='title+summary', description="Field to search in ('title+summary' or 'summary_details')")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    include_score: bool = Field(default=True, description="Whether to include similarity scores")
    limit: int = Field(default=3, ge=1, le=50, description="Maximum number of results per query")


class BatchSearchResponse(BaseModel):
    """Batch search response"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    execution_time: float
    results: Any


class DocumentKey(BaseModel):
    root_id: UUID
    id: Optional[UUID] = None


class FetchRequest(BaseModel):
    keys: List[DocumentKey]
    

class HybridSearchRequest(BaseModel):
    queries: List[str] = Field(..., example=["mental health students"])
    field: Literal["title+summary", "summary_details"] = "title+summary"
    state: Optional[List[str]] = None
    threshold: Optional[float] = 0
    date_from: Optional[str] = None  # ISO date string
    date_to: Optional[str] = None    # ISO date string
    limit: int = 5
    include_score: bool = True


class HybridSearchDocResult(BaseModel):
    id: str
    qdrant_point_id: str
    root_id: str
    similarity_score: Optional[float] = None


class HybridSearchResponse(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    results: Dict[int, List[HybridSearchDocResult]]
