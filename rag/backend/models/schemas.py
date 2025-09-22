from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question or query")
    chat_history: List[ChatMessage] = Field(default=[], description="Previous chat messages")
    include_visualization: bool = Field(default=True, description="Whether to include visualizations")
    max_results: int = Field(default=5, description="Maximum number of relevant documents to retrieve")

class DocumentSource(BaseModel):
    file_id: str
    filename: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float

class Visualization(BaseModel):
    type: str  # "chart", "table", "image", "plotly", etc.
    data: Any  # Can be dict for plotly, string for base64 images, etc.
    config: Optional[Dict[str, Any]] = None
    description: str

class ChatResponse(BaseModel):
    response: str
    sources: List[DocumentSource]
    visualization: Optional[Dict[str, Any]] = None  # Flexible dict instead of strict model
    timestamp: Optional[datetime] = None

class ProcessedDocument(BaseModel):
    file_id: str
    filename: str
    file_type: str
    content: str
    metadata: Dict[str, Any]
    chunk_count: int
    processing_time: float

class UploadResponse(BaseModel):
    success: bool
    message: str
    files: List[ProcessedDocument]

class VisualizationRequest(BaseModel):
    query: str = Field(..., description="Visualization query")
    document_ids: List[str] = Field(default=[], description="Specific document IDs to visualize")

class AnalyticsRequest(BaseModel):
    query: str = Field(..., description="Analytics query")
    chart_types: List[str] = Field(default=["bar", "line", "scatter"], description="Preferred chart types")
    aggregation_level: str = Field(default="auto", description="Data aggregation level")
    
class HealthStatus(BaseModel):
    status: str
    services: Dict[str, bool]
    timestamp: datetime

# Simple response models for easier handling
class SimpleUploadResponse(BaseModel):
    success: bool
    message: str
    files: List[Dict[str, Any]]  # More flexible than strict model

class SimpleAnalyticsResponse(BaseModel):
    query: str
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    data_summary: Dict[str, Any]