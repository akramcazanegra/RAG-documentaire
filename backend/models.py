# backend/models.py
from typing import List
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., description="User question in natural language")
    top_k: int = Field(4, ge=1, le=10, description="How many chunks to retrieve")

class SourceChunk(BaseModel):
    file: str
    chunk: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class HealthResponse(BaseModel):
    status: str = "ok"