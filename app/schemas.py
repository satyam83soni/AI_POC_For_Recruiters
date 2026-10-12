from pydantic import BaseModel, ConfigDict, Field

from app.config import settings


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=500_000)


class IngestResponse(BaseModel):
    source: str
    status: str
    chunks: int


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class Source(BaseModel):
    source: str
    title: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool
