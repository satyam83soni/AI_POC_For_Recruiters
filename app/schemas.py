# >> Pydantic: BaseModel = parent class for data shapes that validate themselves (like a zod schema
# >> + TS type in one). ConfigDict configures a model; Field adds rules like min/max length.
from pydantic import BaseModel, ConfigDict, Field

# >> Import the shared settings object (used below for the default top_k).
from app.config import settings


# >> Shape of the JSON body for POST /ingest. Inheriting BaseModel makes FastAPI validate the
# >> request automatically and return a 422 error if it doesn't match.
class IngestRequest(BaseModel):
    # >> `extra="forbid"`: reject the request if it contains any field not declared below.
    model_config = ConfigDict(extra="forbid")

    # >> Three required string fields, each 1..500 chars (text up to 500,000 chars;
    # >> `500_000` is 500000 written with an underscore for readability).
    # >> source = unique id/URL of the document, title = its title, text = full document content.
    source: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=500_000)


# >> Shape of the JSON returned by POST /ingest.
class IngestResponse(BaseModel):
    # >> Fields with only a type hint (no default) are required: the document source,
    # >> a status string (e.g. created/replaced), and how many chunks were stored (`int` = integer).
    source: str
    status: str
    chunks: int


# >> Shape of the JSON body for POST /ask and POST /ask/stream.
class AskRequest(BaseModel):
    # >> Again: unknown fields in the body are an error.
    model_config = ConfigDict(extra="forbid")

    # >> The user's question, required, 3..2000 characters.
    question: str = Field(min_length=3, max_length=2000)
    # >> Optional number of chunks to retrieve; defaults to settings.top_k and must be
    # >> between 1 and 20 (`ge` = greater-or-equal, `le` = less-or-equal).
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


# >> One retrieved source shown alongside an answer.
class Source(BaseModel):
    # >> The document id, its title, and the similarity score (`float` = decimal number, e.g. 0.73).
    source: str
    title: str
    similarity: float


# >> Shape of the JSON returned by POST /ask.
class AskResponse(BaseModel):
    # >> The answer text; `list[Source]` = a list (array) where each item is a Source object
    # >> (like `Source[]` in TS); `cached` = bool (true/false) saying if it came from Redis.
    answer: str
    sources: list[Source]
    cached: bool
