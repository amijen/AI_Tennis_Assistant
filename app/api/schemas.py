"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Incoming question from the frontend."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A question about tennis rules",
        examples=["How does the tie-break work?"],
    )

    conversation_id: str = Field(
        default="default",
        min_length = 1,
        max_length = 64,
        description="Session/Thread ID for maintaining conversation memory",
    )

    @field_validator("conversation_id")
    @classmethod
    def check_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("conversation_id can only contain letters, numbers, -, _")
        return v


class Source(BaseModel):
    """A cited rulebook excerpt."""

    document: str
    page: int | None = None
    similarity: float


class AskResponse(BaseModel):
    """Complete answer returned by the agent."""

    answer: str
    document_filter: str | None = None
    steps: list[dict] = []


class HealthResponse(BaseModel):
    """Service health status."""

    status: str
    database: str
    documents: int
    chunks: int