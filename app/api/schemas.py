"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


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
        description="Session/Thread ID for maintaining conversation memory",
    )


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