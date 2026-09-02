import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.db.database import engine
from app.graph.builder import agent_graph
from app.graph.state import initial_state

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health", response_model=HealthResponse)
def health():
    """Check API + database connectivity and report index size."""
    try:
        with engine.connect() as conn:
            docs = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            chunks = conn.execute(text("SELECT COUNT(*) FROM child_chunks")).scalar()
        return HealthResponse(
            status="ok",
            database="connected",
            documents=docs or 0,
            chunks=chunks or 0,
        )
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")


# ============================================================
# ASK (blocking — returns the full answer at once)
# ============================================================

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        config = {"configurable": {"thread_id": payload.conversation_id}}
        result = agent_graph.invoke(
            {"question": payload.question},
            config=config,
        )
    except Exception as exc:
        logger.exception("Agent graph failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return AskResponse(
        answer=result.get("answer", ""),
        document_filter=result.get("document_filter"),
        steps=result.get("steps", []),
    )

# ============================================================
# ASK STREAM (Server-Sent Events)
# ============================================================

def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream")
async def ask_stream(payload: AskRequest):
    async def event_generator():
        doc_filter = None
        config = {"configurable": {"thread_id": payload.conversation_id}}
        try:
            async for mode, chunk in agent_graph.astream(
                {"question": payload.question},
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if mode == "updates":
                    for node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        if node_output.get("document_filter") is not None:
                            doc_filter = node_output["document_filter"]
                        info = {
                            k: v for k, v in node_output.items()
                            if k in ("document_filter", "retrieval_grade", "top_similarity", "standalone_question")
                        }
                        yield _sse("step", {"node": node_name, "info": info})

                elif mode == "messages":
                    message, metadata = chunk
                    if metadata.get("langgraph_node") != "synthesize":
                        continue
                    token = getattr(message, "content", "")
                    if token:
                        yield _sse("token", {"text": token})

            yield _sse("done", {"document_filter": doc_filter})

        except Exception as exc:
            logger.exception("Stream failed")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )