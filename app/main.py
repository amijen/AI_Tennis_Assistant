import logging
from pathlib import Path                         
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles       
from fastapi.responses import FileResponse        

from app.api import router
from app.config import settings

# ── Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if settings.is_dev else logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Tennis AI Assistant...")
    from app.ingestion.embedder import embed_text
    embed_text("warmup")
    logger.info("Embedding model warm")
    from app.graph.builder import agent_graph  # noqa: F401
    logger.info("Agent graph ready")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Tennis AI Assistant",
    description="Ask questions about ITF and Grand Slam tennis rules.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── Serve Frontend ────────────────────────────────
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/")
def root():
    return FileResponse(FRONTEND / "index.html")