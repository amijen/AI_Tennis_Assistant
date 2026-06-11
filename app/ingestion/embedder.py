"""
Text embedder using HuggingFace's BAAI/bge-base-en-v1.5.

This model:
- Produces 768-dimensional embeddings 
- Excellent quality (MTEB ~63)
- Runs locally on CPU
"""

from sentence_transformers import SentenceTransformer

import os 
import dotenv
dotenv.load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
print(f"Loading embedding model: {MODEL_NAME}")
_model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded (dimension: {_model.get_embedding_dimension()})")

def embed_text(text: str) -> list[float]:
    """
    Convert a single text into a 768-dimensional vector.

    Args:
        text: The text to embed

    Returns:
        A list of 768 floats representing the embedding.
    """

    if not text or len(text)==0:
        raise ValueError("Cannot embed empty text.")
    
    # The model returns a numpy array -> convert to list for pgvector compatability
    embedding = _model.encode(text, normalize_embeddings = True)
    return embedding.tolist()

def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Batch embed multiple texts (much faster than calling embed_text in a loop).
    
    Args:
        texts: List of texts to embed.
        batch_size: How many texts to process at once.
    
    Returns:
        List of embeddings (each is a 768-dim vector).
    """

    if not texts:
        return []
    
    # Filter out empty texts
    valid_texts = [t for t in texts if t and t.strip()]

    embeddings = _model.encode(
        valid_texts, 
        batch_size = batch_size,
        normalize_embeddings = True,
        show_progress_bar = False
    )

    return [emb.tolist() for emb in embeddings]