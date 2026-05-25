import ollama 
from typing import List 

def embed_text(text):
    try:
        response = ollama.embeddings(
            model = "nomic-embed-text",
            prompt = text
        )
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")
    
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch embed for future optimization
    """
    return [embed_text(t) for t in texts]