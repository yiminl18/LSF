from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from typing import Optional

# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_PATH = PROJECT_ROOT.parent / "api_keys" / "azure_cloudbank" / "embedding.txt"

# Default Azure OpenAI configuration
DEFAULT_API_VERSION = "2025-01-01-preview"
DEFAULT_AZURE_ENDPOINT = "https://east-docetl.openai.azure.com/"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small-3"


def get_embedding(text: str,
                  key_path: str = str(DEFAULT_KEY_PATH),
                  model: str = DEFAULT_EMBEDDING_MODEL,
                  cache: bool = True,
                  cache_dir: Optional[str] = None,
                  azure_endpoint: str = DEFAULT_AZURE_ENDPOINT,
                  api_version: str = DEFAULT_API_VERSION) -> list:
    """
    Get embedding vector for the given text (no per-text caching).
    
    Args:
        text: Text to get embedding for
        key_path: Path to the API key file (default: '/Users/evier/Documents/embedding_key.txt')
        model: Embedding model name
        cache: Deprecated; kept for backwards compatibility. Ignored.
        cache_dir: Deprecated; kept for backwards compatibility. Ignored.
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version to use
    
    Returns:
        Embedding vector as a list
    """
    text = text.replace("\n", " ").strip()
    
    # If text is empty, return zero vector (will result in similarity of 0)
    if not text:
        # Return a zero vector (embedding dimension is typically 1536 for text-embedding-3-small)
        return [0.0] * 1536
    
    # Call API to get embedding
    try:
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Warning: Failed to get embedding for text: {text[:50]}... Error: {e}")
        # Return zero vector as fallback
        return [0.0] * 1536
    
    # API returns a list-like; ensure plain list for downstream serialization
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    elif not isinstance(embedding, list):
        embedding = list(embedding)
    
    return embedding


def cosine_sim(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors (manual calculation to avoid numerical issues).
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score (float between 0 and 1)
    """
    import numpy as np
    
    # Convert to numpy arrays
    v1 = np.array(vec1, dtype=np.float64)
    v2 = np.array(vec2, dtype=np.float64)
    
    # Check if either vector is zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        # If either vector is zero, return 0.0
        return 0.0
    
    # Check for NaN or Inf values
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return 0.0
    if np.any(np.isinf(v1)) or np.any(np.isinf(v2)):
        return 0.0
    
    # Manually calculate cosine similarity: dot(v1, v2) / (norm1 * norm2)
    dot_product = np.dot(v1, v2)
    
    # Check if dot_product is NaN or Inf
    if np.isnan(dot_product) or np.isinf(dot_product):
        return 0.0
    
    # Calculate similarity
    similarity = dot_product / (norm1 * norm2)
    
    # Ensure result is in [-1, 1] range (may exceed due to floating point errors)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    # Check if final result is NaN or Inf
    if np.isnan(similarity) or np.isinf(similarity):
        return 0.0
    
    return float(similarity)


def calculate_similarity(text1: str,
                        text2: str,
                        key_path: str = str(DEFAULT_KEY_PATH),
                        cache: bool = True,
                        cache_dir: Optional[str] = None,
                        model: str = DEFAULT_EMBEDDING_MODEL,
                        azure_endpoint: str = DEFAULT_AZURE_ENDPOINT,
                        api_version: str = DEFAULT_API_VERSION) -> float:
    """
    Calculate embedding similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        key_path: Path to the API key file (default: '/Users/evier/Documents/embedding_key.txt')
        cache: Deprecated; kept for backwards compatibility. Ignored.
        cache_dir: Deprecated; kept for backwards compatibility. Ignored.
        model: Embedding model name
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version to use
    
    Returns:
        Similarity score (float between 0 and 1)
    """
    # Get embeddings for both texts
    embedding1 = get_embedding(text1, key_path=key_path, model=model, cache=cache,
                               cache_dir=cache_dir, azure_endpoint=azure_endpoint,
                               api_version=api_version)
    embedding2 = get_embedding(text2, key_path=key_path, model=model, cache=cache,
                               cache_dir=cache_dir, azure_endpoint=azure_endpoint,
                               api_version=api_version)
    
    # Calculate similarity
    similarity = cosine_sim(embedding1, embedding2)
    
    return similarity


if __name__ == "__main__":
    # Example usage
    text1 = "Hugging Face has emerged as a prominent and innovative force in NLP."
    text2 = "Hugging Face is a leading company in natural language processing."
    
    similarity = calculate_similarity(text1, text2)
    print(f"Similarity: {similarity:.4f}")

