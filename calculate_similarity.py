import os
import hashlib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from typing import Optional

# Default Azure OpenAI configuration
DEFAULT_API_VERSION = "2025-01-01-preview"
DEFAULT_AZURE_ENDPOINT = "https://east-docetl.openai.azure.com/"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small-3"


def get_text_hash(text: str) -> str:
    """Generate MD5 hash of text for use as filename.
    
    Args:
        text: Text to hash
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_embedding(text: str,
                  key_path: str = '/Users/evier/Documents/embedding_key.txt',
                  model: str = DEFAULT_EMBEDDING_MODEL,
                  cache: bool = True,
                  cache_dir: Optional[str] = None,
                  azure_endpoint: str = DEFAULT_AZURE_ENDPOINT,
                  api_version: str = DEFAULT_API_VERSION) -> list:
    """
    Get embedding vector for the given text.
    
    Args:
        text: Text to get embedding for
        key_path: Path to the API key file (default: '/Users/evier/Documents/embedding_key.txt')
        model: Embedding model name
        cache: Whether to use cache, default True
        cache_dir: Directory for caching embeddings (default: 'embedding')
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
    
    # Initialize cache directory
    if cache_dir is None:
        cache_dir = "embedding"
    embedding_dir = Path(cache_dir)
    embedding_dir.mkdir(exist_ok=True)
    
    # If caching is enabled, check if embedding already exists
    if cache:
        text_hash = get_text_hash(text)
        embedding_file = embedding_dir / f"{text_hash}.npy"
        
        if embedding_file.exists():
            import numpy as np
            try:
                embedding = np.load(embedding_file, allow_pickle=True)
                # Check if it's a valid array
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    return embedding.tolist()
                else:
                    # If file is corrupted, delete it and re-fetch
                    print(f"Warning: Corrupted cache file detected, removing: {embedding_file}")
                    embedding_file.unlink()
            except (EOFError, ValueError, OSError) as e:
                # If file is corrupted, delete it and re-fetch
                print(f"Warning: Failed to load cache file {embedding_file}: {e}, removing it")
                try:
                    embedding_file.unlink()
                except:
                    pass
    
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
    
    # If caching is enabled, save embedding
    if cache:
        import numpy as np
        np.save(embedding_file, embedding)
    
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
                        key_path: str = '/Users/evier/Documents/embedding_key.txt',
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
        cache: Whether to use cache, default True
        cache_dir: Directory for caching embeddings (default: 'embedding')
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

