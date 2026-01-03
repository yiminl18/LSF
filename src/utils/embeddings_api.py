import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from typing import List, Optional


def get_embedding_client(key_path: str = '/Users/evier/Documents/embedding_key.txt',
                         azure_endpoint: str = "https://east-docetl.openai.azure.com/",
                         api_version: str = "2025-01-01-preview") -> AzureOpenAI:
    """
    Initialize and return an Azure OpenAI client for embeddings.
    
    Args:
        key_path: Path to the API key file (default: '/Users/evier/Documents/embedding_key.txt')
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version to use
        
    Returns:
        Initialized AzureOpenAI client
    """
    with open(key_path, "r") as f:
        api_key = f.read().strip()
    
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    return client


def create_embeddings(texts: List[str],
                     key_path: str = '/Users/evier/Documents/embedding_key.txt',
                     deployment: str = "text-embedding-3-small-3",
                     azure_endpoint: str = "https://east-docetl.openai.azure.com/",
                     api_version: str = "2025-01-01-preview"):
    """
    Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings to create embeddings for
        key_path: Path to the API key file (default: '/Users/evier/Documents/embedding_key.txt')
        deployment: Deployment name for the embedding model
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: API version to use
        
    Returns:
        Response object containing embeddings
    """
    client = get_embedding_client(key_path, azure_endpoint, api_version)
    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
    return response


if __name__ == "__main__":
    # Example usage
    texts = ["first phrase", "second phrase", "third phrase"]
    response = create_embeddings(texts)
    
    for item in response.data:
        length = len(item.embedding)
        print(
            f'data[{item.index}]: length={length}, '
            f'[{item.embedding[0]}, {item.embedding[1]}, '
            f'..., {item.embedding[length-2]}, {item.embedding[length-1]}]'
        )
    print(response.usage)