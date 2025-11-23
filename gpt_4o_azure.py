import os, math 
from openai import AzureOpenAI
import tiktoken
from typing import Optional

deployment = "gpt-4o"

# Global variable for cumulative cost tracking
_cumulative_cost_usd = 0.0
_price_per_million_input_tokens = 2.5

def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate the number of tokens in the given text.
    
    Args:
        text: The text to estimate tokens for
        model: The model name to use for tokenization
        
    Returns:
        The estimated number of tokens
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def gpt_4o_azure(prompt: str, 
                 key_path: str = '/Users/evier/Documents/gpt-4o.txt',
                 max_tokens: int = 800,
                 temperature: float = 0,
                 estimate_cost: bool = True) -> str:
    """
    Get response from Azure OpenAI API with cost estimation.
    
    Args:
        prompt: The text prompt to send to the model
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
        max_tokens: Maximum tokens for response
        temperature: Response randomness (0-1)
        estimate_cost: Whether to estimate and print cost, default True
        
    Returns:
        The response content from the model
    """
    global _cumulative_cost_usd
    
    # Read API configuration from file
    api_key = None
    api_version = None
    azure_endpoint = None
    
    with open(key_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('api_key:'):
                api_key = line.split(':', 1)[1].strip()
            elif line.startswith('api_version:'):
                api_version = line.split(':', 1)[1].strip()
            elif line.startswith('azure_endpoint='):
                azure_endpoint = line.split('=', 1)[1].strip()
    
    # Validate that all required values were found
    if not api_key:
        raise ValueError("api_key not found in configuration file")
    if not api_version:
        raise ValueError("api_version not found in configuration file")
    if not azure_endpoint:
        raise ValueError("azure_endpoint not found in configuration file")
    
    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    
    # Estimate input tokens and cost
    input_tokens = estimate_tokens(prompt, model=deployment)
    input_cost_usd = input_tokens * (_price_per_million_input_tokens / 1_000_000)
    
    # Generate response
    response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=5000,
    temperature=temperature,
    top_p=0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
    )

    answer = response.choices[0].message.content
    
    # Estimate output tokens and cost
    output_tokens = estimate_tokens(answer, model=deployment)
    output_cost_usd = output_tokens * (_price_per_million_input_tokens / 1_000_000)
    
    total_cost_usd = input_cost_usd + output_cost_usd
    _cumulative_cost_usd += total_cost_usd
    
    # Print cost estimation if enabled
    if estimate_cost:
        print(f"[COST] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Cost: ${total_cost_usd:.6f}")
        print(f"[COST_SUM] Cumulative cost: ${_cumulative_cost_usd:.6f}")
    
    return answer


def reset_cost_counter():
    """Reset the cumulative cost counter."""
    global _cumulative_cost_usd
    _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    """Get the cumulative cost in USD.
    
    Returns:
        The cumulative cost in USD
    """
    return _cumulative_cost_usd