"""
Azure GPT-4o API Wrapper

Provides a wrapper for Azure OpenAI GPT-4o model calls with cost tracking.

Main functions:
- gpt_4o_azure(): Call GPT-4o model and generate a response
- estimate_tokens(): Estimate the token count of a text
- reset_cost_counter(): Reset the cumulative cost counter
- get_cumulative_cost(): Get the cumulative cost

Dependencies:
- openai: Azure OpenAI SDK
- tiktoken: OpenAI token counting library

Environment variables:
- AZURE_API_KEY: Azure OpenAI API key
- AZURE_API_BASE: Azure OpenAI endpoint URL
- AZURE_API_VERSION: API version
"""

import os
from openai import AzureOpenAI
import tiktoken
import threading
from core.config import GPT_PRICE_PER_MILLION_INPUT, GPT_PRICE_PER_MILLION_OUTPUT

deployment = "gpt-4o"

# Global mutable state: cumulative cost tracking (process-level, thread-safe)
# Access via reset_cost_counter() / get_cumulative_cost()
_cumulative_cost_usd = 0.0
_cost_lock = threading.Lock()  # Lock protecting cost accumulation


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate the token count for a given text.

    Uses tiktoken to compute the number of tokens for the specified model.

    Args:
        text: The text to estimate
        model: Tokenizer model name (default: gpt-4o)

    Returns:
        Estimated token count
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def gpt_4o_azure(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0,
    estimate_cost: bool = True,
    model: str = "gpt-4o",
) -> str:
    """
    Call the Azure OpenAI GPT-4o API and estimate cost.

    Sends a prompt to the GPT-4o model, retrieves the response, and tracks API call cost.

    Args:
        prompt: Prompt text to send to the model
        max_tokens: Max response tokens (default: 800)
        temperature: Response randomness 0-1 (default: 0)
        estimate_cost: Whether to estimate and print cost (default: True)
        model: Deployment model name (default: gpt-4o, also supports gpt-4o-mini)

    Returns:
        The model's response content
    """
    global _cumulative_cost_usd

    # Pricing (per 1M tokens)
    if "mini" in model:
        price_input = 0.15
        price_output = 0.60
    else:
        price_input = GPT_PRICE_PER_MILLION_INPUT
        price_output = GPT_PRICE_PER_MILLION_OUTPUT

    # Read API configuration from environment variables
    api_key = os.environ.get("AZURE_API_KEY")
    azure_endpoint = os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_API_VERSION")

    # Validate that all required values were found
    if not all([api_key, azure_endpoint, api_version]):
        raise ValueError(
            "Missing required environment variables. Please set: "
            "AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION"
        )

    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    # Estimate input tokens and cost
    input_tokens = estimate_tokens(prompt, model=model)
    input_cost_usd = input_tokens * (price_input / 1_000_000)

    # Generate response
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,  # Use standard parameter name or max_completion_tokens based on SDK version
            temperature=temperature,
            top_p=0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=model,
        )
        answer = response.choices[0].message.content
    except Exception:
        # Re-raise for upstream handling (judge_header catches content filter errors)
        raise

    # Estimate output tokens and cost
    output_tokens = estimate_tokens(answer, model=model)
    output_cost_usd = output_tokens * (price_output / 1_000_000)

    total_cost_usd = input_cost_usd + output_cost_usd
    # Thread-safe cost accumulation
    with _cost_lock:
        _cumulative_cost_usd += total_cost_usd
        current_cumulative = _cumulative_cost_usd

    # Print cost estimation if enabled
    if estimate_cost:
        print(
            f"[COST] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Cost: ${total_cost_usd:.6f}"
        )
        print(f"[COST_SUM] Cumulative cost: ${current_cumulative:.6f}")

    return answer


def reset_cost_counter():
    """
    Reset the cumulative cost counter.

    Resets the global cumulative cost variable to 0.
    """
    global _cumulative_cost_usd
    with _cost_lock:
        _cumulative_cost_usd = 0.0


def get_cumulative_cost() -> float:
    """
    Get the cumulative cost in USD.

    Returns:
        Cumulative cost in USD
    """
    with _cost_lock:
        return _cumulative_cost_usd
