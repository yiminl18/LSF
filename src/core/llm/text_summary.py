"""
Text Summary Utility

Provides simple truncation-based text summarization for generating short header summaries.

Main functions:
- summarize_text(): Truncate long text to a specified length
"""

from typing import Optional


def summarize_text(text: str, max_length: int = 100) -> Optional[str]:
    """
    Generate a text summary by truncation.

    Truncates the input text to the specified max length, preferring word boundaries.

    Args:
        text: The text to summarize
        max_length: Maximum summary length (default: 100 characters)

    Returns:
        The truncated summary text, or None if input is empty
    """
    if not text or not text.strip():
        return None

    # Simple truncation - take first max_length characters
    if len(text) <= max_length:
        return text.strip()

    # Truncate and add ellipsis
    summary = text[:max_length].strip()
    # Try to break at word boundary
    if len(text) > max_length:
        last_space = summary.rfind(" ")
        if (
            last_space > max_length * 0.7
        ):  # Only break at word if we keep at least 70% of max_length
            summary = summary[:last_space]
        summary += "..."

    return summary
