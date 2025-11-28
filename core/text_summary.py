from typing import Optional


def summarize_text(text: str, max_length: int = 100) -> Optional[str]:
    """Summarize text by truncating to max_length characters.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Summarized text (truncated), or None if text is empty
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
        last_space = summary.rfind(' ')
        if last_space > max_length * 0.7:  # Only break at word if we keep at least 70% of max_length
            summary = summary[:last_space]
        summary += "..."
    
    return summary

