"""
文本摘要工具

提供简单的文本截断摘要功能，用于生成标题的简短摘要。

主要功能：
- summarize_text(): 将长文本截断为指定长度的摘要
"""

from typing import Optional


def summarize_text(text: str, max_length: int = 100) -> Optional[str]:
    """
    通过截断生成文本摘要。

    将输入文本截断到指定的最大长度，尽量在单词边界处截断。

    参数:
        text: 要摘要的文本
        max_length: 摘要的最大长度（默认100字符）

    返回:
        截断后的摘要文本，如果输入为空则返回 None
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
