"""
通用 token 估算工具，基于 tiktoken。
与具体 LLM 提供商解耦。
"""

import tiktoken


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """估算文本的 token 数量，model 用于选择 tiktoken 编码器（cl100k_base 兜底）。"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
