"""
LLM 错误分类模块

统一定义 LLM 调用相关的异常类型和错误分类函数。
所有 provider 共享相同的错误处理策略。
"""

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

# 可重试的 openai SDK 异常类型
_RETRYABLE_TYPES = (
    RateLimitError,       # 429
    APIConnectionError,   # 含子类 APITimeoutError
    APITimeoutError,      # 超时
    InternalServerError,  # 500/502/503
)

# content filter 检测关键字（复用 judge_header 中的检测逻辑）
_CONTENT_FILTER_MARKERS = [
    "content_filter",
    "content management policy",
    "jailbreak",
    "responsible ai policy",
]


class ContentFilterError(Exception):
    """内容过滤错误：当 LLM API 检测到 jailbreak 或内容策略违规时抛出。"""
    pass


def is_content_filter(exc: BaseException) -> bool:
    """检测异常是否为内容过滤/jailbreak 错误。"""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTENT_FILTER_MARKERS)


def is_retryable(exc: BaseException) -> bool:
    """检测异常是否为可重试的瞬态错误（rate limit / connection / timeout / 5xx）。"""
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    return False
