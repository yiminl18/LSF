"""
统一模型 API 封装

提供统一的模型调用接口，根据 provider 选择对应的实现。
包含统一的 retry + 错误分类机制，所有 provider 行为一致。
"""

import time
from typing import Any, Set

from core.llm.errors import ContentFilterError, is_content_filter, is_retryable
from core.llm.gpt_54_azure import gpt_54_azure
from core.llm.gpt_54_azure import reset_cost_counter as _azure54_reset
from core.llm.gpt_54_azure import get_cumulative_cost as _azure54_get_cost
from core.llm.gpt_54mini_azure import gpt_54mini_azure
from core.llm.gpt_54mini_azure import reset_cost_counter as _azure54mini_reset
from core.llm.gpt_54mini_azure import get_cumulative_cost as _azure54mini_get_cost
from core.llm.openrouter import openrouter_chat
from core.llm.openrouter import reset_cost_counter as _openrouter_reset
from core.llm.openrouter import get_cumulative_cost as _openrouter_get_cost

LLM_PROVIDERS: Set[str] = {"azure", "openrouter"}

# 重试参数
_MAX_RETRIES = 3
_BASE_WAIT_S = 5  # 指数退避: 5s, 10s, 20s


def _validate_provider(llm_provider: str) -> None:
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown llm_provider={llm_provider!r}, expected one of {sorted(LLM_PROVIDERS)}"
        )


def _require_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM model must be specified explicitly")
    normalized_model = model.strip()
    if normalized_model == "unspec" + "ified":
        raise ValueError("LLM model uses a reserved invalid name")
    return normalized_model


def _dispatch(
    prompt: str,
    llm_provider: str,
    max_tokens: int,
    *,
    model: str,
    response_schema: dict[str, Any] | None = None,
    temperature: float = 0,
    **kwargs,
) -> str:
    """路由到具体 provider（不含错误处理）。"""
    resolved_model = _require_model(model)
    if llm_provider == "azure":
        if resolved_model.startswith("gpt-5.4-mini"):
            return gpt_54mini_azure(
                prompt,
                max_tokens=max_tokens,
                model=resolved_model,
                response_schema=response_schema,
                temperature=temperature,
                **kwargs,
            )
        if resolved_model.startswith("gpt-5.4"):
            return gpt_54_azure(
                prompt,
                max_tokens=max_tokens,
                model=resolved_model,
                response_schema=response_schema,
                temperature=temperature,
                **kwargs,
            )
        raise ValueError(
            f"Unsupported azure model={resolved_model!r}; supported: gpt-5.4-mini, gpt-5.4"
        )
    if llm_provider == "openrouter":
        return openrouter_chat(
            prompt,
            max_tokens=max_tokens,
            model=resolved_model,
            response_schema=response_schema,
            temperature=temperature,
            **kwargs,
        )
    raise ValueError(
        f"Unknown llm_provider={llm_provider!r}, expected one of {sorted(LLM_PROVIDERS)}"
    )


def llm_call(
    prompt: str,
    llm_provider: str = "azure",
    max_tokens: int = 800,
    *,
    model: str,
    response_schema: dict[str, Any] | None = None,
    temperature: float = 0,
    **kwargs,
) -> str:
    """
    统一 LLM 调用，包含自动重试和错误分类。

    重试策略: 最多 3 次，指数退避 (5s, 10s, 20s)
    - 可重试: RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
    - 不可重试: AuthenticationError, BadRequestError, PermissionDeniedError 等
    - Content filter: 包装为 ContentFilterError 后立即抛出（不重试）

    参数:
        prompt: 提示文本
        llm_provider: "azure" / "openrouter"
        max_tokens: 最大输出 token 数
        model: 模型标识，必须显式指定
        response_schema: 可选 structured-output schema；openrouter 取决于模型能力
        temperature: 采样温度
        **kwargs: 转发给 provider 的额外参数

    返回:
        模型响应文本
    """
    _validate_provider(llm_provider)
    resolved_model = _require_model(model)

    for attempt in range(_MAX_RETRIES):
        try:
            return _dispatch(
                prompt,
                llm_provider,
                max_tokens,
                model=resolved_model,
                response_schema=response_schema,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            # Content filter → 不重试，立即抛出
            if is_content_filter(e):
                raise ContentFilterError(
                    f"Content filter triggered ({llm_provider}): {str(e)[:500]}"
                ) from e

            # 不可重试 → 立即抛出
            if not is_retryable(e):
                raise

            # 可重试 + 还有重试次数
            if attempt < _MAX_RETRIES - 1:
                wait = 2**attempt * _BASE_WAIT_S
                print(
                    f"[LLM_RETRY] {type(e).__name__} "
                    f"attempt {attempt + 1}/{_MAX_RETRIES}, "
                    f"waiting {wait}s... ({llm_provider})"
                )
                time.sleep(wait)
            else:
                # 重试耗尽，抛出原始异常
                raise

    # 理论上不可达，但保持类型安全
    raise RuntimeError("llm_call retry loop exited unexpectedly")


def reset_llm_cost(llm_provider: str = "azure") -> None:
    """重置指定 provider 的成本计数器。"""
    _validate_provider(llm_provider)
    if llm_provider == "azure":
        _azure54_reset()
        _azure54mini_reset()
        return
    _openrouter_reset()


def get_llm_cost(llm_provider: str = "azure") -> float:
    """获取指定 provider 的累计成本。"""
    _validate_provider(llm_provider)
    if llm_provider == "azure":
        return _azure54_get_cost() + _azure54mini_get_cost()
    return _openrouter_get_cost()
