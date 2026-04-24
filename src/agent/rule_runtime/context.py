"""LLM context-window guards shared by rule-generation agents."""

from __future__ import annotations

from core.llm.gpt_54_azure import estimate_tokens as estimate_context_tokens


# Conservative context limits:
# - GPT-5.4: 272K window from the OpenAI release page
# - GPT-5.4-mini: 400K window from the OpenRouter model page
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-5.4": 272_000,
    "gpt-5.4-mini": 400_000,
    "openai/gpt-5.4": 1_050_000,
    "openai/gpt-5.4-mini": 400_000,
}


def _resolve_effective_model_identity(
    llm_provider: str,
    llm_model: str,
) -> str:
    del llm_provider
    return llm_model.strip().lower()


def _resolve_model_context_limit(
    llm_provider: str,
    llm_model: str,
) -> tuple[str, int | None]:
    model_identity = _resolve_effective_model_identity(llm_provider, llm_model)
    limit = _MODEL_CONTEXT_LIMITS.get(model_identity)
    if limit is not None:
        return model_identity, limit

    model_suffix = model_identity.split("/", 1)[-1]
    return model_identity, _MODEL_CONTEXT_LIMITS.get(model_suffix)


def _estimate_tokens_for_model_context(
    text: str,
    llm_provider: str,
    llm_model: str,
) -> int:
    model_identity = _resolve_effective_model_identity(llm_provider, llm_model)
    model_suffix = model_identity.split("/", 1)[-1]
    return estimate_context_tokens(text, model=model_suffix)


def _ensure_request_within_model_context(
    *,
    prompt_text: str,
    max_output_tokens: int,
    llm_provider: str,
    llm_model: str,
    stage_label: str,
) -> None:
    model_identity, context_limit = _resolve_model_context_limit(
        llm_provider,
        llm_model,
    )
    if context_limit is None:
        return

    prompt_tokens = _estimate_tokens_for_model_context(
        prompt_text,
        llm_provider,
        llm_model,
    )
    total_request_tokens = prompt_tokens + max_output_tokens
    if total_request_tokens > context_limit:
        raise RuntimeError(
            f"{stage_label} request exceeds model context limit: "
            f"model={model_identity} prompt_tokens={prompt_tokens} "
            f"max_output_tokens={max_output_tokens} total={total_request_tokens} > {context_limit}"
        )
