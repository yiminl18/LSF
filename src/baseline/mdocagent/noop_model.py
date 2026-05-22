"""Zero-cost placeholder model for agents that should not call an LLM.

Used by image_agent in the LSF MDocAgent integration: with retrieval running
over per-page text only (ColBERT; ColPali bypassed), the image agent has no
visual-modality signal to add — its inputs are the same pages the text agent
already saw, just rendered as PNGs. Routing it through this NoOp avoids paying
for a redundant vision LLM call while preserving the upstream orchestrator's
hardcoded ``self.agents[0]`` / ``self.agents[1]`` / ``self.agents[-1]`` indices.

Contract: same shape as ``MyOpenAI.predict`` so upstream's ``MultiAgentSystem``
can drop it in without changes — returns ``(answer_str, messages_list)``.
"""

from __future__ import annotations

from typing import Any

from models.base_model import BaseModel  # type: ignore[import]


class NoOpModel(BaseModel):
    """Returns an empty answer instantly without invoking any LLM backend."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def predict(
        self,
        question: str,
        texts: list[str] | None = None,
        images: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        return "", list(history) if history else []

    def is_valid_history(self, history: Any) -> bool:
        return isinstance(history, list)
