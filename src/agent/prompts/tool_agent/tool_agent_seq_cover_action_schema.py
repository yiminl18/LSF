"""Structured-output schema for the sequential-covering tool-agent."""

from __future__ import annotations

from typing import Any


def _build_seq_cover_action_schema() -> dict[str, Any]:
    """Return the action schema for a single-rule sequential-cover iteration."""
    return {
        "name": "seq_cover_agent_action",
        "description": "Sequential-covering tool call or one-rule generation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["tool", "generate"]},
                "reasoning": {"type": "string"},
                "tool": {"type": ["string", "null"]},
                "args": {"type": ["string", "null"]},
                "rule": {"type": ["string", "null"]},
            },
            "required": ["action", "reasoning", "tool", "args", "rule"],
        },
    }


__all__ = ["_build_seq_cover_action_schema"]
