"""Structured-output schema for tool-agent code mode."""

from __future__ import annotations

from typing import Any


def _build_code_action_schema() -> dict[str, Any]:
    return {
        "name": "agent_action",
        "description": "Agent tool call or single CodeRule generation",
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


__all__ = ["_build_code_action_schema"]
