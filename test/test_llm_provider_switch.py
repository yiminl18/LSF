from __future__ import annotations

import importlib

import pytest


model_module = importlib.import_module("core.llm.model")


def test_llm_call_requires_explicit_models(monkeypatch):
    with pytest.raises(TypeError, match="model"):
        model_module.llm_call("p", llm_provider="azure", max_tokens=11)


def test_llm_call_routes_explicit_models(monkeypatch):
    seen: dict[str, tuple[str | None, int]] = {}

    def fake_gpt54mini(prompt, max_tokens, model="gpt-5.4-mini", **kwargs):
        seen["azure"] = (model, max_tokens)
        return "azure-ok"

    def fake_openrouter(prompt, max_tokens, model="z-ai/glm-5.1", **kwargs):
        seen["openrouter"] = (model, max_tokens)
        return "openrouter-ok"

    monkeypatch.setattr(model_module, "gpt_54mini_azure", fake_gpt54mini)
    monkeypatch.setattr(model_module, "openrouter_chat", fake_openrouter)

    assert (
        model_module.llm_call(
            "p", llm_provider="azure", model="gpt-5.4-mini", max_tokens=11
        )
        == "azure-ok"
    )
    assert (
        model_module.llm_call(
            "p", llm_provider="openrouter", model="z-ai/glm-5.1", max_tokens=13
        )
        == "openrouter-ok"
    )
    assert seen == {
        "azure": ("gpt-5.4-mini", 11),
        "openrouter": ("z-ai/glm-5.1", 13),
    }


def test_llm_call_preserves_explicit_openrouter_glm_model(monkeypatch):
    seen: dict[str, str] = {}

    def fake_openrouter(prompt, max_tokens, model="z-ai/glm-5.1", **kwargs):
        seen["model"] = model
        return "glm-ok"

    monkeypatch.setattr(model_module, "openrouter_chat", fake_openrouter)

    assert (
        model_module.llm_call(
            "p",
            llm_provider="openrouter",
            model="z-ai/glm-5.1",
            max_tokens=21,
        )
        == "glm-ok"
    )
    assert seen == {"model": "z-ai/glm-5.1"}


def test_llm_call_routes_explicit_azure_gpt54_models(monkeypatch):
    seen: dict[str, tuple[str, int]] = {}

    def fake_gpt54(prompt, max_tokens, model="gpt-5.4", **kwargs):
        seen["gpt-5.4"] = (model, max_tokens)
        return "gpt54-ok"

    def fake_gpt54mini(prompt, max_tokens, model="gpt-5.4-mini", **kwargs):
        seen["gpt-5.4-mini"] = (model, max_tokens)
        return "gpt54mini-ok"

    monkeypatch.setattr(model_module, "gpt_54_azure", fake_gpt54)
    monkeypatch.setattr(model_module, "gpt_54mini_azure", fake_gpt54mini)

    assert (
        model_module.llm_call("p", llm_provider="azure", model="gpt-5.4", max_tokens=21)
        == "gpt54-ok"
    )
    assert (
        model_module.llm_call(
            "p", llm_provider="azure", model="gpt-5.4-mini", max_tokens=22
        )
        == "gpt54mini-ok"
    )
    assert seen == {
        "gpt-5.4": ("gpt-5.4", 21),
        "gpt-5.4-mini": ("gpt-5.4-mini", 22),
    }


def test_llm_call_routes_versioned_azure_gpt54_model_ids(monkeypatch):
    seen: dict[str, str] = {}

    def fake_gpt54(prompt, max_tokens, model="gpt-5.4", **kwargs):
        seen["gpt-5.4"] = model
        return "gpt54-ok"

    def fake_gpt54mini(prompt, max_tokens, model="gpt-5.4-mini", **kwargs):
        seen["gpt-5.4-mini"] = model
        return "gpt54mini-ok"

    monkeypatch.setattr(model_module, "gpt_54_azure", fake_gpt54)
    monkeypatch.setattr(model_module, "gpt_54mini_azure", fake_gpt54mini)

    assert (
        model_module.llm_call(
            "p",
            llm_provider="azure",
            model="gpt-5.4-2026-04-14",
            max_tokens=23,
        )
        == "gpt54-ok"
    )
    assert (
        model_module.llm_call(
            "p",
            llm_provider="azure",
            model="gpt-5.4-mini-2026-04-14",
            max_tokens=24,
        )
        == "gpt54mini-ok"
    )
    assert seen == {
        "gpt-5.4": "gpt-5.4-2026-04-14",
        "gpt-5.4-mini": "gpt-5.4-mini-2026-04-14",
    }


def test_azure_cost_hooks_aggregate_all_azure_wrappers(monkeypatch):
    seen_resets: list[str] = []

    monkeypatch.setattr(
        model_module, "_azure54_reset", lambda: seen_resets.append("54")
    )
    monkeypatch.setattr(
        model_module, "_azure54mini_reset", lambda: seen_resets.append("54mini")
    )
    monkeypatch.setattr(model_module, "_azure54_get_cost", lambda: 2.5)
    monkeypatch.setattr(model_module, "_azure54mini_get_cost", lambda: 0.75)

    model_module.reset_llm_cost("azure")

    assert "54" in seen_resets
    assert "54mini" in seen_resets
    assert model_module.get_llm_cost("azure") == pytest.approx(2.5 + 0.75)


def test_llm_call_rejects_unknown_provider_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "openrouter_chat",
        lambda *args, **kwargs: pytest.fail("dispatch should not run"),
    )

    with pytest.raises(ValueError, match="Unknown llm_provider"):
        model_module.llm_call("p", llm_provider="glm", model="z-ai/glm-5.1")
