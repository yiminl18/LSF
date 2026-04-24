import importlib
import sqlite3

import pytest

from agent.rules.range_rule_json import build_range_rule_response_schema
from core.llm.model import llm_call
from core.pipeline.e2e_utils.cache import CachedLLMCaller

azure_54_module = importlib.import_module("core.llm.gpt_54_azure")
azure_54mini_module = importlib.import_module("core.llm.gpt_54mini_azure")
openrouter_module = importlib.import_module("core.llm.openrouter")


def _simple_schema() -> dict:
    return {
        "name": "simple_payload",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }


class _FakeMessage:
    def __init__(self, content: str | None, **extras) -> None:
        self.content = content
        for key, value in extras.items():
            setattr(self, key, value)


class _FakeChoice:
    def __init__(self, content: str | None, **extras) -> None:
        self.message = _FakeMessage(content, **extras)
        self.finish_reason = extras.get("finish_reason")


class _FakeResponse:
    def __init__(self, content: str | None, **extras) -> None:
        self.choices = [_FakeChoice(content, **extras)]


class _FakeTextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCompletions:
    def __init__(self, seen_kwargs: list[dict]) -> None:
        self._seen_kwargs = seen_kwargs

    def create(self, **kwargs):
        self._seen_kwargs.append(kwargs)
        return _FakeResponse('{"value":"ok"}')


class _StaticResponseCompletions:
    def __init__(self, response, seen_kwargs: list[dict] | None = None) -> None:
        self._response = response
        self._seen_kwargs = seen_kwargs

    def create(self, **kwargs):
        if self._seen_kwargs is not None:
            self._seen_kwargs.append(kwargs)
        return self._response


class _StaticResponseClient:
    def __init__(self, response, seen_kwargs: list[dict] | None = None) -> None:
        self.chat = type(
            "_StaticChat",
            (),
            {"completions": _StaticResponseCompletions(response, seen_kwargs)},
        )()


class _FakeChat:
    def __init__(self, seen_kwargs: list[dict]) -> None:
        self.completions = _FakeCompletions(seen_kwargs)


class _FakeClient:
    def __init__(self, seen_kwargs: list[dict]) -> None:
        self.chat = _FakeChat(seen_kwargs)


class TestStructuredOutputSchema:
    def test_build_range_rule_response_schema(self):
        schema = build_range_rule_response_schema()
        assert schema["name"] == "range_rule_bundle"
        assert schema["strict"] is True
        assert schema["schema"]["properties"]["rules"]["minItems"] == 1
        assert schema["schema"]["properties"]["rules"]["maxItems"] == 5
        spec_props = schema["schema"]["properties"]["rules"]["items"]["properties"][
            "retrieval_spec"
        ]["properties"]
        assert "page_idx" in spec_props
        assert "boundary_context_chars" in spec_props


class TestLLMCallStructuredDispatch:
    def test_llm_call_dispatches_response_schema(self, monkeypatch):
        seen = {}

        def fake_azure_mini(
            prompt,
            max_tokens,
            response_schema=None,
            temperature=0,
            model="gpt-5.4-mini",
            **kwargs,
        ):
            seen["azure"] = (response_schema, temperature, model)
            return "{}"

        def fake_openrouter(
            prompt,
            max_tokens,
            response_schema=None,
            temperature=0,
            model="z-ai/glm-5.1",
            **kwargs,
        ):
            seen["openrouter"] = (response_schema, temperature, model)
            return "{}"

        monkeypatch.setattr("core.llm.model.gpt_54mini_azure", fake_azure_mini)
        monkeypatch.setattr("core.llm.model.openrouter_chat", fake_openrouter)

        schema = _simple_schema()
        assert (
            llm_call(
                "p",
                llm_provider="azure",
                model="gpt-5.4-mini",
                response_schema=schema,
            )
            == "{}"
        )
        assert (
            llm_call(
                "p",
                llm_provider="openrouter",
                model="z-ai/glm-5.1",
                response_schema=schema,
            )
            == "{}"
        )
        assert seen["azure"] == (schema, 0, "gpt-5.4-mini")
        assert seen["openrouter"] == (schema, 0, "z-ai/glm-5.1")

    def test_llm_call_rejects_unsupported_provider_with_schema(self):
        schema = _simple_schema()
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            llm_call(
                "p",
                llm_provider="unsupported",
                model="unsupported-model",
                response_schema=schema,
            )

    def test_llm_call_dispatches_response_schema_to_azure_gpt54_family(
        self, monkeypatch
    ):
        seen = {}

        def fake_gpt54(
            prompt,
            max_tokens,
            response_schema=None,
            temperature=0,
            model="gpt-5.4",
            **kwargs,
        ):
            seen["gpt-5.4"] = (response_schema, temperature, model)
            return "{}"

        def fake_gpt54mini(
            prompt,
            max_tokens,
            response_schema=None,
            temperature=0,
            model="gpt-5.4-mini",
            **kwargs,
        ):
            seen["gpt-5.4-mini"] = (response_schema, temperature, model)
            return "{}"

        monkeypatch.setattr("core.llm.model.gpt_54_azure", fake_gpt54)
        monkeypatch.setattr("core.llm.model.gpt_54mini_azure", fake_gpt54mini)

        schema = _simple_schema()
        assert (
            llm_call(
                "p",
                llm_provider="azure",
                model="gpt-5.4",
                response_schema=schema,
            )
            == "{}"
        )
        assert (
            llm_call(
                "p",
                llm_provider="azure",
                model="gpt-5.4-mini",
                response_schema=schema,
            )
            == "{}"
        )
        assert seen == {
            "gpt-5.4": (schema, 0, "gpt-5.4"),
            "gpt-5.4-mini": (schema, 0, "gpt-5.4-mini"),
        }


class TestProviderStructuredResponseFormat:
    def test_gpt_54_azure_uses_deployment_and_max_completion_tokens(self, monkeypatch):
        seen_kwargs: list[dict] = []
        monkeypatch.setenv("AZURE_54_API_KEY", "test")
        monkeypatch.setenv("AZURE_54_API_BASE", "https://example54.openai.azure.com")
        monkeypatch.setenv("AZURE_54_API_VERSION", "2025-01-01-preview")
        monkeypatch.setenv("AZURE_54_DEPLOYMENT", "gpt-5.4-deployment")
        monkeypatch.setattr(
            azure_54_module, "AzureOpenAI", lambda **kwargs: _FakeClient(seen_kwargs)
        )

        schema = _simple_schema()
        result = azure_54_module.gpt_54_azure(
            "prompt", response_schema=schema, estimate_cost=False
        )

        assert result == '{"value":"ok"}'
        assert seen_kwargs == [
            {
                "messages": [{"role": "user", "content": "prompt"}],
                "max_completion_tokens": 800,
                "temperature": 0,
                "model": "gpt-5.4-deployment",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            }
        ]

    def test_gpt_54mini_azure_uses_deployment_and_max_completion_tokens(
        self, monkeypatch
    ):
        seen_kwargs: list[dict] = []
        monkeypatch.setenv("AZURE_54MINI_API_KEY", "test")
        monkeypatch.setenv(
            "AZURE_54MINI_API_BASE", "https://example54mini.openai.azure.com"
        )
        monkeypatch.setenv("AZURE_54MINI_API_VERSION", "2025-01-01-preview")
        monkeypatch.setenv("AZURE_54MINI_DEPLOYMENT", "gpt-5.4-mini-deployment")
        monkeypatch.setattr(
            azure_54mini_module,
            "run_azure_gpt54_family_request",
            lambda **kwargs: seen_kwargs.append(kwargs) or ('{"value":"ok"}', 10, 5),
        )

        schema = _simple_schema()
        result = azure_54mini_module.gpt_54mini_azure(
            "prompt", response_schema=schema, estimate_cost=False
        )

        assert result == '{"value":"ok"}'
        assert seen_kwargs == [
            {
                "prompt": "prompt",
                "max_tokens": 800,
                "temperature": 0,
                "model": "gpt-5.4-mini",
                "response_schema": schema,
                "env_prefix": "AZURE_54MINI",
            }
        ]

    def test_gpt_54_azure_accepts_object_content_parts(self, monkeypatch):
        monkeypatch.setenv("AZURE_54_API_KEY", "test")
        monkeypatch.setenv("AZURE_54_API_BASE", "https://example54.openai.azure.com")
        monkeypatch.setenv("AZURE_54_API_VERSION", "2025-01-01-preview")
        monkeypatch.setenv("AZURE_54_DEPLOYMENT", "gpt-5.4-deployment")
        content_part_response = _FakeResponse([_FakeTextPart('{"value":"ok"}')])
        monkeypatch.setattr(
            azure_54_module,
            "AzureOpenAI",
            lambda **kwargs: _StaticResponseClient(content_part_response),
        )

        result = azure_54_module.gpt_54_azure("prompt", estimate_cost=False)

        assert result == '{"value":"ok"}'

    def test_openrouter_schema_happy_path_for_supported_models(self, monkeypatch):
        seen_kwargs: list[dict] = []
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setattr(
            openrouter_module, "OpenAI", lambda **kwargs: _FakeClient(seen_kwargs)
        )

        schema = _simple_schema()
        for model in ("openai/gpt-5.4", "z-ai/glm-5.1"):
            result = openrouter_module.openrouter_chat(
                "prompt",
                model=model,
                response_schema=schema,
                estimate_cost=False,
            )

            assert result == '{"value":"ok"}'

        assert seen_kwargs == [
            {
                "messages": [{"role": "user", "content": "prompt"}],
                "max_tokens": 800,
                "temperature": 0,
                "model": "openai/gpt-5.4",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            },
            {
                "messages": [{"role": "user", "content": "prompt"}],
                "max_tokens": 800,
                "temperature": 0,
                "model": "z-ai/glm-5.1",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            },
        ]

    def test_openrouter_schema_fail_fast_for_unsupported_models(self, monkeypatch):
        seen_kwargs: list[dict] = []
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setattr(
            openrouter_module, "OpenAI", lambda **kwargs: _FakeClient(seen_kwargs)
        )

        schema = _simple_schema()
        with pytest.raises(ValueError, match="not supported"):
            openrouter_module.openrouter_chat(
                "prompt",
                model="unsupported/model",
                response_schema=schema,
                estimate_cost=False,
            )

        with pytest.raises(ValueError, match="requires schema name"):
            openrouter_module.openrouter_chat(
                "prompt",
                model="z-ai/glm-5.1",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
                estimate_cost=False,
            )

        assert seen_kwargs == []

    def test_openrouter_glm_falls_back_to_reasoning_content_for_plain_text(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        fallback_response = _FakeResponse(None, reasoning_content="True")
        monkeypatch.setattr(
            openrouter_module,
            "OpenAI",
            lambda **kwargs: _StaticResponseClient(fallback_response),
        )

        result = openrouter_module.openrouter_chat(
            "prompt",
            model="z-ai/glm-5.1",
            estimate_cost=False,
        )

        assert result == "True"

    def test_openrouter_glm_falls_back_to_reasoning_json_for_schema(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        fallback_response = _FakeResponse(None, reasoning='{"value":"ok"}')
        monkeypatch.setattr(
            openrouter_module,
            "OpenAI",
            lambda **kwargs: _StaticResponseClient(fallback_response),
        )

        result = openrouter_module.openrouter_chat(
            "prompt",
            model="z-ai/glm-5.1",
            response_schema=_simple_schema(),
            estimate_cost=False,
        )

        assert result == '{"value":"ok"}'

    def test_openrouter_glm_rejects_invalid_json_fallback_for_schema(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        fallback_response = _FakeResponse(None, reasoning="not-json")
        monkeypatch.setattr(
            openrouter_module,
            "OpenAI",
            lambda **kwargs: _StaticResponseClient(fallback_response),
        )

        with pytest.raises(ValueError, match="non-JSON fallback content"):
            openrouter_module.openrouter_chat(
                "prompt",
                model="z-ai/glm-5.1",
                response_schema=_simple_schema(),
                estimate_cost=False,
            )

    def test_openrouter_non_glm_still_rejects_empty_content(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        fallback_response = _FakeResponse(None, reasoning_content="True")
        monkeypatch.setattr(
            openrouter_module,
            "OpenAI",
            lambda **kwargs: _StaticResponseClient(fallback_response),
        )

        with pytest.raises(ValueError, match="OpenRouter returned empty content"):
            openrouter_module.openrouter_chat(
                "prompt",
                model="openai/gpt-5.4",
                estimate_cost=False,
            )


class TestStructuredCache:
    def test_cached_llm_caller_separates_schema_identity(self, monkeypatch, tmp_path):
        seen_schemas: list[dict | None] = []

        def fake_llm_call(
            prompt,
            llm_provider="azure",
            max_tokens=800,
            response_schema=None,
            temperature=0,
            **kwargs,
        ):
            seen_schemas.append(response_schema)
            return '{"value":"ok"}'

        monkeypatch.setattr("core.pipeline.e2e_utils.cache.llm_call", fake_llm_call)
        caller = CachedLLMCaller(db_path=str(tmp_path / "cache.db"))
        schema = _simple_schema()

        caller.call("prompt", model="gpt-5.4-mini", response_schema=None)
        caller.call("prompt", model="gpt-5.4-mini", response_schema=schema)
        caller.call("prompt", model="gpt-5.4-mini", response_schema=schema)

        assert seen_schemas == [None, schema]

    def test_cached_llm_caller_bypasses_cache_for_nonzero_temperature(
        self, monkeypatch, tmp_path
    ):
        seen_calls: list[float] = []

        def fake_llm_call(
            prompt,
            llm_provider="azure",
            max_tokens=800,
            response_schema=None,
            temperature=0,
            **kwargs,
        ):
            seen_calls.append(temperature)
            return '{"value":"sampled"}'

        monkeypatch.setattr("core.pipeline.e2e_utils.cache.llm_call", fake_llm_call)
        caller = CachedLLMCaller(db_path=str(tmp_path / "cache.db"))

        first = caller.call("prompt", model="gpt-5.4-mini", temperature=0.3)
        second = caller.call("prompt", model="gpt-5.4-mini", temperature=0.3)

        assert first.cache_hit is False
        assert second.cache_hit is False
        assert seen_calls == [0.3, 0.3]

    def test_cache_distinguishes_model_identity(self, monkeypatch, tmp_path):
        seen_models: list[str | None] = []

        def fake_llm_call(
            prompt,
            llm_provider="azure",
            max_tokens=800,
            model=None,
            response_schema=None,
            temperature=0,
            **kwargs,
        ):
            seen_models.append(model)
            return f'{{"model":"{model}"}}'

        monkeypatch.setattr("core.pipeline.e2e_utils.cache.llm_call", fake_llm_call)
        db_path = tmp_path / "cache.db"
        caller = CachedLLMCaller(db_path=str(db_path))

        gpt54_first = caller.call(
            "prompt",
            llm_provider="openrouter",
            model="openai/gpt-5.4",
        )
        gpt54_second = caller.call(
            "prompt",
            llm_provider="openrouter",
            model="openai/gpt-5.4",
        )
        glm_first = caller.call(
            "prompt",
            llm_provider="openrouter",
            model="z-ai/glm-5.1",
        )
        glm_second = caller.call(
            "prompt",
            llm_provider="openrouter",
            model="z-ai/glm-5.1",
        )

        assert gpt54_first.cache_hit is False
        assert gpt54_second.cache_hit is True
        assert glm_first.cache_hit is False
        assert glm_second.cache_hit is True
        assert seen_models == ["openai/gpt-5.4", "z-ai/glm-5.1"]

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT model, llm_provider, COUNT(*) FROM llm_cache GROUP BY model, llm_provider"
            ).fetchall()

        assert sorted(rows) == [
            ("openai/gpt-5.4", "openrouter", 1),
            ("z-ai/glm-5.1", "openrouter", 1),
        ]
