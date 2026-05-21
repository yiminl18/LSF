"""OpenAI-compatible model adapter for running MDocAgent inside LSF.

The upstream ``models.openai.MyOpenAI`` path uses the generic OpenAI client and
``max_tokens``. Azure GPT-5.4 deployments require AzureOpenAI plus
``max_completion_tokens``, so the LSF wrapper points Hydra at this adapter.

Cost telemetry: upstream ``MultiAgentSystem`` does not surface token usage.
When the env var ``LSF_MDOCAGENT_USAGE_LOG`` is set, every ``predict()`` call
appends one JSON line to that path with ``input_tokens``, ``output_tokens``,
and ``cost_usd`` so the extractor can aggregate the run's reader cost.
``LSF_MDOCAGENT_LOGICAL_MODEL`` overrides the model name used for cost
lookup (necessary on Azure where ``self.model`` becomes the deployment name,
not the logical model name expected by ``core.llm.cost``).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from models.base_model import BaseModel  # type: ignore[import]
from openai import AzureOpenAI, OpenAI

from core.llm.cost import compute_cost


def _encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _image_mime(image_path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(image_path)
    return guessed or "image/png"


class MyOpenAI(BaseModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = self.config.model
        provider = os.environ.get("LSF_MDOCAGENT_PROVIDER", "").strip().lower()
        self._is_azure = provider == "azure"
        self._provider = provider or "azure"
        # Logical model name for cost lookup. On Azure, `self.model` is
        # replaced with the deployment name below; the cost table keys off
        # the logical name (e.g. "gpt-5.4-mini"), so we capture it first.
        self._logical_model = (
            os.environ.get("LSF_MDOCAGENT_LOGICAL_MODEL", "").strip()
            or self.config.model
        )
        self._usage_log_path = os.environ.get("LSF_MDOCAGENT_USAGE_LOG", "").strip()
        if self._is_azure:
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["LSF_MDOCAGENT_AZURE_API_BASE"],
                api_key=os.environ["OPENAI_API_KEY"],
                api_version=os.environ["LSF_MDOCAGENT_AZURE_API_VERSION"],
            )
            self.model = os.environ.get("LSF_MDOCAGENT_AZURE_DEPLOYMENT", self.model)
        else:
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

    def create_ask_message(self, question: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }

    def create_ans_message(self, ans: str) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": ans}],
        }

    def create_text_message(self, texts: list[str], question: str) -> dict[str, Any]:
        content = [{"type": "text", "text": text} for text in texts]
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}

    def create_image_message(self, images: list[str], question: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for image_path in images:
            mime = _image_mime(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{_encode_image(image_path)}"
                    },
                }
            )
        content.append({"type": "text", "text": question})
        return {"role": "user", "content": content}

    def predict(
        self,
        question: str,
        texts: list[str] | None = None,
        images: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        messages = self.process_message(question, texts, images, history)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self._is_azure:
            request_kwargs["max_completion_tokens"] = self.config.max_new_tokens
        else:
            request_kwargs["max_tokens"] = self.config.max_new_tokens

        response = self.client.chat.completions.create(**request_kwargs)
        result = response.choices[0].message.content or ""
        self._record_usage(response)
        messages.append(self.create_ans_message(result))
        return result, messages

    def _record_usage(self, response: Any) -> None:
        """Append one JSONL line with token usage + computed cost.

        No-op when LSF_MDOCAGENT_USAGE_LOG is unset (e.g. running outside the
        LSF baseline harness) or when the response carries no usage block.
        """
        if not self._usage_log_path:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if input_tokens == 0 and output_tokens == 0:
            return
        try:
            cost_usd = compute_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_provider=self._provider,
                model=self._logical_model,
            )
        except Exception:
            cost_usd = 0.0
        record = {
            "ts": time.time(),
            "provider": self._provider,
            "model": self._logical_model,
            "deployment": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        try:
            Path(self._usage_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Telemetry must never break the prediction path.
            pass

    def is_valid_history(self, history: Any) -> bool:
        return isinstance(history, list)
