"""OpenAI-compatible model adapter for running MDocAgent inside LSF.

The upstream ``models.openai.MyOpenAI`` path uses the generic OpenAI client and
``max_tokens``. Azure GPT-5.4 deployments require AzureOpenAI plus
``max_completion_tokens``, so the LSF wrapper points Hydra at this adapter.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from models.base_model import BaseModel  # type: ignore[import]
from openai import AzureOpenAI, OpenAI


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
        messages.append(self.create_ans_message(result))
        return result, messages

    def is_valid_history(self, history: Any) -> bool:
        return isinstance(history, list)
