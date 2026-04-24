from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


_OPENAI_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _response_text_to_json(text: str) -> dict[str, Any] | None:
    content = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
    if fenced_match:
        content = fenced_match.group(1)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def call_openai_json(
    *,
    system_prompt: str,
    user_prompt: str,
    default: dict[str, Any],
    max_output_tokens: int = 700,
) -> dict[str, Any]:
    client = _get_client()
    if client is None:
        return default

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            max_output_tokens=max_output_tokens,
        )
    except Exception:
        return default

    output_text = (response.output_text or "").strip()
    if not output_text:
        return default

    parsed = _response_text_to_json(output_text)
    if parsed is None:
        return default

    return parsed


def call_openai_text(
    *,
    system_prompt: str,
    user_prompt: str,
    default: str,
    max_output_tokens: int = 280,
) -> str:
    client = _get_client()
    if client is None:
        return default

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            max_output_tokens=max_output_tokens,
        )
    except Exception:
        return default

    output_text = (response.output_text or "").strip()
    return output_text or default
