"""
llm.py

Thin wrapper around an OpenAI-compatible chat completions endpoint.

Why OpenAI-compatible instead of a specific SDK: Groq, OpenRouter, and Gemini
(via its OpenAI-compat endpoint) all speak this same interface, so swapping
providers is just changing three env vars, not rewriting call sites. Point
this at whichever free tier you're using:

  Groq:       LLM_BASE_URL=https://api.groq.com/openai/v1   LLM_MODEL=llama-3.3-70b-versatile
  OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1      LLM_MODEL=<any free model slug>
  Gemini:     LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/  LLM_MODEL=gemini-2.0-flash

Set LLM_API_KEY to the provider's key.
"""

import json
import os

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
    return _client


def call_llm_json(system: str, user: str, timeout: float = 12.0) -> dict:
    """Call the LLM asking for a strict JSON object reply, parse it, and
    return a dict. Raises on malformed output so the caller can decide how
    to degrade (we do NOT want to silently guess at controller intent)."""
    client = get_client()
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        timeout=timeout,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences some models add despite instructions.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def call_llm_text(system: str, user: str, timeout: float = 12.0) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()
