"""LLM API client wrapper supporting Claude (Anthropic) and DeepSeek (OpenAI-compatible)."""

import json
import logging
import time
from typing import Optional

import anthropic
import openai

from . import config

logger = logging.getLogger(__name__)

# ── Provider Detection ──────────────────────────────────────────────────────

def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


def _is_deepseek_model(model: str) -> bool:
    return "deepseek" in model.lower()


# ── Clients (lazy singletons) ──────────────────────────────────────────────

_anthropic_client: Optional[anthropic.Anthropic] = None
_openai_client: Optional[openai.OpenAI] = None
_deepseek_client: Optional[openai.OpenAI] = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = config.ANTHROPIC_API_KEY
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set. Put it in api_key_anthropic.txt or env.")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _get_deepseek_client() -> openai.OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        key = config.DEEPSEEK_API_KEY
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set. Put it in api_key_deepseek.txt or env.")
        _deepseek_client = openai.OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_client


def _get_openai_client() -> openai.OpenAI:
    """Generic OpenAI-compatible client (can use apiyi proxy)."""
    global _openai_client
    if _openai_client is None:
        key = config.APIYI_API_KEY or config.OPENAI_API_KEY
        base_url = "https://api.apiyi.com/v1" if config.APIYI_API_KEY else None
        if not key:
            raise ValueError("No OpenAI/apiyi API key found.")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client = openai.OpenAI(**kwargs)
    return _openai_client


# ── Core Chat Function ──────────────────────────────────────────────────────

def chat(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
) -> str:
    """Send a chat completion request. Auto-detects provider from model name."""
    model = model or config.PILOT_MODEL

    for attempt in range(retries):
        try:
            if _is_anthropic_model(model):
                return _chat_anthropic(prompt, system, model, temperature, max_tokens)
            elif _is_deepseek_model(model):
                return _chat_openai_compat(
                    _get_deepseek_client(), prompt, system, model, temperature, max_tokens
                )
            else:
                return _chat_openai_compat(
                    _get_openai_client(), prompt, system, model, temperature, max_tokens
                )
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def _chat_anthropic(
    prompt: str, system: str, model: str, temperature: float, max_tokens: int
) -> str:
    client = _get_anthropic_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def _chat_openai_compat(
    client: openai.OpenAI,
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── JSON Chat ───────────────────────────────────────────────────────────────

def chat_json(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """Chat expecting JSON output. Extracts and repairs JSON from response."""
    full_prompt = prompt + "\n\nRespond with valid JSON only. No markdown, no explanation."
    raw = chat(full_prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens)
    return _parse_json(raw)


def _parse_json(text: str) -> dict:
    """Extract JSON object from LLM response, with repair."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = text[start:end + 1]
        block = _repair_json(block)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")


def _repair_json(s: str) -> str:
    """Repair common LLM JSON errors."""
    import re
    # Remove single-line comments
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    # Remove multi-line comments
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    # Replace Python booleans/None
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s
