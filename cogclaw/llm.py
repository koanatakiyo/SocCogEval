"""LLM API client wrapper supporting multiple providers.

Model strategy:
- Qwen 3 122B FP8 (local vLLM): Primary model, free unlimited
- Llama 4 Scout (local vLLM): Second local model family
- GPT (OpenAI API): Cloud cross-model validation
- DeepSeek v3 (API): Cheap cloud cross-model check
- Gemini Flash (Google API): Fifth model family
- Grok (xAI API): Backup
"""

import json
import logging
import os
import time
from typing import Optional

import openai

from . import config

logger = logging.getLogger(__name__)

# ── Provider Detection ──────────────────────────────────────────────────────

def _is_qwen_model(model: str) -> bool:
    return "qwen" in model.lower()


def _is_deepseek_model(model: str) -> bool:
    return "deepseek" in model.lower()


def _is_openai_model(model: str) -> bool:
    return "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower()


def _is_llama_model(model: str) -> bool:
    return "llama" in model.lower()


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _is_grok_model(model: str) -> bool:
    return "grok" in model.lower()


# ── Clients (lazy singletons) ──────────────────────────────────────────────

_clients: dict[str, openai.OpenAI] = {}


def _get_client(name: str, api_key: str, base_url: Optional[str] = None) -> openai.OpenAI:
    """Get or create a named client singleton."""
    if name not in _clients:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _clients[name] = openai.OpenAI(**kwargs)
    return _clients[name]


def _get_client_for_model(model: str) -> openai.OpenAI:
    """Route model name to the correct client."""
    if _is_qwen_model(model):
        return _get_client("qwen", config.LOCAL_API_KEY, config.QWEN_BASE_URL)

    elif _is_llama_model(model):
        return _get_client("llama", config.LOCAL_API_KEY, config.LLAMA_BASE_URL)

    elif _is_deepseek_model(model):
        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not set. Put it in APIkeys/api_key_deepseek.txt or env.")
        return _get_client("deepseek", config.DEEPSEEK_API_KEY, "https://api.deepseek.com/v1")

    elif _is_openai_model(model):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Put it in APIkeys/api_key_openai.txt or env.")
        return _get_client("openai", config.OPENAI_API_KEY)

    elif _is_gemini_model(model):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set. Put it in APIkeys/api_key_gemini.txt or env.")
        return _get_client(
            "gemini", config.GEMINI_API_KEY,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    elif _is_grok_model(model):
        if not config.GROK_API_KEY:
            raise ValueError("GROK_API_KEY not set. Put it in APIkeys/api_key_grok.txt or env.")
        return _get_client("grok", config.GROK_API_KEY, "https://api.x.ai/v1")

    else:
        raise ValueError(f"Unknown model: {model}. Cannot route to provider.")


# ── Core Chat Function ──────────────────────────────────────────────────────

def chat(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
) -> str:
    """Send a chat completion request. Auto-routes to correct provider."""
    model = model or config.PILOT_MODEL

    for attempt in range(retries):
        try:
            client = _get_client_for_model(model)
            return _chat_openai_compat(
                client, prompt, system, model, temperature, max_tokens
            )
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


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
