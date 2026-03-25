"""LLM API client wrapper supporting multiple providers.

Model strategy:
- Qwen 3 122B (local): Primary model, free unlimited
- Llama 4 Scout (local): Second local model family
- GPT-5.3 mini (OpenAI API): Cloud cross-model validation
- DeepSeek v3.2 (API): Cheap cloud cross-model check
- Gemini 3.1 Flash (API): Fifth model family
- Grok (backup)
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

_qwen_client: Optional[openai.OpenAI] = None
_openai_client: Optional[openai.OpenAI] = None
_deepseek_client: Optional[openai.OpenAI] = None
_llama_client: Optional[openai.OpenAI] = None
_apiyi_client: Optional[openai.OpenAI] = None


def _get_qwen_client() -> openai.OpenAI:
    """Local Qwen 3 122B via OpenAI-compatible endpoint (vLLM/Ollama)."""
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = openai.OpenAI(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
        )
    return _qwen_client


def _get_openai_client() -> openai.OpenAI:
    """OpenAI API client for GPT models."""
    global _openai_client
    if _openai_client is None:
        key = config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY not set. Put it in api_key_openai.txt or env.")
        _openai_client = openai.OpenAI(api_key=key)
    return _openai_client


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


def _get_llama_client() -> openai.OpenAI:
    """Llama 4 Scout — local via vLLM/Ollama or cloud via apiyi/together."""
    global _llama_client
    if _llama_client is None:
        # Try local first (same server, different model name), fallback to apiyi
        llama_base = os.environ.get("LLAMA_BASE_URL", config.QWEN_BASE_URL)
        llama_key = os.environ.get("LLAMA_API_KEY", config.QWEN_API_KEY)
        _llama_client = openai.OpenAI(api_key=llama_key, base_url=llama_base)
    return _llama_client


def _get_apiyi_client() -> openai.OpenAI:
    """Apiyi proxy client (for Gemini, Grok, etc.)."""
    global _apiyi_client
    if _apiyi_client is None:
        key = config.APIYI_API_KEY
        if not key:
            raise ValueError("APIYI_API_KEY not set. Put it in api_key_apiyi.txt or env.")
        _apiyi_client = openai.OpenAI(
            api_key=key,
            base_url="https://api.apiyi.com/v1",
        )
    return _apiyi_client


def _get_client_for_model(model: str) -> openai.OpenAI:
    """Route model name to the correct client."""
    if _is_qwen_model(model):
        return _get_qwen_client()
    elif _is_deepseek_model(model):
        return _get_deepseek_client()
    elif _is_openai_model(model):
        return _get_openai_client()
    elif _is_llama_model(model):
        return _get_llama_client()
    elif _is_gemini_model(model) or _is_grok_model(model):
        return _get_apiyi_client()
    else:
        # Fallback: apiyi proxy
        return _get_apiyi_client()


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
