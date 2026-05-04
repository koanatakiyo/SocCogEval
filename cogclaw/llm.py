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


def _uses_max_completion_tokens(model: str) -> bool:
    m = model.lower()
    return _is_openai_model(model) and (
        m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))
    )


def _is_gpt5_model(model: str) -> bool:
    return model.lower().startswith("gpt-5")


def _uses_default_temperature_only(model: str) -> bool:
    return _uses_max_completion_tokens(model)


def _gemini_reasoning_effort(model: str) -> str | None:
    m = model.lower()
    if m.startswith("gemini-2.5"):
        # Gemini 2.5 thinking consumes completion tokens unless disabled. The
        # probe suite uses short max_tokens values, so keep these calls non-thinking.
        return "none"
    return None


def _use_llama_api() -> bool:
    """Whether Llama should be routed to a hosted OpenAI-compatible endpoint."""
    return bool(config.LLAMA_API_BASE_URL or config.LLAMA_API_KEY)


def _request_model_name(model: str) -> str:
    """Map local canonical model names to provider-specific API model IDs."""
    if _is_llama_model(model) and _use_llama_api():
        return config.LLAMA_API_MODEL or model
    return model


# ── Clients (lazy singletons) ──────────────────────────────────────────────

_clients: dict[str, openai.OpenAI] = {}
_temperature_warning_models: set[str] = set()
_NON_RETRYABLE_STATUS_CODES = {400, 401, 402, 403, 404}


def _is_non_retryable_error(exc: Exception) -> bool:
    """Return true for provider errors that another immediate retry cannot fix."""
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code in _NON_RETRYABLE_STATUS_CODES


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
        if _use_llama_api():
            if not config.LLAMA_API_BASE_URL:
                raise ValueError("LLAMA_API_BASE_URL not set for hosted Llama API.")
            if not config.LLAMA_API_KEY:
                raise ValueError("LLAMA_API_KEY not set. Put it in APIkeys/api_key_llama.txt or env.")
            return _get_client("llama_api", config.LLAMA_API_KEY, config.LLAMA_API_BASE_URL)
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

    # OpenRouter Llama4-Scout intermittently returns empty content with
    # finish_reason=stop; on a 30-round rep (~1800 LLM calls) a 3-attempt
    # budget will eventually hit a triple-empty and kill the rep. Bump
    # retries to 10 specifically for the Llama API path; other providers
    # stay at the caller's default (typically 3).
    if _is_llama_model(model) and _use_llama_api() and retries < 10:
        retries = 10

    for attempt in range(retries):
        try:
            client = _get_client_for_model(model)
            request_model = _request_model_name(model)
            return _chat_openai_compat(
                client, prompt, system, request_model, temperature, max_tokens
            )
        except Exception as e:
            if _is_non_retryable_error(e):
                logger.warning(
                    "LLM call failed with non-retryable status: %s",
                    e,
                )
                raise
            # Cap exponential backoff at 8s — protects all providers from a
            # runaway 128s sleep on attempt 7+.
            wait = min(2 ** attempt, 8)
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

    kwargs = dict(
        model=model,
        messages=messages,
    )
    if _uses_default_temperature_only(model):
        if model not in _temperature_warning_models:
            logger.warning(
                "%s does not support custom temperature in Chat Completions; "
                "using provider default temperature.",
                model,
            )
            _temperature_warning_models.add(model)
    else:
        kwargs["temperature"] = temperature

    if _uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max(max_tokens, 64)
        kwargs["reasoning_effort"] = "minimal" if _is_gpt5_model(model) else "low"
        if _is_gpt5_model(model):
            kwargs["verbosity"] = "low"
    else:
        kwargs["max_tokens"] = max_tokens

    gemini_reasoning_effort = _gemini_reasoning_effort(model)
    if gemini_reasoning_effort:
        kwargs["reasoning_effort"] = gemini_reasoning_effort

    # Disable thinking/reasoning mode for all models that support it.
    # These produce huge chain-of-thought blocks that pollute output and waste tokens.

    if _is_qwen_model(model) or (_is_llama_model(model) and not _use_llama_api()):
        # Local vLLM models: Qwen3 and Llama4 both use chat_template_kwargs
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    elif _is_deepseek_model(model):
        # DeepSeek API: no official toggle, but we can suppress via extra_body
        # and rely on the safety-net strip below for <think> blocks
        pass
    elif _is_gemini_model(model):
        # Gemini's OpenAI-compatible endpoint rejects provider-specific fields.
        pass

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    text = choice.message.content
    if text is None:
        finish_reason = getattr(choice, "finish_reason", None)
        raise RuntimeError(
            f"LLM returned no text content (finish_reason={finish_reason})"
        )
    if not text.strip():
        finish_reason = getattr(choice, "finish_reason", None)
        raise RuntimeError(
            f"LLM returned empty text content (finish_reason={finish_reason})"
        )

    # Safety net: strip any leaked thinking blocks from Qwen3/other models
    if text:
        import re
        # Strip <think>...</think> XML blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip "Thinking Process:" preamble — if the response starts with it,
        # try to find where actual content begins (after the numbered analysis)
        if text.startswith(("Thinking Process:", "Thought Process:")):
            # Look for a JSON block
            json_start = text.find("{")
            if json_start != -1:
                text = text[json_start:]
            else:
                # Look for where the actual response starts after thinking
                # (usually after a double newline followed by non-numbered text)
                match = re.search(r"\n\n(?=[A-Z][a-z])", text[100:])
                if match:
                    text = text[100 + match.start():].strip()

    return text


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
