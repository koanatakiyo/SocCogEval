"""LLM API client wrapper supporting multiple providers.

Model strategy:
- Qwen 3 122B FP8 (local vLLM): Primary model, free unlimited
- Llama 4 Scout (local vLLM): Second local model family
- GPT (OpenAI API): Cloud cross-model validation
- DeepSeek v3 (API): Cheap cloud cross-model check
- Gemini Flash (Google Gen AI SDK + ADC): Fifth model family
- Grok (xAI API): Backup
"""

import json
import logging
import os
import threading
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


def _use_llama_api() -> bool:
    """Whether Llama should be routed to a hosted OpenAI-compatible endpoint."""
    return bool(config.LLAMA_API_BASE_URL or config.LLAMA_API_KEY)


def _request_model_name(model: str) -> str:
    """Map local canonical model names to provider-specific API model IDs."""
    if _is_llama_model(model) and _use_llama_api():
        return config.LLAMA_API_MODEL or model
    if _is_deepseek_model(model) and config.DEEPSEEK_API_MODEL:
        return config.DEEPSEEK_API_MODEL
    return model


# ── Clients (lazy singletons) ──────────────────────────────────────────────

_clients: dict[str, openai.OpenAI] = {}
_google_genai_client = None
_gemini_rate_lock = threading.Lock()
_gemini_last_request_at = 0.0
_temperature_warning_models: set[str] = set()
_NON_RETRYABLE_STATUS_CODES = {400, 401, 402, 403, 404}


def _is_non_retryable_error(exc: Exception) -> bool:
    """Return true for provider errors that another immediate retry cannot fix."""
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        # google.genai.errors.APIError exposes HTTP status as ``code``.
        status_code = getattr(exc, "code", None)
    return isinstance(status_code, int) and status_code in _NON_RETRYABLE_STATUS_CODES


def _get_client(name: str, api_key: str, base_url: Optional[str] = None) -> openai.OpenAI:
    """Get or create a named client singleton."""
    if name not in _clients:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _clients[name] = openai.OpenAI(**kwargs)
    return _clients[name]


def _get_google_genai_client():
    """Get the native Google Gen AI client configured for ADC authentication."""
    global _google_genai_client
    if _google_genai_client is not None:
        return _google_genai_client

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
    use_enterprise = os.environ.get(
        "GOOGLE_GENAI_USE_ENTERPRISE", ""
    ).strip().lower()

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not set. Configure Google ADC and export "
            "the Google Cloud project ID before using Gemini."
        )
    if not location:
        raise ValueError(
            "GOOGLE_CLOUD_LOCATION is not set. Export "
            "GOOGLE_CLOUD_LOCATION=global before using Gemini."
        )
    if use_enterprise not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "GOOGLE_GENAI_USE_ENTERPRISE is not enabled. Export "
            "GOOGLE_GENAI_USE_ENTERPRISE=True so Gemini uses ADC rather than "
            "the retired API-key route."
        )

    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini. Install requirements.txt "
            "(google-genai==2.13.0)."
        ) from exc

    # Project, location, and backend selection are intentionally read from
    # the environment, matching Google's current ADC quickstart.
    _google_genai_client = genai.Client(
        http_options=HttpOptions(api_version="v1")
    )
    return _google_genai_client


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
        use_openrouter = "openrouter.ai" in config.DEEPSEEK_API_BASE_URL
        api_key = (
            config.OPENROUTER_API_KEY if use_openrouter
            else config.DEEPSEEK_API_KEY
        )
        if not api_key:
            key_name = "OPENROUTER_API_KEY" if use_openrouter else "DEEPSEEK_API_KEY"
            raise ValueError(f"{key_name} not set for DeepSeek.")
        return _get_client(
            "deepseek_openrouter" if use_openrouter else "deepseek",
            api_key,
            config.DEEPSEEK_API_BASE_URL,
        )

    elif _is_openai_model(model):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Put it in APIkeys/api_key_openai.txt or env.")
        return _get_client("openai", config.OPENAI_API_KEY)

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
    response_metadata: Optional[dict] = None,
) -> str:
    """Send a chat completion request.

    ``response_metadata`` optionally receives provider token usage and request
    metadata. Existing callers keep the same string return value and behavior.
    """
    model = model or config.PILOT_MODEL
    if response_metadata is not None:
        response_metadata.clear()

    # OpenRouter Llama4-Scout intermittently returns empty content with
    # finish_reason=stop; on a 30-round rep (~1800 LLM calls) a 3-attempt
    # budget will eventually hit a triple-empty and kill the rep. Bump
    # retries to 10 specifically for the Llama API path; other providers
    # stay at the caller's default (typically 3).
    if _is_llama_model(model) and _use_llama_api() and retries < 10:
        retries = 10
    if _is_gemini_model(model) and retries < 8:
        retries = 8

    for attempt in range(retries):
        try:
            if _is_gemini_model(model):
                return _chat_google_genai(
                    prompt,
                    system,
                    model,
                    temperature,
                    max_tokens,
                    response_metadata=response_metadata,
                )

            client = _get_client_for_model(model)
            request_model = _request_model_name(model)
            return _chat_openai_compat(
                client,
                prompt,
                system,
                request_model,
                temperature,
                max_tokens,
                response_metadata=response_metadata,
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
            wait_cap = 32 if _is_gemini_model(model) else 8
            wait = min(2 ** attempt, wait_cap)
            logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def _chat_google_genai(
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_metadata: Optional[dict] = None,
) -> str:
    """Call Gemini through the native Google Gen AI SDK using ADC."""
    global _gemini_last_request_at
    from google.genai import types

    try:
        min_interval = float(
            os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "2.0")
        )
    except ValueError as exc:
        raise ValueError(
            "GEMINI_MIN_INTERVAL_SECONDS must be a non-negative number."
        ) from exc
    if min_interval < 0:
        raise ValueError(
            "GEMINI_MIN_INTERVAL_SECONDS must be a non-negative number."
        )

    # Smooth request starts across all callers in this process. Gemini
    # Standard PayGo rejects sharp bursts with transient HTTP 429 responses.
    with _gemini_rate_lock:
        delay = min_interval - (time.monotonic() - _gemini_last_request_at)
        if delay > 0:
            time.sleep(delay)
        _gemini_last_request_at = time.monotonic()

    generation_config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        # Gemini 2.5 Flash defaults to automatic thinking. Disable it to
        # preserve the frozen experiment's short, non-reasoning probe budget.
        thinking_config=(
            types.ThinkingConfig(thinking_budget=0, include_thoughts=False)
            if model.lower().startswith("gemini-2.5")
            else None
        ),
    )
    response = _get_google_genai_client().models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config,
    )

    try:
        text = response.text
    except ValueError:
        text = None
    if text is None or not text.strip():
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        prompt_feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            "Gemini returned no text content "
            f"(finish_reason={getattr(candidate, 'finish_reason', None)}, "
            f"block_reason={getattr(prompt_feedback, 'block_reason', None)})"
        )
    if response_metadata is not None:
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        response_metadata.update(
            {
                "provider": "google_genai",
                "response_id": getattr(response, "response_id", None),
                "finish_reason": str(getattr(candidate, "finish_reason", None)),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "completion_tokens": getattr(usage, "candidates_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                    "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
                },
            }
        )
    return text.strip()


def _chat_openai_compat(
    client: openai.OpenAI,
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_metadata: Optional[dict] = None,
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

    # Disable thinking/reasoning mode for all models that support it.
    # These produce huge chain-of-thought blocks that pollute output and waste tokens.

    if _is_qwen_model(model) or (_is_llama_model(model) and not _use_llama_api()):
        # Local vLLM models: Qwen3 and Llama4 both use chat_template_kwargs
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    elif _is_deepseek_model(model):
        # The official endpoint and OpenRouter use different extension fields
        # for the same non-thinking behavior.
        if "openrouter.ai" in config.DEEPSEEK_API_BASE_URL:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
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

    if response_metadata is not None:
        usage = getattr(response, "usage", None)
        response_metadata.update(
            {
                "provider": "openai_compatible",
                "response_id": getattr(response, "id", None),
                "served_model": getattr(response, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        )

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
