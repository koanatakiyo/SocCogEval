"""Regression tests for endpoint-specific DeepSeek non-thinking parameters."""

from types import SimpleNamespace
from unittest.mock import patch

from cogclaw.llm import _chat_openai_compat


def _fake_client(calls):
    choice = SimpleNamespace(
        message=SimpleNamespace(content="OK"),
        finish_reason="stop",
    )
    response = SimpleNamespace(
        choices=[choice],
        usage=None,
        id="response-deepseek",
        model="deepseek-v4-flash",
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: calls.append(kwargs) or response
            )
        )
    )


def test_official_deepseek_uses_official_non_thinking_parameter():
    calls = []
    with patch(
        "cogclaw.llm.config.DEEPSEEK_API_BASE_URL",
        "https://api.deepseek.com/v1",
    ):
        _chat_openai_compat(
            _fake_client(calls),
            "prompt",
            "system",
            "deepseek-v4-flash",
            0.7,
            32,
        )
    assert calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openrouter_deepseek_keeps_openrouter_non_thinking_parameter():
    calls = []
    with patch(
        "cogclaw.llm.config.DEEPSEEK_API_BASE_URL",
        "https://openrouter.ai/api/v1",
    ):
        _chat_openai_compat(
            _fake_client(calls),
            "prompt",
            "system",
            "deepseek/deepseek-v4-flash",
            0.7,
            32,
        )
    assert calls[-1]["extra_body"] == {"reasoning": {"enabled": False}}
