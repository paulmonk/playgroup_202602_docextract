"""Tests for the OpenRouter LLM client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from lib import llm_openrouter
from lib.llm_openrouter import _get_providers, call_llm


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the module-level client between tests."""
    llm_openrouter._client = None
    yield
    llm_openrouter._client = None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-3.5-haiku", None),
        ("openai/gpt-4", None),
        ("google/gemini-pro", None),
        ("deepseek/deepseek-v3.2-speciale", ("atlas-cloud",)),
        ("z-ai/glm-4.7", ("z-ai",)),
    ],
)
def test_get_providers(model, expected):
    assert _get_providers(model) == expected


def test_get_providers_unknown_model_raises():
    with pytest.raises(ValueError, match="No provider configured"):
        _get_providers("unknown/model-name")


def test_get_client_creates_on_first_call(mocker):
    mock_openai = mocker.patch("lib.llm_openrouter.OpenAI")
    mocker.patch("lib.llm_openrouter.os.getenv", return_value="test-key")
    client = llm_openrouter._get_client()
    mock_openai.assert_called_once()
    assert client is mock_openai.return_value


def test_get_client_reuses_on_second_call(mocker):
    mock_openai = mocker.patch("lib.llm_openrouter.OpenAI")
    mocker.patch("lib.llm_openrouter.os.getenv", return_value="test-key")
    first = llm_openrouter._get_client()
    second = llm_openrouter._get_client()
    assert first is second
    mock_openai.assert_called_once()


def test_get_client_missing_api_key_raises(mocker):
    mocker.patch("lib.llm_openrouter.os.getenv", return_value=None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        llm_openrouter._get_client()


def test_get_client_empty_api_key_raises(mocker):
    mocker.patch("lib.llm_openrouter.os.getenv", return_value="")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        llm_openrouter._get_client()


def test_call_llm_text_only(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.return_value = SimpleNamespace(output_text="some result")

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "some result"
    mock_client.responses.create.assert_called_once()


def test_call_llm_returns_none_for_empty(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.return_value = SimpleNamespace(output_text="")

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result is None


def test_call_llm_returns_none_for_whitespace_only(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.return_value = SimpleNamespace(output_text="   \n  ")

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result is None


def test_call_llm_strips_whitespace(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.return_value = SimpleNamespace(output_text="  hello  ")

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "hello"


def test_call_llm_with_images(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_choice = SimpleNamespace(message=SimpleNamespace(content="image result"))
    mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
        images=["base64data"],
    )
    assert result == "image result"
    mock_client.chat.completions.create.assert_called_once()


def test_call_llm_retries_then_succeeds(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.side_effect = [
        json.JSONDecodeError("bad", "", 0),
        SimpleNamespace(output_text="recovered"),
    ]

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "recovered"
    assert mock_client.responses.create.call_count == 2


def test_call_llm_exhausts_retries_raises(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.side_effect = json.JSONDecodeError("bad", "", 0)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        call_llm(
            "anthropic/claude-3.5-haiku",
            system_prompt="sys",
            user_prompt="user",
        )
    assert mock_client.responses.create.call_count == 3


def test_call_llm_images_exhausts_retries_raises(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.chat.completions.create.side_effect = json.JSONDecodeError("bad", "", 0)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        call_llm(
            "anthropic/claude-3.5-haiku",
            system_prompt="sys",
            user_prompt="user",
            images=["img"],
        )
    assert mock_client.chat.completions.create.call_count == 3


def test_call_llm_retries_on_rate_limit(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.side_effect = [
        RateLimitError(
            message="rate limited",
            response=mocker.MagicMock(status_code=429, headers={}),
            body=None,
        ),
        SimpleNamespace(output_text="recovered after rate limit"),
    ]

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "recovered after rate limit"
    assert mock_client.responses.create.call_count == 2


def test_call_llm_retries_on_api_timeout(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.side_effect = [
        APITimeoutError(request=mocker.MagicMock()),
        SimpleNamespace(output_text="recovered after timeout"),
    ]

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "recovered after timeout"
    assert mock_client.responses.create.call_count == 2


def test_call_llm_retries_on_connection_error(mocker):
    mock_client = mocker.MagicMock()
    mocker.patch("lib.llm_openrouter._get_client", return_value=mock_client)
    mock_client.responses.create.side_effect = [
        APIConnectionError(request=mocker.MagicMock()),
        SimpleNamespace(output_text="recovered after connection error"),
    ]

    result = call_llm(
        "anthropic/claude-3.5-haiku",
        system_prompt="sys",
        user_prompt="user",
    )
    assert result == "recovered after connection error"
    assert mock_client.responses.create.call_count == 2
