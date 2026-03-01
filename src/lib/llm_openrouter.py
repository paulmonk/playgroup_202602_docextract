"""OpenRouter LLM client with provider pinning for reproducible results."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Final

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

MAX_RETRIES: Final[int] = 3

# Errors worth retrying: transient API/network failures plus malformed JSON
_RETRYABLE_ERRORS = (
    json.JSONDecodeError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)

_client: OpenAI | None = None

# Pin a single provider per model family for reproducible results.
# OpenRouter falls back to other providers with different quantisation
# levels, which gives inconsistent outputs.
_PROVIDER_BY_PREFIX: Final[Mapping[str, tuple[str, ...] | None]] = {
    "deepseek/deepseek-v3.2-speciale": ("atlas-cloud",),
    "deepseek/deepseek-v3.1-terminus": ("atlas-cloud",),
    "z-ai/glm-4.7": ("z-ai",),
    "anthropic": None,
    "openai": None,
    "google": None,
}


def _get_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on first use.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not set.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def _retry_on_transient_error[T](fn: Callable[[], T]) -> T:
    """Call fn up to MAX_RETRIES times, retrying on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            logger.warning(
                "%s on attempt %d/%d, retrying: %s",
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_exc!r}") from last_exc


def _get_providers(model_name: str) -> tuple[str, ...] | None:
    """Look up the pinned provider list for a model family.

    Args:
        model_name: Full OpenRouter model identifier.

    Returns:
        Provider tuple to pass to OpenRouter, or None for default routing.

    Raises:
        ValueError: If no provider is configured for the model prefix.
    """
    for prefix, providers in _PROVIDER_BY_PREFIX.items():
        if model_name.startswith(prefix):
            return providers
    raise ValueError(
        f"No provider configured for {model_name!r}."
        f" Add an entry to _PROVIDER_BY_PREFIX in {__name__}."
    )


def _call_chat_with_images(
    model_name: str,
    *,
    system_prompt: str,
    user_prompt: str,
    images: list[str],
    extra_params: dict,
) -> str:
    """Use the Chat Completions API for multimodal (text+image) requests.

    The Responses API doesn't support image inputs, so we fall back to
    chat completions when images are provided.

    Args:
        model_name: OpenRouter model identifier.
        system_prompt: System-level instructions.
        user_prompt: User-level prompt text.
        images: Base64-encoded PNG images.
        extra_params: Provider routing params.

    Returns:
        Raw output text from the model.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
        *[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
            for img_b64 in images
        ],
    ]
    logger.debug("Sending %d images via chat completions API", len(images))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    client = _get_client()
    response = _retry_on_transient_error(
        lambda: client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore[arg-type]
            extra_body=extra_params,
        )
    )
    return response.choices[0].message.content or ""


def call_llm(
    model_name: str,
    *,
    system_prompt: str,
    user_prompt: str,
    images: list[str] | None = None,
) -> str | None:
    """Send a prompt to OpenRouter and return the raw response text.

    Args:
        model_name: OpenRouter model identifier.
        system_prompt: System-level instructions (passed via the instructions role).
        user_prompt: User-level prompt with field definitions and document text.
        images: Optional list of base64-encoded PNG images to include.

    Returns:
        Raw response text, or None if empty.
    """
    logger.debug("Prompt:\n%s", user_prompt)
    only_providers = _get_providers(model_name)
    extra_params = {"provider": {"allow_fallbacks": False, "only": only_providers}}
    logger.info("LLM calling with %s", model_name)

    if images:
        raw_output = _call_chat_with_images(
            model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            extra_params=extra_params,
        )
    else:
        client = _get_client()
        response = _retry_on_transient_error(
            lambda: client.responses.create(
                model=model_name,
                instructions=system_prompt,
                input=user_prompt,
                extra_body=extra_params,
            )
        )
        raw_output = response.output_text or ""

    logger.debug("Raw return from llm call:\n%s", raw_output)
    return raw_output.strip() or None
