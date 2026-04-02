"""Centralized LLM calling helper for AEL framework.

Handles model name normalization for litellm routing across providers:
  - "gpt-4o-mini"                                 → openai/gpt-4o-mini
  - "bedrock/us.anthropic.claude-haiku-..."        → passed through
  - "claude-sonnet-4-20250514"                     → passed through
  - "ollama/llama3.2:3b"                           → passed through

litellm.drop_params silently drops unsupported params (e.g., temperature
on models that don't support it).
"""

from __future__ import annotations

import litellm

# Allow litellm to silently drop unsupported params
litellm.drop_params = True

# Retry on rate limit errors (429) with exponential backoff
litellm.num_retries = 5
litellm.request_timeout = 120


def normalize_model(model: str) -> str:
    """Ensure model string has provider prefix for litellm routing.

    Models that already have a provider prefix (contain '/') are passed
    through unchanged. Bare OpenAI model names get 'openai/' prepended.
    """
    if not model:
        return model
    # Already has a provider prefix (bedrock/, openai/, ollama/, etc.)
    if "/" in model:
        return model
    # Add openai/ prefix for OpenAI models
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{model}"
    return model


def llm_call(
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    response_format: dict | None = None,
    **kwargs,
) -> litellm.ModelResponse:
    """Centralized LLM call with provider-agnostic routing.

    - Normalizes model name (adds provider prefix if needed)
    - litellm.drop_params=True handles unsupported params gracefully
    """
    model = normalize_model(model)

    call_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    if response_format is not None:
        call_kwargs["response_format"] = response_format

    return litellm.completion(**call_kwargs)
