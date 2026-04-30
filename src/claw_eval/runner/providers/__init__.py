"""Model providers."""

from __future__ import annotations

from .openai_compat import OpenAICompatProvider

__all__ = ["OpenAICompatProvider", "make_provider"]


def make_provider(model_cfg):
    """Factory: build the right provider for a ModelConfig.

    Dispatches on ``model_cfg.provider`` ("openai_compat" default,
    "litellm" optional). Unknown providers raise so config typos surface
    immediately rather than silently falling back.
    """
    common = {
        "model_id": model_cfg.model_id,
        "api_key": model_cfg.api_key,
        "base_url": model_cfg.base_url,
        "extra_body": model_cfg.extra_body,
        "temperature": model_cfg.temperature,
        "reasoning_effort": model_cfg.reasoning_effort,
    }

    if model_cfg.provider == "litellm":
        from .litellm import LiteLLMProvider

        return LiteLLMProvider(**common, litellm_kwargs=model_cfg.litellm_kwargs)
    if model_cfg.provider == "openai_compat":
        return OpenAICompatProvider(**common)
    raise ValueError(
        f"Unknown provider {model_cfg.provider!r} (expected 'openai_compat' or 'litellm')."
    )
