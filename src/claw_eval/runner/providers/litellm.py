"""LiteLLM AI Gateway provider.

Routes every chat completion through ``litellm.completion`` so a single
provider can run claw-eval against OpenAI, Anthropic, Vertex AI, Bedrock,
Azure, Cohere, Mistral, Groq, Ollama, and 90+ other backends without
configuring a separate OpenAI-compatible proxy server (OpenRouter, vLLM,
LiteLLM proxy, etc.). Pick the model with the standard LiteLLM
provider-prefixed name ("anthropic/claude-3-5-sonnet-20241022",
"vertex_ai/gemini-1.5-pro", "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
"azure/gpt-4o", ...) and credentials are resolved from provider-specific
environment variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``AWS_ACCESS_KEY_ID``, ...) by default.

``drop_params=True`` is on by default so kwargs that some providers reject
(``frequency_penalty`` / ``presence_penalty`` on Anthropic, Gemini, Bedrock;
``response_format`` on Bedrock; etc.) are silently dropped instead of raising
``UnsupportedParamsError``.
"""

from __future__ import annotations

from typing import Any

from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import TokenUsage
from .openai_compat import OpenAICompatProvider, _message_to_openai, _tool_spec_to_openai


class LiteLLMProvider(OpenAICompatProvider):
    """LiteLLM-routed provider; embedded SDK, no proxy server."""

    def __init__(
        self,
        model_id: str = "anthropic/claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        litellm_kwargs: dict | None = None,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "litellm is required for LiteLLMProvider. "
                'Install via `pip install "claw-eval[litellm]"` or '
                '`pip install "litellm>=1.60,<1.85"`.'
            ) from exc

        # We deliberately skip OpenAICompatProvider.__init__ because that one
        # builds an `openai.OpenAI` client we never use. Set the same
        # attributes its inherited methods (notably _parse_response) read.
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.extra_body = extra_body or {}
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        merged: dict[str, Any] = {"drop_params": True, "num_retries": 5}
        if litellm_kwargs:
            merged.update(litellm_kwargs)
        self.litellm_kwargs = merged

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> tuple[Message, TokenUsage]:
        """Send messages to the model via litellm.completion and return parsed response."""
        import litellm

        oai_messages: list[dict[str, Any]] = []
        for msg in messages:
            converted = _message_to_openai(msg)
            if isinstance(converted, list):
                oai_messages.extend(converted)
            else:
                oai_messages.append(converted)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": oai_messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            # litellm forwards extra_body to the underlying provider, same as
            # the openai SDK path.
            kwargs["extra_body"] = dict(self.extra_body)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = [_tool_spec_to_openai(t) for t in tools]

        # Provider-level credentials override per-request env-var resolution
        # so users with a single shared key (private LiteLLM proxy / custom
        # OpenAI-compatible endpoint) can configure once.
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # litellm_kwargs (drop_params, num_retries, etc.) come last so users
        # can override anything above by setting it in litellm_kwargs.
        kwargs.update(self.litellm_kwargs)

        response = litellm.completion(**kwargs)
        return self._parse_response(response)
