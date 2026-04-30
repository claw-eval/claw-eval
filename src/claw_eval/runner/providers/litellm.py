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

The class subclasses ``OpenAICompatProvider`` and overrides only the two
methods that do the network call (``_call_without_stream`` and
``_call_with_stream``); the rest of the chat loop, multimodal check,
retry-on-error logic, streaming-fallback, and response parsing is inherited
unchanged.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatProvider


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

        # Skip OpenAICompatProvider.__init__ (we don't need its openai.OpenAI
        # client). Set every attribute the inherited chat()/_parse_response
        # methods read.
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

    def _full_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add credentials and litellm-specific kwargs onto the request."""
        full = dict(kwargs)
        if self.api_key:
            full["api_key"] = self.api_key
        if self.base_url:
            full["api_base"] = self.base_url
        # litellm_kwargs come last so users can override anything in `kwargs`
        # (drop_params, num_retries, custom timeouts, etc.).
        full.update(self.litellm_kwargs)
        return full

    def _call_without_stream(self, kwargs: dict[str, Any]) -> Any:
        """Non-streaming completion via litellm.completion()."""
        import litellm

        return litellm.completion(**self._full_kwargs(kwargs))

    def _call_with_stream(self, kwargs: dict[str, Any]) -> Any:
        """Streaming completion via litellm.completion(stream=True).

        Mirrors the chunk-assembly pattern in OpenAICompatProvider so the
        inherited ``_parse_response`` sees the same duck-typed object shape.
        """
        import litellm

        stream_kwargs: dict[str, Any] = {"stream": True}
        # stream_options is OpenAI-specific; Anthropic / Claude endpoints
        # reject it. LiteLLM normalizes the response shape either way.
        model_lower = kwargs.get("model", "").lower()
        is_anthropic = any(s in model_lower for s in ("claude", "anthropic"))
        if not is_anthropic:
            stream_kwargs["stream_options"] = {"include_usage": True}

        full = self._full_kwargs({**kwargs, **stream_kwargs})
        stream = litellm.completion(**full)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage_info = None
        has_any_choice = False

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_info = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            has_any_choice = True
            delta = choices[0].delta

            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                reasoning_parts.append(rc)

            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_by_index[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_by_index[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_by_index[idx]["arguments"] += tc_delta.function.arguments

        if not has_any_choice:
            raise RuntimeError("Model returned empty choices (choices=None or [])")

        class _Msg:
            pass

        msg = _Msg()
        msg.content = "".join(content_parts) if content_parts else None
        msg.reasoning_content = "".join(reasoning_parts) if reasoning_parts else None

        if tool_calls_by_index:
            assembled = []
            for idx in sorted(tool_calls_by_index):
                tc = tool_calls_by_index[idx]

                class _Fn:
                    pass

                fn = _Fn()
                fn.name = tc["name"]
                fn.arguments = tc["arguments"]

                class _TC:
                    pass

                t = _TC()
                t.id = tc["id"]
                t.function = fn
                assembled.append(t)
            msg.tool_calls = assembled
        else:
            msg.tool_calls = None

        class _Choice:
            pass

        choice = _Choice()
        choice.message = msg

        class _Resp:
            pass

        resp = _Resp()
        resp.choices = [choice]
        resp.usage = usage_info
        return resp
