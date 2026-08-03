"""LiteLLM provider: route chat completions through litellm.completion()."""

from __future__ import annotations

import json
from typing import Any

from ...models.content import TextBlock, ToolUseBlock
from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import TokenUsage
from .openai_compat import _extract_text_tool_calls, _message_to_openai, _tool_spec_to_openai


class LiteLLMProvider:
    """LiteLLM SDK in-process; no proxy server."""

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
            raise ImportError('litellm is required. Install via `pip install "claw-eval[litellm]"`.') from exc

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
            kwargs["extra_body"] = dict(self.extra_body)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = [_tool_spec_to_openai(t) for t in tools]
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # litellm_kwargs (drop_params, num_retries, custom timeouts, ...) come
        # last so users can override anything above by setting it in config.
        kwargs.update(self.litellm_kwargs)

        response = litellm.completion(**kwargs)
        return _parse_response(response)


def _parse_response(response: Any) -> tuple[Message, TokenUsage]:
    """Convert a litellm.completion response into Message + TokenUsage.

    Mirrors OpenAICompatProvider._parse_response. Kept as a sibling function
    rather than reusing the method by inheritance because LiteLLMProvider is
    a standalone class.
    """
    if not response.choices:
        raise RuntimeError("Model returned empty choices (choices=None or [])")
    choice = response.choices[0]

    content_blocks: list[Any] = []
    if choice.message.content:
        if isinstance(choice.message.content, str):
            content_blocks.append(TextBlock(text=choice.message.content))
        elif isinstance(choice.message.content, list):
            text_chunks = []
            for part in choice.message.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            text_chunks.append(text)
                    continue
                if getattr(part, "type", None) == "text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        text_chunks.append(text)
            if text_chunks:
                content_blocks.append(TextBlock(text="\n".join(text_chunks)))

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            content_blocks.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))
    else:
        # Some providers emit pseudo tool markup in plain text.
        text_blocks = [b for b in content_blocks if b.type == "text"]
        if text_blocks:
            merged = "\n".join(b.text for b in text_blocks)
            cleaned, fallback_tools = _extract_text_tool_calls(merged)
            if fallback_tools:
                content_blocks = []
                if cleaned:
                    content_blocks.append(TextBlock(text=cleaned))
                content_blocks.extend(fallback_tools)

    usage = TokenUsage()
    if response.usage:
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
    return (
        Message(role="assistant", content=content_blocks, reasoning_content=reasoning),
        usage,
    )
