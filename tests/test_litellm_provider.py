"""Unit tests for the LiteLLM provider and the make_provider factory.

Tests run without external dependencies by mocking ``litellm.completion``.
Live integration is exercised via a separate scratch script.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from claw_eval.config import ModelConfig
from claw_eval.models.content import TextBlock, ToolUseBlock
from claw_eval.models.message import Message
from claw_eval.models.tool import ToolSpec
from claw_eval.runner.providers import OpenAICompatProvider, make_provider
from claw_eval.runner.providers.litellm import LiteLLMProvider


# ---------------------------------------------------------------------------
# Mocked litellm.completion responses (OpenAI shape)
# ---------------------------------------------------------------------------


def _completion_response(content, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _user(text):
    return Message(role="user", content=[TextBlock(text=text)])


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestMakeProvider(unittest.TestCase):
    def test_default_is_openai_compat(self):
        cfg = ModelConfig(model_id="openai/gpt-4o-mini")
        self.assertEqual(cfg.provider, "openai_compat")
        provider = make_provider(cfg)
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertNotIsInstance(provider, LiteLLMProvider)

    def test_litellm_dispatch(self):
        cfg = ModelConfig(
            provider="litellm",
            model_id="anthropic/claude-3-5-sonnet-20241022",
        )
        provider = make_provider(cfg)
        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual(provider.model_id, "anthropic/claude-3-5-sonnet-20241022")

    def test_unknown_provider_raises(self):
        cfg = ModelConfig(provider="bogus")
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            make_provider(cfg)

    def test_litellm_kwargs_pass_through(self):
        cfg = ModelConfig(
            provider="litellm",
            model_id="openai/gpt-4o-mini",
            litellm_kwargs={"num_retries": 7},
        )
        provider = make_provider(cfg)
        self.assertEqual(provider.litellm_kwargs.get("num_retries"), 7)
        # default still preserved
        self.assertIs(provider.litellm_kwargs.get("drop_params"), True)


# ---------------------------------------------------------------------------
# LiteLLMProvider initialization
# ---------------------------------------------------------------------------


class TestLiteLLMProviderInit(unittest.TestCase):
    def test_drop_params_default_on(self):
        p = LiteLLMProvider(model_id="openai/gpt-4o")
        self.assertIs(p.litellm_kwargs.get("drop_params"), True)

    def test_drop_params_can_be_disabled(self):
        p = LiteLLMProvider(
            model_id="openai/gpt-4o",
            litellm_kwargs={"drop_params": False},
        )
        self.assertIs(p.litellm_kwargs.get("drop_params"), False)

    def test_user_kwargs_merge_with_default(self):
        p = LiteLLMProvider(
            model_id="openai/gpt-4o",
            litellm_kwargs={"timeout": 120},
        )
        self.assertEqual(p.litellm_kwargs.get("timeout"), 120)
        self.assertIs(p.litellm_kwargs.get("drop_params"), True)

    def test_skips_openai_client_setup(self):
        # Parent OpenAICompatProvider sets self.client; we deliberately don't.
        p = LiteLLMProvider(model_id="openai/gpt-4o")
        self.assertFalse(hasattr(p, "client"))


# ---------------------------------------------------------------------------
# chat() routing
# ---------------------------------------------------------------------------


class TestChatRouting(unittest.TestCase):
    def test_chat_routes_through_litellm(self):
        p = LiteLLMProvider(model_id="anthropic/claude-3-5-sonnet-20241022")
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _completion_response("4")

        with patch("litellm.completion", side_effect=fake_completion):
            response_msg, usage = p.chat([_user("What is 2+2?")])

        self.assertEqual(captured["model"], "anthropic/claude-3-5-sonnet-20241022")
        self.assertIs(captured["drop_params"], True)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 5)
        text_blocks = [b for b in response_msg.content if b.type == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0].text, "4")

    def test_chat_forwards_credentials(self):
        p = LiteLLMProvider(
            model_id="anthropic/claude-3-5-sonnet-20241022",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        )
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _completion_response("ok")

        with patch("litellm.completion", side_effect=fake_completion):
            p.chat([_user("hi")])

        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["api_base"], "https://example.invalid/v1")

    def test_chat_no_credentials_omits_keys(self):
        p = LiteLLMProvider(model_id="anthropic/claude-3-5-sonnet-20241022")
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _completion_response("ok")

        with patch("litellm.completion", side_effect=fake_completion):
            p.chat([_user("hi")])

        self.assertNotIn("api_key", captured)
        self.assertNotIn("api_base", captured)

    def test_chat_forwards_tools(self):
        p = LiteLLMProvider(model_id="anthropic/claude-3-5-sonnet-20241022")
        tool = ToolSpec(
            name="get_weather",
            description="Get weather",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _completion_response("call the tool")

        with patch("litellm.completion", side_effect=fake_completion):
            p.chat([_user("weather?")], tools=[tool])

        self.assertEqual(len(captured["tools"]), 1)
        self.assertEqual(captured["tools"][0]["function"]["name"], "get_weather")

    def test_chat_propagates_tool_calls(self):
        p = LiteLLMProvider(model_id="anthropic/claude-3-5-sonnet-20241022")
        tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="get_weather", arguments='{"city":"Tokyo"}'
            ),
        )

        with patch(
            "litellm.completion",
            return_value=_completion_response("", tool_calls=[tool_call]),
        ):
            response_msg, _ = p.chat([_user("weather?")])

        tool_uses = [b for b in response_msg.content if b.type == "tool_use"]
        self.assertEqual(len(tool_uses), 1)
        self.assertEqual(tool_uses[0].name, "get_weather")
        self.assertEqual(tool_uses[0].input, {"city": "Tokyo"})


if __name__ == "__main__":
    unittest.main()
