from types import SimpleNamespace

from claw_eval.runner.providers.openai_compat import OpenAICompatProvider


def _response_with_usage(usage):
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_parse_response_preserves_openai_cached_tokens():
    provider = OpenAICompatProvider()
    response = _response_with_usage(
        SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        )
    )

    _, usage = provider._parse_response(response)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 40
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0


def test_parse_response_preserves_anthropic_cache_tokens():
    provider = OpenAICompatProvider()
    response = _response_with_usage(
        SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            cache_creation_input_tokens=15,
            cache_read_input_tokens=35,
        )
    )

    _, usage = provider._parse_response(response)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 35
    assert usage.cache_creation_input_tokens == 15
    assert usage.cache_read_input_tokens == 35
