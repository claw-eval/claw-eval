"""Compaction keeps tool exchanges intact and leaves a readable evaluation trace."""

import json

import pytest

from claw_eval.models.content import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claw_eval.models.message import Message
from claw_eval.models.task import Environment, Prompt, TaskDefinition
from claw_eval.models.trace import CompactEvent, TokenUsage
from claw_eval.runner.compact import do_auto_compact, micro_compact
from claw_eval.runner.loop import run_task
from claw_eval.runner.providers.openai_compat import _message_to_openai
from claw_eval.trace.reader import load_trace, read_events


def call(*ids):
    return Message(
        role="assistant", content=[ToolUseBlock(id=i, name="todo") for i in ids]
    )


def result(*ids):
    return Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id=i, content=[TextBlock(text="evidence " * 100)])
            for i in ids
        ],
    )


def prefix():
    return [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
    ]


def assert_complete_tool_wire(messages):
    """Check the actual provider serializer, not only canonical content blocks."""
    pending = set()
    for message in messages:
        converted = _message_to_openai(message)
        for wire in converted if isinstance(converted, list) else [converted]:
            if wire["role"] == "tool":
                assert wire["tool_call_id"] in pending
                pending.remove(wire["tool_call_id"])
            else:
                assert not pending, "tool call was separated from its results"
                pending.update(c["id"] for c in wire.get("tool_calls", []))
    assert not pending


class SummaryProvider:
    def __init__(self, response="summary", error=None):
        self.response = response
        self.error = error
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append(messages)
        assert_complete_tool_wire(messages)
        if self.error:
            raise self.error
        return Message(role="assistant", content=self.response), TokenUsage()


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_micro_compact_preserves_entire_recent_window(turns):
    messages = prefix()
    for i in range(turns):
        messages.extend(
            [
                call(str(i)),
                result(str(i)),
                Message(
                    role="user",
                    content=[
                        TextBlock(
                            text="[Visual content from tool results: 1 image(s)]"
                        ),
                        ImageBlock(data="AA==", mime_type="image/png"),
                    ],
                ),
            ]
        )
    before = [m.model_dump() for m in messages]
    micro_compact(messages, keep_recent=3, min_chars=20)
    assert [m.model_dump() for m in messages] == before


def test_micro_compact_changes_only_old_turns():
    messages = prefix() + [call("old"), result("old"), call("recent"), result("recent")]
    protected = [m.model_dump() for m in messages[:2] + messages[4:]]
    micro_compact(messages, keep_recent=1, min_chars=20)
    assert "truncated" in messages[3].content[0].content[0].text
    assert [m.model_dump() for m in messages[:2] + messages[4:]] == protected
    assert_complete_tool_wire(messages)


@pytest.mark.parametrize("split", [4, 5, 6])
@pytest.mark.parametrize("budget", ["messages", "tokens"])
def test_auto_compact_keeps_parallel_exchange_on_one_side(split, budget):
    messages = prefix() + [
        Message(role="assistant", content="older conclusion"),
        call("a", "b"),
        result("b"),
        result("a"),
        Message(role="assistant", content="latest conclusion"),
    ]
    provider = SummaryProvider()
    if budget == "messages":
        kwargs = {"keep_recent_on_summary": len(messages) - split, "protect_tokens": 0}
    else:
        # Result messages contribute 225 tokens each; latest text contributes 4.
        threshold = {4: 230, 5: 5, 6: 1}[split]
        kwargs = {"keep_recent_on_summary": 0, "protect_tokens": threshold}
    compacted = do_auto_compact(messages, provider, **kwargs)
    assert len(provider.requests) == 1
    assert compacted[1].text == "[Context Summary]\nsummary"
    assert_complete_tool_wire(provider.requests[0])
    assert_complete_tool_wire(compacted)
    first_kept = 3 if split < 6 else 6
    assert compacted[2:] == messages[first_kept:]


@pytest.mark.parametrize(
    "response,error", [("", None), ("  \n", None), ("unused", RuntimeError("offline"))]
)
def test_summary_failure_preserves_exact_history(response, error):
    messages = prefix() + [
        Message(role="assistant", content="older"),
        Message(role="user", content="latest"),
    ]
    before = [m.model_dump() for m in messages]
    provider = SummaryProvider(response, error)
    compacted = do_auto_compact(
        messages, provider, keep_recent_on_summary=1, protect_tokens=0
    )
    assert compacted is messages
    assert [m.model_dump() for m in messages] == before


def test_no_safe_removable_prefix_does_not_call_summary_model():
    messages = prefix() + [call("a", "b"), result("a"), result("b")]
    provider = SummaryProvider()
    assert (
        do_auto_compact(messages, provider, keep_recent_on_summary=1, protect_tokens=0)
        is messages
    )
    assert provider.requests == []


def test_pending_manual_compact_call_is_retained_for_result():
    messages = prefix() + [
        call("previous"),
        result("previous"),
        call("compact", "other"),
    ]
    provider = SummaryProvider()
    compacted = do_auto_compact(
        messages, provider, keep_recent_on_summary=1, protect_tokens=0
    )
    assert compacted[1].text == "[Context Summary]\nsummary"
    assert compacted[-1] is messages[-1]
    assert_complete_tool_wire(compacted + [result("compact", "other")])


def test_manual_compaction_in_real_loop_produces_gradeable_trace(tmp_path):
    class LoopProvider(SummaryProvider):
        model_id = "offline-test"

        def __init__(self):
            super().__init__()
            self.turn = 0
            self.agent_requests = []

        def chat(self, messages, tools=None):
            if tools is None:
                return super().chat(messages)
            assert_complete_tool_wire(messages)
            self.agent_requests.append([m.model_copy(deep=True) for m in messages])
            self.turn += 1
            if self.turn == 1:
                response = call("previous")
            elif self.turn == 2:
                response = Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="compact", name="compact"),
                        ToolUseBlock(id="other", name="todo"),
                    ],
                )
            else:
                response = Message(role="assistant", content="finished")
            return response, TokenUsage()

    task = TaskDefinition(
        task_id="T000_compact",
        task_name="offline compaction",
        prompt=Prompt(text="task"),
        environment=Environment(
            enable_compact=True,
            enable_todo=True,
            max_turns=4,
            compact_max_auto_compacts=0,
            compact_keep_recent_on_summary=1,
            compact_protect_tokens=0,
        ),
    )
    provider = LoopProvider()
    path = run_task(task, provider, trace_dir=tmp_path)
    raw_events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(event["type"] == "compact" for event in raw_events)
    assert len(provider.requests) == 1
    assert "[Context Summary]" in provider.agent_requests[-1][1].text
    events = list(read_events(path))
    assert any(isinstance(event, CompactEvent) for event in events)
    start, messages, dispatches, media, end, audit = load_trace(path)
    assert start.task_id == task.task_id
    assert end.failure_modes == []
    assert end.total_turns == 3
    assert messages[-1].message.text == "finished"
    assert not dispatches and not media and not audit
