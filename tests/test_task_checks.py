from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from claw_eval.graders.base import AbstractGrader
from claw_eval.models.content import TextBlock
from claw_eval.models.message import Message
from claw_eval.models.task import DeterministicCheck, Prompt, SafetyCheck, ScoringComponent, TaskDefinition
from claw_eval.models.trace import DimensionScores, ToolDispatch, TraceMessage


def make_message(role: str, text: str) -> TraceMessage:
    return TraceMessage(
        trace_id="trace-1",
        message=Message(role=role, content=[TextBlock(text=text)]),
    )


def make_dispatch(tool_name: str, status: int = 200) -> ToolDispatch:
    return ToolDispatch(
        trace_id="trace-1",
        tool_use_id=f"call-{tool_name}",
        tool_name=tool_name,
        endpoint_url=f"http://localhost/{tool_name}",
        response_status=status,
    )


def make_task(*, scoring_components=None, safety_checks=None) -> TaskDefinition:
    return TaskDefinition(
        task_id="TTEST",
        task_name="Test Task",
        prompt=Prompt(text="test", language="en"),
        scoring_components=scoring_components or [],
        safety_checks=safety_checks or [],
    )


def test_declared_completion_reads_tool_calls_and_keywords() -> None:
    task = make_task(
        scoring_components=[
            ScoringComponent(
                name="tool usage",
                weight=0.4,
                check=DeterministicCheck(type="tool_called", tool_name="search", min_calls=2),
            ),
            ScoringComponent(
                name="keywords",
                weight=0.6,
                check=DeterministicCheck(type="keywords_present", keywords=["alpha", "beta"]),
            ),
        ]
    )
    messages = [make_message("assistant", "Alpha and beta are both present.")]
    dispatches = [make_dispatch("search"), make_dispatch("search")]

    declared_completion = AbstractGrader.compute_declared_completion(task, messages, dispatches)

    assert declared_completion == 1.0


def test_declared_safety_fails_when_forbidden_tool_called() -> None:
    task = make_task(
        safety_checks=[SafetyCheck(type="tool_not_called", tool_name="dangerous_tool")]
    )

    declared_safety = AbstractGrader.compute_declared_safety(
        task,
        [make_message("assistant", "done")],
        [make_dispatch("dangerous_tool")],
    )

    assert declared_safety == 0.0


def test_declared_task_checks_merge_into_grader_scores() -> None:
    task = make_task(
        scoring_components=[
            ScoringComponent(
                name="tool usage",
                weight=1.0,
                check=DeterministicCheck(type="tool_called", tool_name="lookup", min_calls=1),
            )
        ],
        safety_checks=[SafetyCheck(type="tool_not_called", tool_name="forbidden")],
    )
    scores = DimensionScores(completion=0.2, robustness=1.0, communication=0.0, safety=1.0)
    messages = [make_message("assistant", "done")]
    dispatches = [make_dispatch("lookup"), make_dispatch("forbidden")]

    merged = AbstractGrader.apply_declared_task_checks(scores, task, messages, dispatches)

    assert merged.completion == 1.0
    assert merged.safety == 0.0
