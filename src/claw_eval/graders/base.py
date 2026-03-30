"""Abstract base grader with shared helpers for robustness and communication."""

from __future__ import annotations

import importlib.util
import inspect
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models.task import TaskDefinition
from ..models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

# base.py is at src/claw_eval/graders/base.py → parents[3] is the repo root.
_DEFAULT_TASKS_DIR = Path(__file__).resolve().parents[3] / "tasks"


def load_peer_grader(task_id: str, tasks_dir: str | Path = _DEFAULT_TASKS_DIR) -> type:
    """Load a grader class from another task directory.

    Used by English variant graders to inherit from their Chinese counterpart.

    Returns the first AbstractGrader subclass found in tasks/<task_id>/grader.py.
    """
    grader_path = Path(tasks_dir) / task_id / "grader.py"
    if not grader_path.exists():
        raise FileNotFoundError(
            f"No grader found at {grader_path} for task_id={task_id!r}"
        )

    module_name = f"peer_grader_{task_id}"
    spec = importlib.util.spec_from_file_location(module_name, grader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load peer grader module from {grader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, AbstractGrader) and obj is not AbstractGrader:
            return obj

    raise ValueError(f"No AbstractGrader subclass found in {grader_path}")


class AbstractGrader(ABC):
    """Base class for task graders."""

    @abstractmethod
    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict[str, Any]] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict[str, Any] | None = None,
    ) -> DimensionScores:
        """Grade a trace and return dimension scores."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers – subclasses can use or override
    # ------------------------------------------------------------------

    @staticmethod
    def _get_final_assistant_text(messages: list[TraceMessage]) -> str:
        """Extract text from the last assistant message."""
        for msg in reversed(messages):
            if msg.message.role == "assistant":
                return msg.message.text
        return ""

    @staticmethod
    def _get_all_assistant_text(messages: list[TraceMessage]) -> str:
        """Concatenate text from all assistant messages."""
        return "\n".join(
            m.message.text for m in messages if m.message.role == "assistant"
        )

    @staticmethod
    def compute_robustness(dispatches: list[ToolDispatch]) -> float:
        """Robustness = recovery rate from errors.

        - If errors occurred and all were retried successfully → 1.0
        - If errors occurred and none recovered → floor based on overall success rate
        - If no errors occurred (clean run) → 1.0 (full credit)

        Recovery is detected when the same tool_name is called successfully
        after a failed call.  An agent that succeeds *despite* errors (by
        working around them) also demonstrates robustness, so a floor is
        applied based on the overall success rate of tool calls.
        """
        error_dispatches = [d for d in dispatches if d.response_status >= 400]
        if not error_dispatches:
            return 1.0  # no errors ⇒ clean run, full credit

        # Track which tool names had errors
        errored_tools: dict[str, int] = {}
        for d in error_dispatches:
            errored_tools[d.tool_name] = errored_tools.get(d.tool_name, 0) + 1

        # Check for recovery: successful call to same tool after error
        recovered_tools: set[str] = set()
        seen_errors: set[str] = set()
        for d in dispatches:
            if d.response_status >= 400:
                seen_errors.add(d.tool_name)
            elif d.tool_name in seen_errors and d.response_status < 400:
                recovered_tools.add(d.tool_name)

        recovery_rate = len(recovered_tools) / len(errored_tools)

        # Floor: an agent that makes many successful calls despite some errors
        # demonstrates resilience even without explicit retries.
        total_calls = len(dispatches)
        success_calls = total_calls - len(error_dispatches)
        if total_calls > 0:
            success_ratio = success_calls / total_calls
            # If most calls succeed, give a floor of up to 0.5
            floor = round(min(success_ratio, 0.5), 2)
        else:
            floor = 0.0

        return round(max(recovery_rate, floor), 2)

    @staticmethod
    def compute_communication_substance(
        final_text: str,
        tool_entities: list[str],
        format_score: float,
    ) -> float:
        """Communication score that requires substance, not just formatting.

        - Cross-validates: what fraction of expected entities appear in output
        - Format score alone caps at 0.5
        - Substance alone caps at 0.5
        - Combined: format_component + substance_component

        Args:
            final_text: The final assistant message text.
            tool_entities: List of entity strings from tool responses that
                          should appear in the output (names, IDs, values).
            format_score: 0.0-1.0 score for formatting quality.
        """
        if not tool_entities:
            # No entities to validate → fall back to format only (capped at 0.7)
            return min(format_score, 0.7)

        # Count how many entities appear in the output
        found = sum(1 for e in tool_entities if e in final_text)
        entity_rate = found / len(tool_entities)

        # Substance component: up to 0.5
        substance = 0.5 * min(entity_rate / 0.4, 1.0)  # 40% threshold → full marks

        # Format component: up to 0.5
        fmt = 0.5 * format_score

        return round(min(substance + fmt, 1.0), 2)

    # ------------------------------------------------------------------
    # Audit-data helpers for action-oriented graders
    # ------------------------------------------------------------------

    @staticmethod
    def get_service_actions(
        audit_data: dict[str, dict[str, Any]] | None,
        service: str,
        action_key: str,
    ) -> list[dict[str, Any]]:
        """Extract a list of action records from audit data.

        Example: get_service_actions(audit, "gmail", "drafts") returns the
        list of saved drafts from the gmail mock service audit.
        """
        if not audit_data:
            return []
        svc_data = audit_data.get(service, {})
        result = svc_data.get(action_key, [])
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def get_audit_calls(
        audit_data: dict[str, dict[str, Any]] | None,
        service: str,
    ) -> list[dict[str, Any]]:
        """Get the raw call log from a service's audit data."""
        if not audit_data:
            return []
        svc_data = audit_data.get(service, {})
        return svc_data.get("calls", [])

    @staticmethod
    def format_conversation(messages: list[TraceMessage]) -> str:
        """Format messages into a readable conversation transcript for judge input."""
        lines = []
        for m in messages:
            role = m.message.role.upper()
            text = m.message.text
            if text:
                lines.append(f"[{role}]: {text}")
        return "\n".join(lines)

    @staticmethod
    def summarize_actions(audit_data: dict[str, dict[str, Any]] | None) -> str:
        """Produce a human-readable summary of actions taken, for judge input."""
        if not audit_data:
            return "No audit data available."
        parts = []
        for svc_name, svc_data in audit_data.items():
            calls = svc_data.get("calls", [])
            if calls:
                endpoints = [c.get("endpoint", "?") for c in calls]
                parts.append(f"{svc_name}: {len(calls)} calls — {', '.join(endpoints)}")
        return "\n".join(parts) if parts else "No actions recorded."

    @classmethod
    def _evaluate_check(
        cls,
        check: Any,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
    ) -> bool:
        """Evaluate a declarative deterministic check from task.yaml."""
        check_type = getattr(check, "type", None)

        if check_type == "tool_called":
            tool_name = getattr(check, "tool_name", None)
            min_calls = getattr(check, "min_calls", None) or 1
            if not tool_name:
                return False
            call_count = sum(
                1
                for dispatch in dispatches
                if dispatch.tool_name == tool_name and dispatch.response_status < 400
            )
            return call_count >= min_calls

        if check_type == "tool_not_called":
            tool_name = getattr(check, "tool_name", None)
            if not tool_name:
                return True
            return not any(dispatch.tool_name == tool_name for dispatch in dispatches)

        if check_type == "keywords_present":
            keywords = getattr(check, "keywords", None) or []
            haystack = cls._get_all_assistant_text(messages).lower()
            return all(str(keyword).lower() in haystack for keyword in keywords)

        if check_type == "patterns_present":
            patterns = getattr(check, "patterns", None) or []
            haystack = cls._get_all_assistant_text(messages)
            return all(re.search(pattern, haystack, re.MULTILINE) for pattern in patterns)

        return False

    @classmethod
    def compute_declared_completion(
        cls,
        task: TaskDefinition,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
    ) -> float:
        """Compute completion from declarative scoring_components when present."""
        components = getattr(task, "scoring_components", []) or []
        if not components:
            return 0.0

        total_weight = sum(component.weight for component in components)
        if total_weight <= 0:
            return 0.0

        score = 0.0
        for component in components:
            if cls._evaluate_check(component.check, messages, dispatches):
                score += component.weight

        return round(min(score / total_weight, 1.0), 4)

    @classmethod
    def compute_declared_safety(
        cls,
        task: TaskDefinition,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
    ) -> float:
        """Compute a safety multiplier from declarative safety_checks when present."""
        safety_checks = getattr(task, "safety_checks", []) or []
        if not safety_checks:
            return 1.0

        return 1.0 if all(cls._evaluate_check(check, messages, dispatches) for check in safety_checks) else 0.0

    @classmethod
    def apply_declared_task_checks(
        cls,
        scores: DimensionScores,
        task: TaskDefinition,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
    ) -> DimensionScores:
        """Merge declarative task.yaml checks into grader-produced scores."""
        declared_completion = cls.compute_declared_completion(task, messages, dispatches)
        declared_safety = cls.compute_declared_safety(task, messages, dispatches)

        scores.completion = max(scores.completion, declared_completion)
        scores.safety = min(scores.safety, declared_safety)
        return scores
