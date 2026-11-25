"""OLS trend detection across sequential batch runs (issue #27).

Reads ordered ``batch_results.json`` files produced by ``claw-eval batch``
and fits per-run mean score / pass rate over run index. Degrading trends on
either metric raise ``any_regression`` so CI can gate on ``--exit-on-regression``.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Pass threshold mirrors models.scoring.is_pass default; duplicated as a plain
# constant to keep this module importable without touching scoring internals.
PASS_THRESHOLD = 0.75

# Slope (points/run) at or below this counts as degrading; at or above the
# positive mirror counts as improving. Matches the spec in issue #27.
STABLE_BAND = 0.01


@dataclass
class BatchRunSummary:
    trace_dir: str
    mean_score: float
    pass_rate: float  # fraction with task score >= PASS_THRESHOLD
    task_count: int


@dataclass
class MetricTrend:
    slope: float  # OLS slope (points/run); positive = improving
    direction: str  # "improving" | "degrading" | "stable"


@dataclass
class BatchTrendReport:
    window: int
    runs: list[BatchRunSummary] = field(default_factory=list)
    mean_score_trend: MetricTrend | None = None
    pass_rate_trend: MetricTrend | None = None
    any_regression: bool = False


def _task_score(r: dict) -> float | None:
    """Extract a per-task score, tolerating both result schemas.

    ``cmd_batch`` writes ``avg_score`` on non-errored tasks; older or partial
    files may only carry per-trial ``task_score`` values, so fall back to the
    trial mean when the top-level key is absent.
    """
    if r.get("error"):
        return None
    if "avg_score" in r:
        return float(r["avg_score"])
    trials = [t for t in r.get("trials", []) if "task_score" in t]
    if not trials:
        return None
    return statistics.mean(float(t["task_score"]) for t in trials)


def _task_passed(r: dict, score: float) -> bool:
    if "avg_passed" in r:
        return bool(r["avg_passed"])
    return score >= PASS_THRESHOLD


class RunTrendAnalyzer:
    def __init__(self, batch_dirs: Sequence[str | Path], window: int = 10):
        self.batch_dirs = [Path(d) for d in batch_dirs][-window:]

    def analyze(self) -> BatchTrendReport:
        runs: list[BatchRunSummary] = []
        for d in self.batch_dirs:
            p = d / "batch_results.json"
            if not p.exists():
                continue
            results = json.loads(p.read_text())
            scores = []
            passed = 0
            for r in results:
                s = _task_score(r)
                if s is None:
                    continue
                scores.append(s)
                if _task_passed(r, s):
                    passed += 1
            if not scores:
                continue
            runs.append(
                BatchRunSummary(
                    trace_dir=str(d),
                    mean_score=statistics.mean(scores),
                    pass_rate=passed / len(scores),
                    task_count=len(scores),
                )
            )

        report = BatchTrendReport(window=len(runs), runs=runs)
        if len(runs) < 2:
            # A trend needs at least two runs; report no regression.
            return report

        def _ols(values: list[float]) -> MetricTrend:
            xs = list(range(len(values)))
            slope = statistics.linear_regression(xs, values).slope
            direction = (
                "improving"
                if slope > STABLE_BAND
                else ("degrading" if slope < -STABLE_BAND else "stable")
            )
            return MetricTrend(slope=round(slope, 4), direction=direction)

        report.mean_score_trend = _ols([r.mean_score for r in runs])
        report.pass_rate_trend = _ols([r.pass_rate for r in runs])
        report.any_regression = (
            report.mean_score_trend.direction == "degrading"
            or report.pass_rate_trend.direction == "degrading"
        )
        return report
