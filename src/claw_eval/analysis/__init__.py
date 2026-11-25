"""Cross-run trend analysis over batch_results.json files."""

from .trend import (
    BatchRunSummary,
    BatchTrendReport,
    MetricTrend,
    RunTrendAnalyzer,
)

__all__ = [
    "BatchRunSummary",
    "BatchTrendReport",
    "MetricTrend",
    "RunTrendAnalyzer",
]
