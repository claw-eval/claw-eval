"""Tests for RunTrendAnalyzer (issue #27 — cross-run batch trend detection)."""

import json

from claw_eval.analysis import (
    BatchTrendReport,
    BatchRunSummary,
    MetricTrend,
    RunTrendAnalyzer,
)


def _task(task_id: str, avg: float, passed: bool) -> dict:
    return {
        "task_id": task_id,
        "trials": [{"task_score": avg, "passed": passed}],
        "error": None,
        "avg_score": avg,
        "avg_passed": passed,
    }


def _errored(task_id: str) -> dict:
    return {"task_id": task_id, "trials": [], "error": "boom"}


def _write_batch(path, results):
    path.mkdir(parents=True, exist_ok=True)
    (path / "batch_results.json").write_text(json.dumps(results))
    return path


def test_degrading_trend_detected(tmp_path):
    means = [0.81, 0.78, 0.74, 0.70]  # ~ -0.036/run
    dirs = [_write_batch(tmp_path / f"run{i:02d}",
                         [_task("A", m, m >= 0.75), _task("B", m - 0.05, m - 0.05 >= 0.75)])
            for i, m in enumerate(means, start=1)]
    report = RunTrendAnalyzer(dirs).analyze()
    assert isinstance(report, BatchTrendReport)
    assert report.window == 4
    assert report.mean_score_trend.direction == "degrading"
    assert report.mean_score_trend.slope < -0.01
    assert report.pass_rate_trend.direction == "degrading"  # 2/2 -> 1/2 -> 0/2 -> 0/2
    assert report.any_regression is True


def test_improving_and_stable_directions(tmp_path):
    up = [_write_batch(tmp_path / f"up{i}", [_task("A", 0.60 + 0.05 * i, False)])
          for i in range(5)]
    rep_up = RunTrendAnalyzer(up).analyze()
    assert rep_up.mean_score_trend.direction == "improving"
    assert rep_up.any_regression is False

    flat = [_write_batch(tmp_path / f"flat{i}", [_task("A", 0.800 + 0.001 * i, True)])
            for i in range(4)]
    rep_flat = RunTrendAnalyzer(flat).analyze()
    assert rep_flat.mean_score_trend.direction == "stable"
    assert rep_flat.any_regression is False


def test_window_truncates_to_last_n(tmp_path):
    dirs = [_write_batch(tmp_path / f"run{i:02d}", [_task("A", 0.5 + 0.01 * i, False)])
            for i in range(8)]
    report = RunTrendAnalyzer(dirs, window=3).analyze()
    assert report.window == 3
    assert [r.trace_dir for r in report.runs] == [str(d) for d in dirs[-3:]]


def test_error_tasks_and_missing_files_skipped(tmp_path):
    good = _write_batch(tmp_path / "good", [
        _task("A", 0.9, True),
        _errored("B"),  # excluded from both mean and denominator
    ])
    empty = _write_batch(tmp_path / "empty", [])  # no valid scores -> skipped
    (tmp_path / "nogfile").mkdir()  # no batch_results.json at all
    report = RunTrendAnalyzer([good, empty, tmp_path / "nogfile", good]).analyze()
    assert report.window == 2  # empty dir contributes nothing
    assert all(r.task_count == 1 for r in report.runs)
    assert report.any_regression is False  # identical runs -> stable


def test_trial_score_fallback_when_avg_score_missing(tmp_path):
    # Older schema: only per-trial task_score, no top-level avg_score/avg_passed
    legacy = {
        "task_id": "A",
        "trials": [{"task_score": 0.9, "passed": True}],
        "error": None,
    }
    d1 = _write_batch(tmp_path / "d1", [dict(legacy)])
    d2 = _write_batch(tmp_path / "d2", [dict(legacy)])
    report = RunTrendAnalyzer([d1, d2]).analyze()
    assert report.runs[0].mean_score == 0.9
    assert report.runs[0].pass_rate == 1.0


def test_fewer_than_two_runs_means_no_regression(tmp_path):
    d = _write_batch(tmp_path / "only", [_task("A", 0.1, False)])
    report = RunTrendAnalyzer([d]).analyze()
    assert report.window == 1
    assert report.mean_score_trend is None
    assert report.pass_rate_trend is None
    assert report.any_regression is False


def test_no_directories_found(tmp_path):
    report = RunTrendAnalyzer([tmp_path / "does-not-exist"]).analyze()
    assert report.runs == []
    assert report.any_regression is False


def test_cli_trend_subcommand(tmp_path, capsys):
    """End-to-end through cli.main: table printed, exit code reflects regression."""
    import pytest
    from claw_eval.cli import main

    down = [_write_batch(tmp_path / f"run{i}",
                         [_task("A", 0.85 - 0.06 * i, i == 0)])
            for i in range(3)]
    argv = ["trend", *[str(d) for d in down]]
    main(argv)
    out = capsys.readouterr().out
    assert "Batch trend analysis" in out
    assert "DEGRADING" in out

    with pytest.raises(SystemExit) as exc:
        main(argv + ["--exit-on-regression"])
    assert exc.value.code == 1

    # Non-regressing run must not raise
    up = [_write_batch(tmp_path / f"up{i}", [_task("A", 0.50 + 0.05 * i, False)])
          for i in range(3)]
    main(["trend", *[str(d) for d in up], "--exit-on-regression"])
    assert "No regression" in capsys.readouterr().out
