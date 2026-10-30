"""Regression tests for batch summary accounting after result merges.

Covers the bug reported in issue #61: ``--rerun-errors`` merged rerun results
into the previous batch without recomputing summary metrics, so the console
summary and ``batch_summary.json`` could report e.g. ``pass^3: 1/1`` alongside
``tasks=3``. The ``--continue`` merge path had the same inconsistency.
"""

from claw_eval.cli import _recompute_batch_counters


def _trial(score: float, passed: bool) -> dict:
    return {"task_score": score, "passed": passed}


def _task(task_id: str, trials: list[dict]) -> dict:
    return {"task_id": task_id, "trials": trials, "error": None}


def _errored(task_id: str, error: str = "boom") -> dict:
    return {"task_id": task_id, "trials": [], "error": error}


def test_mixed_batch_counters():
    results = [
        _task("A", [_trial(1.0, True), _trial(1.0, True), _trial(1.0, True)]),
        _task("B", [_trial(0.0, False), _trial(0.0, False), _trial(0.3, False)]),
        _errored("C"),
    ]
    score_sum, finished, n_pass_hat, n_pass_at = _recompute_batch_counters(results)
    assert finished == 3
    assert score_sum == (1.0 + 0.1)  # errored task contributes nothing
    assert n_pass_hat == 1  # only A passed all trials
    assert n_pass_at == 1  # only A passed any trial


def test_recovered_summary_matches_fresh_run():
    """Core #61 scenario: rerun only sees task C, but the merged summary must
    equal a fresh run over the full task set."""
    prev_results = [
        _task("A", [_trial(1.0, True)] * 3),
        _task("B", [_trial(0.0, False)] * 3),
        _errored("C", "api timeout"),
    ]
    rerun_results = [
        _task("C", [_trial(1.0, True)] * 3),
    ]
    merged = []
    rerun_by_id = {r["task_id"]: r for r in rerun_results}
    for prev in prev_results:
        merged.append(rerun_by_id.get(prev["task_id"], prev))

    fresh = _recompute_batch_counters(prev_results[:2] + rerun_results)
    recovered = _recompute_batch_counters(merged)

    assert recovered == fresh
    assert recovered == (2.0, 3, 2, 2)


def test_partial_pass_counts_toward_pass_at_only():
    results = [_task("A", [_trial(1.0, True), _trial(0.0, False), _trial(0.0, False)])]
    score_sum, finished, n_pass_hat, n_pass_at = _recompute_batch_counters(results)
    assert (finished, n_pass_hat, n_pass_at) == (1, 0, 1)
    assert abs(score_sum - 1.0 / 3.0) < 1e-9


def test_all_errored_batch_still_has_denominator():
    results = [_errored("A"), _errored("B")]
    score_sum, finished, n_pass_hat, n_pass_at = _recompute_batch_counters(results)
    assert (score_sum, finished, n_pass_hat, n_pass_at) == (0.0, 2, 0, 0)


def test_non_errored_task_with_empty_trials_is_skipped_safely():
    results = [_task("A", []), _task("B", [_trial(0.5, True)])]
    score_sum, finished, n_pass_hat, n_pass_at = _recompute_batch_counters(results)
    assert (score_sum, finished, n_pass_hat, n_pass_at) == (0.5, 2, 1, 1)


def test_empty_results():
    assert _recompute_batch_counters([]) == (0.0, 0, 0, 0)
