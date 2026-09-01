"""Best-effort logging for billed SERP requests during evaluation."""

from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from typing import Any


_lock = threading.Lock()
_path_cache: dict[tuple[str, str], str] = {}
_turn_counter: dict[tuple[str, str], int] = {}
_DEFAULT_ROLES = "serp"


def _enabled() -> bool:
    return os.getenv("CLAW_TOKEN_LOG", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _role_enabled(role: str) -> bool:
    raw = os.getenv("CLAW_TOKEN_LOG_ROLES", _DEFAULT_ROLES).strip().lower()
    if raw in {"all", "*"}:
        return True
    return role.lower() in {item.strip() for item in raw.split(",") if item.strip()}


def _monthly_root(base: str) -> str:
    leaf = os.path.basename(os.path.normpath(base))
    try:
        datetime.datetime.strptime(leaf, "%Y-%m")
        return base
    except ValueError:
        return os.path.join(base, datetime.date.today().strftime("%Y-%m"))


def build_token_log_path(api_key: str | None) -> str:
    today = str(datetime.date.today())
    suffix = api_key[-8:] if api_key else "no-key"
    cache_key = (suffix, today)
    cached = _path_cache.get(cache_key)
    if cached:
        return cached

    base = os.getenv("TOKEN_LOG_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "token_usage_log",
    )
    root = _monthly_root(base)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"token_usage_{suffix}_{today.replace('-', '')}.jsonl")
    _path_cache[cache_key] = path
    return path


def _extract_usage(usage: Any) -> tuple[int, int, int] | None:
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
    prompt = get("prompt_tokens")
    completion = get("completion_tokens")
    total = get("total_tokens")
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return prompt, completion, int(total if total is not None else prompt + completion)


def _flatten_messages(messages: Any) -> list[str]:
    if not isinstance(messages, (list, tuple)):
        return [str(messages)]
    flattened: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            flattened.append(str(message.get("role", "")) + str(message.get("content", "")))
        else:
            flattened.append(str(message))
    return flattened


def log_usage(
    *,
    role: str,
    model: str,
    usage: Any,
    messages: Any,
    answer: Any,
    api_key: str | None,
    task_id: str | None = None,
) -> None:
    """Append one JSONL record; logging failures never affect an evaluation."""
    if not _enabled() or not _role_enabled(role):
        return
    try:
        parsed = _extract_usage(usage)
        if parsed is None:
            return
        prompt_tokens, completion_tokens, total_tokens = parsed
        resolved_task = task_id or os.getenv("CLAW_TASK_ID") or None
        counter_key = (resolved_task or "-", role)
        with _lock:
            turn = _turn_counter.get(counter_key, 0) + 1
            _turn_counter[counter_key] = turn

        record = {
            "task_id": resolved_task,
            "turn": turn,
            "call_id": uuid.uuid4().hex[:12],
            "role": role,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt": _flatten_messages(messages),
            "answer": answer if isinstance(answer, str) else str(answer),
        }
        path = build_token_log_path(api_key)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _lock, open(path, "a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(line)
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                handle.write(line)
                handle.flush()
    except Exception:
        return
