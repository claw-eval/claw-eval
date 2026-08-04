#!/usr/bin/env python3
"""Accumulate GitHub traffic counters and dataset download totals into traffic.json.

GitHub only retains 14 days of traffic data, so daily rows are merged into a
persistent archive. Dataset download counts are already all-time totals and are
stored as latest snapshots, never accumulated.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = os.environ.get("TRAFFIC_REPO", "claw-eval/claw-eval")
HF_DATASET = os.environ.get("HF_DATASET", "claw-eval/Claw-Eval")
MS_DATASET = os.environ.get("MS_DATASET", "claw-eval/Claw-Eval")
UA = "claw-eval-traffic-collector"

warnings: list[str] = []


def fetch_json(url: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        # GitHub explains 403s in the response body ("Resource not accessible by
        # personal access token", "Must have push access to repository"); the
        # exception's str() carries only the status line.
        detail = exc.read().decode("utf-8", "replace").replace("\n", " ")[:300]
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} — {detail}") from None


def fetch_traffic(kind: str, token: str) -> dict[str, dict[str, int]]:
    data = fetch_json(
        f"https://api.github.com/repos/{REPO}/traffic/{kind}?per=day",
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    return {
        row["timestamp"][:10]: {"count": row["count"], "uniques": row["uniques"]}
        for row in data.get(kind, [])
    }


def fetch_hf_downloads() -> int:
    data = fetch_json(
        f"https://huggingface.co/api/datasets/{HF_DATASET}"
        "?expand%5B%5D=downloadsAllTime"
    )
    value = data.get("downloadsAllTime")
    if not isinstance(value, int):
        raise ValueError(f"downloadsAllTime missing or not an int: {value!r}")
    return value


def fetch_modelscope_downloads() -> int:
    data = fetch_json(f"https://modelscope.cn/api/v1/datasets/{MS_DATASET}")
    value = (data.get("Data") or {}).get("Downloads")
    if not isinstance(value, int):
        raise ValueError(f"Data.Downloads missing or not an int: {value!r}")
    return value


def humanize(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}k".replace(".0k", "k")
    return str(n)


def collect(archive: dict, token: str | None) -> dict:
    if token:
        kind = "fine-grained" if token.startswith("github_pat_") else "classic/other"
        print(f"::notice::traffic token type: {kind}", file=sys.stderr)
    for kind in ("clones", "views"):
        if not token:
            warnings.append(f"no token available, skipped {kind}")
            continue
        try:
            # Overwrite same-day rows: the current day's count is still partial.
            archive.setdefault(kind, {}).update(fetch_traffic(kind, token))
        except Exception as exc:
            warnings.append(f"{kind} fetch failed ({exc}); keeping previous values")

    for key, fetcher in (
        ("hf_downloads_all_time", fetch_hf_downloads),
        ("modelscope_downloads", fetch_modelscope_downloads),
    ):
        try:
            archive[key] = fetcher()
        except Exception as exc:
            warnings.append(f"{key} fetch failed ({exc}); keeping previous values")

    for kind in ("clones", "views"):
        rows = archive.get(kind, {})
        archive[f"{kind}_total"] = sum(r["count"] for r in rows.values())
        archive[f"{kind}_uniques_sum"] = sum(r["uniques"] for r in rows.values())

    archive["dataset_downloads_total"] = archive.get(
        "hf_downloads_all_time", 0
    ) + archive.get("modelscope_downloads", 0)

    dates = sorted({*archive.get("clones", {}), *archive.get("views", {})})
    if dates:
        archive["since"] = dates[0]
    archive["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return archive


def badges(archive: dict) -> dict[str, dict]:
    specs = {
        "badge-clones.json": ("clones", archive["clones_total"], "1f6feb"),
        "badge-views.json": ("views", archive["views_total"], "2da44e"),
        "badge-downloads.json": (
            "dataset downloads",
            archive["dataset_downloads_total"],
            "ff9d00",
        ),
    }
    return {
        name: {
            "schemaVersion": 1,
            "label": label,
            "message": humanize(total),
            "color": color,
            "cacheSeconds": 3600,
        }
        for name, (label, total, color) in specs.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="archive", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive_path = args.out_dir / "traffic.json"
    archive = json.loads(archive_path.read_text()) if archive_path.exists() else {}
    ordered_keys = (
        "updated_at",
        "since",
        "clones_total",
        "clones_uniques_sum",
        "views_total",
        "views_uniques_sum",
        "hf_downloads_all_time",
        "modelscope_downloads",
        "dataset_downloads_total",
        "clones",
        "views",
    )
    archive = collect(archive, os.environ.get("GITHUB_TOKEN"))
    archive = {k: archive[k] for k in ordered_keys if k in archive}

    for warning in warnings:
        print(f"::warning::{warning}", file=sys.stderr)

    if not archive.get("clones") and not archive.get("views"):
        print(
            "::error::no traffic data collected and no prior archive; "
            "GITHUB_TOKEN likely lacks access to the traffic API",
            file=sys.stderr,
        )
        return 1

    payload = {"traffic.json": archive, **badges(archive)}
    if args.dry_run:
        print(json.dumps(payload["traffic.json"], indent=2, sort_keys=False))
        for name in ("badge-clones.json", "badge-views.json", "badge-downloads.json"):
            print(f"{name}: {json.dumps(payload[name])}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        (args.out_dir / name).write_text(json.dumps(content, indent=2) + "\n")
    print(
        f"clones={archive['clones_total']} views={archive['views_total']} "
        f"downloads={archive['dataset_downloads_total']} since={archive.get('since')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
