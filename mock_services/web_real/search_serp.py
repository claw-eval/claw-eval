"""Search SERP through yibuapi's Serper-compatible endpoint.

The public ``search_serp`` return shape stays compatible with the web_real
service while the upstream request uses POST JSON plus Bearer authentication.
"""

from __future__ import annotations

import os
import re
import requests

SERP_API_URL = os.getenv("SERP_API_URL", "https://yibuapi.com/serper/search")
SERP_DEV_KEY = (
    os.getenv("SERP_API_KEY")
    or os.getenv("SERP_DEV_KEY")
    or os.getenv("YIBUAPI_KEY", "")
)


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh", "cn"
    return "en", "us"


def _log_serp_request(query: str, status: int) -> None:
    """Record a Serper request without making logging affect search."""
    try:
        from claw_eval.token_log import log_usage

        log_usage(
            role="serp",
            model="serper/search",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            messages=[{"role": "user", "content": query}],
            answer=f"HTTP {status}",
            api_key=SERP_DEV_KEY,
        )
    except Exception:
        pass


def search_serp(
    query: str,
    timeout: int = 20,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google via SERP API and return extracted results.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset.

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on error.
            output (list[dict]): List of result dicts with keys:
                title, link, snippet, date, query.
    """
    hl, gl = _detect_language(query)
    n = min(max(num, 1), 10)
    page = max(1, ((max(start, 1) - 1) // n) + 1)
    body = {
        "q": query,
        "num": n,
        "hl": hl,
        "gl": gl,
        "page": page,
    }
    headers = {
        "Authorization": f"Bearer {SERP_DEV_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(SERP_API_URL, headers=headers, json=body, timeout=timeout)
        _log_serp_request(query, resp.status_code)
        if raw_save_path and resp.status_code == 200:
            os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
            with open(raw_save_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        if resp.status_code != 200:
            return {"status": resp.status_code, "output": []}
        data = resp.json()
        results = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "query": query,
            }
            for item in data.get("organic", [])
        ]
        return {"status": resp.status_code, "output": results}
    except Exception as e:
        return {"status": -1, "output": []}


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])
