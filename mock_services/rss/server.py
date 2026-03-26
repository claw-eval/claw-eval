"""Mock RSS API service for agent evaluation (FastAPI on port 9109)."""

from __future__ import annotations

import json
import copy
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mock RSS API")

from mock_services._base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "RSS_FIXTURES",
    str(Path(__file__).resolve().parent.parent.parent / "tasks" / "T21zh_newsletter_curation" / "fixtures" / "rss" / "articles.json"),
))

_articles: list[dict[str, Any]] = []
_audit_log: list[dict[str, Any]] = []
_published: list[dict[str, Any]] = []


def _load_fixtures() -> None:
    global _articles
    with open(FIXTURES_PATH, encoding='utf-8') as f:
        _articles = json.load(f)


_load_fixtures()


def _log_call(endpoint: str, request_body: dict[str, Any], response_body: Any) -> None:
    _audit_log.append({
        "endpoint": endpoint,
        "request_body": request_body,
        "response_body": response_body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


class ListFeedsRequest(BaseModel):
    category: str | None = None


class ListArticlesRequest(BaseModel):
    source: str | None = None
    category: str | None = None
    max_results: int = 20


class GetArticleRequest(BaseModel):
    article_id: str


class PublishRequest(BaseModel):
    title: str
    content: str
    recipients: list[str] = Field(default_factory=list)


@app.post("/rss/feeds")
def list_feeds(req: ListFeedsRequest | None = None) -> dict[str, Any]:
    if req is None:
        req = ListFeedsRequest()
    # Derive feeds from articles
    feeds = {}
    for a in _articles:
        src = a["source"]
        cat = a["category"]
        if req.category and cat != req.category:
            continue
        if src not in feeds:
            feeds[src] = {"source": src, "categories": set(), "article_count": 0}
        feeds[src]["categories"].add(cat)
        feeds[src]["article_count"] += 1
    result = [
        {"source": v["source"], "categories": list(v["categories"]), "article_count": v["article_count"]}
        for v in feeds.values()
    ]
    resp = {"feeds": result, "total": len(result)}
    _log_call("/rss/feeds", req.model_dump(), resp)
    return resp


@app.post("/rss/articles")
def list_articles(req: ListArticlesRequest | None = None) -> dict[str, Any]:
    if req is None:
        req = ListArticlesRequest()
    results = []
    for a in _articles:
        if req.source and a["source"] != req.source:
            continue
        if req.category and a["category"] != req.category:
            continue
        results.append({
            "article_id": a["article_id"],
            "title": a["title"],
            "source": a["source"],
            "category": a["category"],
            "published_at": a["published_at"],
            "summary": a["summary"],
            "word_count": a["word_count"],
        })
    results = results[:req.max_results]
    resp = {"articles": results, "total": len(results)}
    _log_call("/rss/articles", req.model_dump(), resp)
    return resp


@app.post("/rss/articles/get")
def get_article(req: GetArticleRequest) -> dict[str, Any]:
    for a in _articles:
        if a["article_id"] == req.article_id:
            resp = copy.deepcopy(a)
            _log_call("/rss/articles/get", req.model_dump(), resp)
            return resp
    resp = {"error": f"Article {req.article_id} not found"}
    _log_call("/rss/articles/get", req.model_dump(), resp)
    return resp


@app.post("/rss/publish")
def publish_newsletter(req: PublishRequest) -> dict[str, Any]:
    record = {
        "title": req.title,
        "content": req.content,
        "recipients": req.recipients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _published.append(record)
    resp = {"status": "published", "record": record}
    _log_call("/rss/publish", req.model_dump(), resp)
    return resp


@app.get("/rss/audit")
def get_audit() -> dict[str, Any]:
    return {"calls": _audit_log, "published": _published}


@app.post("/rss/reset")
def reset_state() -> dict[str, str]:
    global _audit_log, _published
    _audit_log = []
    _published = []
    _load_fixtures()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9109")))
