from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    file_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


def _assert_normalized_result(result: dict[str, Any], query: str) -> None:
    assert result["status"] == 200
    assert result["output"] == [
        {
            "title": "Example",
            "link": "https://example.com",
            "snippet": "Summary",
            "date": "2026-03-23",
            "query": query,
        }
    ]


def test_web_real_supports_novada_response(monkeypatch) -> None:
    module = _load_module("web_real_search_serp", "mock_services/web_real/search_serp.py")
    captured: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, str], timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "data": {
                    "organic_results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "description": "Summary",
                            "date": "2026-03-23",
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module, "SERP_API_URL", "https://scraperapi.novada.com/search")
    monkeypatch.setattr(module, "SERP_DEV_KEY", "test-key")

    result = module.search_serp("python", timeout=12)

    assert captured["url"] == "https://scraperapi.novada.com/search"
    assert captured["params"]["q"] == "python"
    assert "query" not in captured["params"]
    assert captured["timeout"] == 12
    _assert_normalized_result(result, "python")


def test_web_real_supports_standard_scraperapi_response(monkeypatch) -> None:
    module = _load_module("web_real_search_serp_standard", "mock_services/web_real/search_serp.py")
    captured: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, str], timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse(
            200,
            {
                "organic_results": [
                    {
                        "title": "Example",
                        "link": "https://example.com",
                        "snippet": "Summary",
                        "date": "2026-03-23",
                    }
                ]
            },
        )

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module, "SERP_API_URL", "https://api.scraperapi.com/structured/google/search")
    monkeypatch.setattr(module, "SERP_DEV_KEY", "test-key")

    result = module.search_serp("python", timeout=8)

    assert captured["url"] == "https://api.scraperapi.com/structured/google/search"
    assert captured["params"]["query"] == "python"
    assert "q" not in captured["params"]
    assert captured["timeout"] == 8
    _assert_normalized_result(result, "python")


def test_web_real_handles_http_200_error_payload(monkeypatch) -> None:
    module = _load_module("web_real_search_serp_error", "mock_services/web_real/search_serp.py")

    def fake_get(url: str, params: dict[str, str], timeout: int) -> DummyResponse:
        return DummyResponse(200, {"code": 402, "msg": "Api Key error: User has no permission"})

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module, "SERP_API_URL", "https://api.scraperapi.com/structured/google/search")
    monkeypatch.setattr(module, "SERP_DEV_KEY", "test-key")

    result = module.search_serp("python")

    assert result == {"status": 200, "output": []}


def test_web_real_injection_supports_standard_scraperapi_response(monkeypatch) -> None:
    module = _load_module(
        "web_real_injection_search_serp_standard",
        "mock_services/web_real_injection/search_serp.py",
    )

    def fake_get(url: str, params: dict[str, str], timeout: int) -> DummyResponse:
        return DummyResponse(
            200,
            {
                "organic_results": [
                    {
                        "title": "Example",
                        "link": "https://example.com",
                        "snippet": "Summary",
                        "date": "2026-03-23",
                    }
                ]
            },
        )

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module, "SERP_API_URL", "https://api.scraperapi.com/structured/google/search")
    monkeypatch.setattr(module, "SERP_DEV_KEY", "test-key")

    result = module.search_serp("python")

    _assert_normalized_result(result, "python")
