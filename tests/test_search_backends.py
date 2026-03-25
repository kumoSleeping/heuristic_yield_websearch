from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest
from unittest import mock

search_jina_ai = importlib.import_module("core.search_jina_ai")
web_search = importlib.import_module("core.web_search")


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload
        self.text = ""
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, recorder: dict[str, object], *args, **kwargs):
        del args, kwargs
        self._recorder = recorder

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    async def get(self, url: str, *, params=None, headers=None):
        self._recorder["url"] = url
        self._recorder["params"] = params
        self._recorder["headers"] = dict(headers or {})
        return _FakeResponse(
            {
                "data": [
                    {
                        "title": "Example",
                        "url": "https://example.com/result",
                        "snippet": "snippet",
                    }
                ]
            }
        )


class JinaSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_jina_search_uses_path_query_not_q_param(self) -> None:
        recorder: dict[str, object] = {}

        class _FakeHttpxModule:
            def AsyncClient(self, *args, **kwargs):
                return _FakeAsyncClient(recorder, *args, **kwargs)

        async def _fake_usage_meta(*args, **kwargs):
            del args, kwargs
            return {}

        with (
            mock.patch.object(search_jina_ai, "_load_httpx_module", return_value=_FakeHttpxModule()),
            mock.patch.object(search_jina_ai, "_build_usage_meta", side_effect=_fake_usage_meta),
        ):
            payload = await search_jina_ai.jina_ai_search("葱喵Bot site:x.com/test")

        self.assertEqual(
            recorder.get("url"),
            "https://s.jina.ai/%E8%91%B1%E5%96%B5Bot%20site%3Ax.com%2Ftest",
        )
        self.assertIsNone(recorder.get("params"))
        self.assertEqual(payload["results"][0]["url"], "https://example.com/result")

    def test_jina_page_extract_keeps_authorization_when_configured(self) -> None:
        headers = search_jina_ai._jina_headers(
            config={
                "tools": {
                    "config": {
                        "jina_ai": {
                            "page_extract": {
                                "headers": {
                                    "Authorization": "Bearer test-token",
                                }
                            }
                        }
                    }
                }
            },
            capability="page_extract",
            default_accept="text/plain",
        )

        self.assertEqual(headers.get("Authorization"), "Bearer test-token")
        self.assertEqual(headers.get("X-Engine"), "browser")


class WebSearchFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_falls_back_to_ddgs_on_primary_connect_error(self) -> None:
        async def _raise_connect_error(**kwargs):
            del kwargs
            raise RuntimeError("ConnectError")

        async def _return_ddgs_result(**kwargs):
            del kwargs
            return [
                {
                    "title": "Fallback Result",
                    "url": "https://example.com/fallback",
                    "snippet": "ok",
                }
            ]

        jina_handler = SimpleNamespace(provider="jina_ai", target="jina", callable=_raise_connect_error)
        ddgs_handler = SimpleNamespace(provider="ddgs", target="ddgs", callable=_return_ddgs_result)

        def _fake_resolve(config, capability, selection=None):
            del config
            if capability == "search" and selection is None:
                return [jina_handler]
            if capability == "search" and selection == "ddgs":
                return [ddgs_handler]
            return []

        with mock.patch.object(web_search, "resolve_tool_handlers", side_effect=_fake_resolve):
            suite = web_search.WebToolSuite(config={"tools": {"use": {"search": "jina_ai"}}})
            rows, meta = await suite._search_text("葱喵Bot", None, 5)

        self.assertEqual(meta, {})
        self.assertEqual(rows[0]["provider"], "ddgs")
        self.assertEqual(rows[0]["url"], "https://example.com/fallback")

    async def test_search_falls_back_to_ddgs_on_primary_timeout(self) -> None:
        async def _sleep_forever_enough(**kwargs):
            del kwargs
            await __import__("asyncio").sleep(0.05)
            return []

        async def _return_ddgs_result(**kwargs):
            del kwargs
            return [
                {
                    "title": "Fallback Result",
                    "url": "https://example.com/fallback-timeout",
                    "snippet": "ok",
                }
            ]

        jina_handler = SimpleNamespace(provider="jina_ai", target="jina", callable=_sleep_forever_enough)
        ddgs_handler = SimpleNamespace(provider="ddgs", target="ddgs", callable=_return_ddgs_result)

        def _fake_resolve(config, capability, selection=None):
            del config
            if capability == "search" and selection is None:
                return [jina_handler]
            if capability == "search" and selection == "ddgs":
                return [ddgs_handler]
            return []

        with (
            mock.patch.object(web_search, "resolve_tool_handlers", side_effect=_fake_resolve),
            mock.patch.object(web_search, "_resolve_search_handler_timeout_s", return_value=0.01),
        ):
            suite = web_search.WebToolSuite(config={"tools": {"use": {"search": "jina_ai"}}})
            rows, meta = await suite._search_text("葱喵Bot", None, 5)

        self.assertEqual(meta, {})
        self.assertEqual(rows[0]["provider"], "ddgs")
        self.assertEqual(rows[0]["url"], "https://example.com/fallback-timeout")


if __name__ == "__main__":
    unittest.main()
