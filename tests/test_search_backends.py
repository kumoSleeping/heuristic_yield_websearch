from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest
from unittest import mock

search_ddgs = importlib.import_module("core.search_ddgs")
search_jina_ai = importlib.import_module("core.search_jina_ai")
web_runtime = importlib.import_module("core.web_runtime")


class _FakeResponse:
    def __init__(self, *, payload: dict[str, object] | None = None, text: str = ""):
        self._payload = payload if isinstance(payload, dict) else {}
        self.text = text
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        if self._payload:
            return self._payload
        raise ValueError("no json payload")


class _FakeAsyncClient:
    def __init__(self, recorder: dict[str, object], response: _FakeResponse, *args, **kwargs):
        del args, kwargs
        self._recorder = recorder
        self._response = response

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    async def get(self, url: str, *, params=None, headers=None):
        self._recorder["url"] = url
        self._recorder["params"] = params
        self._recorder["headers"] = dict(headers or {})
        return self._response


class SearchProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_build_duckduckgo_html_url_keeps_region_and_time_filters(self) -> None:
        url = search_ddgs.build_duckduckgo_html_url(
            "claudecode",
            kl="us-en",
            df="2026-03-13..2026-03-26",
            t="h_",
            ia="web",
        )

        self.assertEqual(
            url,
            "https://html.duckduckgo.com/html/?q=claudecode&l=us-en&df=2026-03-13..2026-03-26&t=h_&ia=web",
        )

    async def test_jina_ddgs_search_wraps_duckduckgo_html_url(self) -> None:
        recorder: dict[str, object] = {}
        response = _FakeResponse(
            text=(
                "# claudecode at DuckDuckGo\n\n"
                "## [Claude Code Ultimate Guide - GitHub]"
                "(https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fdemo%2Frepo&rut=test)\n\n"
                "[github.com/demo/repo]"
                "(https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fdemo%2Frepo&rut=test)\n\n"
                "[Guide summary line]"
                "(https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fdemo%2Frepo&rut=test)\n"
            )
        )

        class _FakeHttpxModule:
            def AsyncClient(self, *args, **kwargs):
                return _FakeAsyncClient(recorder, response, *args, **kwargs)

        async def _fake_usage_meta(*args, **kwargs):
            del args, kwargs
            return {}

        with (
            mock.patch.object(search_ddgs, "load_httpx_module", return_value=_FakeHttpxModule()),
            mock.patch.object(search_ddgs, "build_usage_meta", side_effect=_fake_usage_meta),
        ):
            payload = await search_ddgs.jina_ddgs_search(
                "claudecode",
                kl="us-en",
                df="2026-03-13..2026-03-26",
                t="h_",
                ia="web",
            )

        self.assertEqual(
            recorder.get("url"),
            "https://r.jina.ai/https://html.duckduckgo.com/html/?q=claudecode&l=us-en&df=2026-03-13..2026-03-26&t=h_&ia=web",
        )
        self.assertEqual(payload["results"][0]["title"], "Claude Code Ultimate Guide - GitHub")
        self.assertEqual(payload["results"][0]["url"], "https://github.com/demo/repo")

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


class SearchFallbackTests(unittest.IsolatedAsyncioTestCase):
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

        jina_ddgs_handler = SimpleNamespace(provider="jina_ddgs", target="jina_ddgs", callable=_raise_connect_error)
        ddgs_handler = SimpleNamespace(provider="ddgs", target="ddgs", callable=_return_ddgs_result)

        def _fake_resolve(config, capability, selection=None):
            del config
            if capability == "search" and selection is None:
                return [jina_ddgs_handler]
            if capability == "search" and selection == "ddgs":
                return [ddgs_handler]
            return []

        with mock.patch.object(web_runtime, "resolve_tool_handlers", side_effect=_fake_resolve):
            suite = web_runtime.WebToolSuite(config={"tools": {"use": {"search": "jina_ddgs"}}})
            rows, meta = await suite._search_text("葱喵Bot", None, None, None, None, 5)

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

        jina_ddgs_handler = SimpleNamespace(provider="jina_ddgs", target="jina_ddgs", callable=_sleep_forever_enough)
        ddgs_handler = SimpleNamespace(provider="ddgs", target="ddgs", callable=_return_ddgs_result)

        def _fake_resolve(config, capability, selection=None):
            del config
            if capability == "search" and selection is None:
                return [jina_ddgs_handler]
            if capability == "search" and selection == "ddgs":
                return [ddgs_handler]
            return []

        with (
            mock.patch.object(web_runtime, "resolve_tool_handlers", side_effect=_fake_resolve),
            mock.patch.object(web_runtime, "_resolve_search_handler_timeout_s", return_value=0.01),
        ):
            suite = web_runtime.WebToolSuite(config={"tools": {"use": {"search": "jina_ddgs"}}})
            rows, meta = await suite._search_text("葱喵Bot", None, None, None, None, 5)

        self.assertEqual(meta, {})
        self.assertEqual(rows[0]["provider"], "ddgs")
        self.assertEqual(rows[0]["url"], "https://example.com/fallback-timeout")


if __name__ == "__main__":
    unittest.main()
