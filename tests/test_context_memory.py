from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

main = importlib.import_module("core.main")
prompt = importlib.import_module("core.prompt")


class CompactContextTests(unittest.TestCase):
    def test_build_compact_context_preserves_last_menu_for_choice_reply(self) -> None:
        menu = (
            "1. 苹果\n"
            "2. 香蕉\n"
            "3. 梨\n"
            + ("补充说明 " * 40)
            + "TAIL_TOKEN"
        )
        memory = [
            {"role": "user", "content": "选一个"},
            {"role": "assistant", "content": menu},
        ]

        context = main.build_compact_context(memory, current_user_text="2")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("Assistant (full because current user reply looks like an option selection):", context)
        self.assertIn("TAIL_TOKEN", context)

    def test_build_compact_context_trims_last_menu_for_normal_followup(self) -> None:
        menu = (
            "1. 苹果\n"
            "2. 香蕉\n"
            "3. 梨\n"
            + ("补充说明 " * 40)
            + "TAIL_TOKEN"
        )
        memory = [
            {"role": "user", "content": "选一个"},
            {"role": "assistant", "content": menu},
        ]

        context = main.build_compact_context(memory, current_user_text="详细说说第二个")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertNotIn("Assistant (full because current user reply looks like an option selection):", context)
        self.assertNotIn("TAIL_TOKEN", context)
        self.assertIn("Assistant:", context)

    def test_search_payload_markdown_uses_line_based_search_document(self) -> None:
        rendered = main._build_search_payload_markdown(
            query="entari 0.17",
            search_filters={"kl": "us-en", "df": "2026-03-13..2026-03-26", "ia": "news"},
            public_results=[
                {
                    "title": "Release Entari 0.17.4",
                    "domain": "github.com",
                    "snippet": "Fix directory-style plugin config lookup.",
                    "matched_queries": ["entari 0.17"],
                }
            ],
            skipped_duplicate=False,
        )

        self.assertIn("Filters: kl=us-en, df=2026-03-13..2026-03-26", rendered)
        self.assertNotIn("ia=", rendered)
        self.assertIn("Result lines:", rendered)
        self.assertIn(
            "L01 | [1] Release Entari 0.17.4 | github.com | matched=entari 0.17 | Fix directory-style plugin config lookup.",
            rendered,
        )

    def test_navigate_search_builds_openable_result_lines_and_ref_resolution(self) -> None:
        runtime_state = main._SessionRuntimeState()

        original_search_backend = main._search_backend

        def _fake_search_backend(query, **kwargs):
            del kwargs
            self.assertEqual(query, "entari 0.17")
            return {
                "ok": True,
                "provider": "ddgs",
                "results": [
                    {
                        "title": "Release Entari 0.17.4",
                        "url": "https://github.com/ArcletProject/Entari/releases/tag/v0.17.4",
                        "snippet": "Fix directory-style plugin config lookup.",
                        "domain": "github.com",
                    }
                ],
            }

        main._search_backend = _fake_search_backend
        try:
            payload = main._run_navigate(
                url="",
                search="entari 0.17",
                search_filters={"df": "2026-03-13..2026-03-26"},
                cfg={},
                runtime_state=runtime_state,
            )
        finally:
            main._search_backend = original_search_backend

        self.assertTrue(payload["ok"])
        self.assertIn(
            "[Release Entari 0.17.4](https://github.com/ArcletProject/Entari/releases/tag/v0.17.4)",
            payload["_raw_lines"][0],
        )

        phase_state = main._PhaseRuntimeState()
        page_id = main._set_active_page_item(phase_state, payload, round_no=1, call_id="r1c1")
        calls = [{"name": "navigate", "args": {"ref": f"{page_id}:1"}}]
        main._resolve_page_call_refs(calls, phase_state)

        self.assertEqual(
            calls[0]["args"].get("url"),
            "https://github.com/ArcletProject/Entari/releases/tag/v0.17.4",
        )

    def test_resolve_page_call_refs_supports_memory_page_markdown_links(self) -> None:
        phase_state = main._PhaseRuntimeState()
        item = main._add_context_item(
            phase_state,
            "memory.page",
            {
                "text": (
                    "L02 | [2] [记忆中的彩叶](https://tieba.baidu.com/p/10574520448) | tieba.baidu.com\n"
                    "L04 | [4] [御宅族彩叶](https://tieba.baidu.com/p/10576080529) | tieba.baidu.com"
                )
            },
        )

        calls = [
            {"name": "navigate", "args": {"ref": f"{item.item_id}:2"}},
            {"name": "navigate", "args": {"ref": f"{item.item_id}:4"}},
        ]
        main._resolve_page_call_refs(calls, phase_state)

        self.assertEqual(calls[0]["args"].get("url"), "https://tieba.baidu.com/p/10574520448")
        self.assertEqual(calls[1]["args"].get("url"), "https://tieba.baidu.com/p/10576080529")

    def test_execute_tool_payload_navigate_supports_runtime_ref_targets(self) -> None:
        runtime_state = main._SessionRuntimeState()
        runtime_state.ref_targets["1:2"] = "https://example.com/resolved"

        original_load_page_payload = main._load_page_payload

        def _fake_load_page_payload(*, url, cfg, runtime_state, progress_callback):
            del cfg, runtime_state, progress_callback
            self.assertEqual(url, "https://example.com/resolved")
            return (
                {
                    "ok": True,
                    "provider": "jina_ai",
                    "url": url,
                    "title": "Resolved",
                    "content": "alpha\nbeta",
                },
                False,
            )

        main._load_page_payload = _fake_load_page_payload
        try:
            payload = main.execute_tool_payload(
                "navigate",
                {"ref": "1:2"},
                config={},
                runtime_state=runtime_state,
            )
        finally:
            main._load_page_payload = original_load_page_payload

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["url"], "https://example.com/resolved")

    def test_navigate_search_reuses_runtime_cached_results_without_refetch(self) -> None:
        runtime_state = main._SessionRuntimeState()
        runtime_state.search_results_deduped = [
            {
                "title": "Release Entari 0.17.4",
                "url": "https://github.com/ArcletProject/Entari/releases/tag/v0.17.4",
                "snippet": "Fix directory-style plugin config lookup.",
                "provider": "ddgs",
                "domain": "github.com",
                "matched_queries": ["entari 0.17 [df=2026-03-13..2026-03-26]"],
                "_order": 0,
            }
        ]

        original_search_backend = main._search_backend

        def _unexpected_search_backend(*args, **kwargs):
            raise AssertionError("cached navigate(search=...) should not refetch")

        main._search_backend = _unexpected_search_backend
        try:
            payload = main._run_navigate(
                url="",
                search="entari 0.17",
                search_filters={"df": "2026-03-13..2026-03-26"},
                cfg={},
                runtime_state=runtime_state,
            )
        finally:
            main._search_backend = original_search_backend

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["from_cache"])

    def test_context_ref_supports_page_local_link_refs(self) -> None:
        self.assertEqual(main._normalize_context_ref("44:2"), "44:2")
        self.assertEqual(main._parse_context_ref("44:2"), (44, 2))
        self.assertEqual(main._parse_context_ref("0:2"), (0, 2))

    def test_default_max_rounds_is_20(self) -> None:
        self.assertEqual(main._resolve_max_rounds({}), 20)

    def test_late_round_warning_is_injected_into_system_and_user_messages(self) -> None:
        state = main._PhaseRuntimeState()
        history = [{"role": "user", "content": "查一下 OpenAI 最新模型"}]

        early_msgs = main._build_loop_messages(
            cfg={},
            prompt_text="查一下 OpenAI 最新模型",
            history=history,
            state=state,
            round_i=14,
            context=None,
        )
        late_msgs = main._build_loop_messages(
            cfg={},
            prompt_text="查一下 OpenAI 最新模型",
            history=history,
            state=state,
            round_i=15,
            context=None,
        )

        early_text = "\n".join(str(item.get("content") or "") for item in early_msgs if isinstance(item, dict))
        late_system = next(str(item.get("content") or "") for item in late_msgs if item.get("role") == "system")
        late_user_text = "\n".join(
            str(item.get("content") or "")
            for item in late_msgs
            if isinstance(item, dict) and item.get("role") == "user"
        )

        self.assertNotIn(main.LATE_ROUND_FINAL_REPLY_PROMPT, early_text)
        self.assertIn(main.LATE_ROUND_FINAL_REPLY_PROMPT, late_system)
        self.assertIn(main.LATE_ROUND_FINAL_REPLY_PROMPT, late_user_text)

    def test_first_search_prompt_includes_formatted_user_message(self) -> None:
        state = main._PhaseRuntimeState()
        history = [{"role": "user", "content": "告诉我最近发布的 OpenAI 模型有哪些"}]

        msgs = main._build_loop_messages(
            cfg={},
            prompt_text="告诉我最近发布的 OpenAI 模型有哪些",
            history=history,
            state=state,
            round_i=0,
            context=None,
        )

        user_text = "\n".join(
            str(item.get("content") or "")
            for item in msgs
            if isinstance(item, dict) and item.get("role") == "user"
        )

        self.assertIn("告诉我最近发布的 OpenAI 模型有哪些", user_text)
        self.assertNotIn("{user_message}", user_text)

    def test_first_search_prompt_is_skipped_when_prior_turn_summary_exists(self) -> None:
        state = main._PhaseRuntimeState()
        memory = [
            {"role": "user", "content": "葱喵Bot"},
            {"role": "assistant", "content": "它是一个待查询对象，请补充你的问题。"},
        ]
        context = main.build_compact_context(memory, current_user_text="name在配置里怎么配置")
        history = [{"role": "user", "content": "name在配置里怎么配置"}]

        msgs = main._build_loop_messages(
            cfg={},
            prompt_text="name在配置里怎么配置",
            history=history,
            state=state,
            round_i=0,
            context=context,
        )

        user_text = "\n".join(
            str(item.get("content") or "")
            for item in msgs
            if isinstance(item, dict) and item.get("role") == "user"
        )

        self.assertNotIn("首轮搜索严格参考", user_text)
        self.assertNotIn("默认认为任何孤立名词", user_text)

    def test_post_search_prompt_no_longer_repeats_latest_query_guardrail(self) -> None:
        state = main._PhaseRuntimeState()
        state.disclosure_step = 1
        state.disclosure_refine_consumed = False
        history = [{"role": "user", "content": "告诉我最近发布的 OpenAI 模型有哪些"}]

        msgs = main._build_loop_messages(
            cfg={},
            prompt_text="告诉我最近发布的 OpenAI 模型有哪些",
            history=history,
            state=state,
            round_i=1,
            context=None,
        )

        user_text = "\n".join(
            str(item.get("content") or "")
            for item in msgs
            if isinstance(item, dict) and item.get("role") == "user"
        )

        self.assertNotIn("先建立官方时间轴，再下结论", user_text)

    def test_tool_validation_currently_relies_on_prompt_control(self) -> None:
        state = main._PhaseRuntimeState()
        payload = {
            "url": "https://example.com/start",
            "title": "Demo",
            "from_cache": False,
            "_raw_lines": ["hello"],
            "count": 1,
        }
        main._set_active_page_item(state, payload, round_no=1, call_id="r1c1")

        result = main._validate_progressive_tool_requirements(
            state,
            [{"name": "navigate", "args": {"url": "https://example.com/next"}}],
        )
        self.assertEqual(result, "")

    def test_multiple_open_pages_remain_outstanding_and_visible(self) -> None:
        state = main._PhaseRuntimeState()
        payload_a = {
            "url": "https://example.com/a",
            "title": "Page A",
            "from_cache": False,
            "_raw_lines": ["alpha"],
            "count": 1,
        }
        payload_b = {
            "url": "https://example.com/b",
            "title": "Page B",
            "from_cache": False,
            "_raw_lines": ["beta"],
            "count": 1,
        }

        first_id = main._set_active_page_item(state, payload_a, round_no=1, call_id="r1c1")
        second_id = main._set_active_page_item(state, payload_b, round_no=1, call_id="r1c2")

        self.assertEqual(state.active_page_ids, [first_id, second_id])
        message = main._build_active_page_message(state)
        self.assertIn(f"# Page [{first_id}]", message)
        self.assertIn(f"# Page [{second_id}]", message)

    def test_replace_active_pages_keeps_lines_and_clears_previous_pages(self) -> None:
        state = main._PhaseRuntimeState()
        payload_a = {
            "url": "https://example.com/a",
            "title": "Page A",
            "from_cache": False,
            "_raw_lines": ["alpha"],
            "count": 1,
        }
        payload_b = {
            "url": "https://example.com/b",
            "title": "Page B",
            "from_cache": False,
            "_raw_lines": ["beta"],
            "count": 1,
        }

        first_id = main._set_active_page_item(state, payload_a, round_no=1, call_id="r1c1")
        second_id = main._set_active_page_item(state, payload_b, round_no=1, call_id="r1c2")

        replaced_ids, created_ids = main._replace_active_pages(
            args={"keep": ["L1"]},
            state=state,
            round_no=2,
        )

        self.assertEqual(replaced_ids, [first_id, second_id])
        self.assertEqual(state.active_page_ids, [])
        self.assertTrue(created_ids)
        self.assertEqual(len(created_ids), 2)

    def test_round_brief_explicitly_shows_unclosed_open_count(self) -> None:
        state = main._PhaseRuntimeState()
        payload_a = {
            "url": "https://example.com/a",
            "title": "Page A",
            "from_cache": False,
            "_raw_lines": ["alpha"],
            "count": 1,
        }
        payload_b = {
            "url": "https://example.com/b",
            "title": "Page B",
            "from_cache": False,
            "_raw_lines": ["beta"],
            "count": 1,
        }

        first_id = main._set_active_page_item(state, payload_a, round_no=1, call_id="r1c1")
        second_id = main._set_active_page_item(state, payload_b, round_no=1, call_id="r1c2")
        brief = main._build_round_brief_message(state, round_i=1, user_message="test")

        expected = prompt.ACTIVE_PAGE_STATE_PROMPT.format(count=2, ids=f"{first_id}, {second_id}")
        self.assertIn(expected, brief)

    def test_log_model_call_uses_new_markdown_structure_without_tool_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_id = "session.md"
            config = {"log_dir": tmpdir}
            messages = [
                {"role": "system", "content": "You are hyw."},
                {"role": "user", "content": "最近发布的 openai 模型"},
            ]
            tool_calls = [{"name": "navigate", "args": {"search": "OpenAI latest models"}}]

            main.log_model_call(
                label="round 1",
                model="demo-model",
                messages=messages,
                output="这是整理后的回答。",
                tool_calls=tool_calls,
                turn=1,
                title="最近发布的 openai 模型",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                cost=0.123456,
                duration_ms=789.0,
                config=config,
                log_id=log_id,
            )
            main.log_tool_selection(
                round_i=0,
                calls=tool_calls,
                config=config,
                log_id=log_id,
            )
            main.log_tool_call(
                name="navigate",
                args={"search": "OpenAI latest models"},
                payload={"ok": True, "results": [{"title": "Official"}]},
                duration_ms=456.0,
                turn=1,
                config=config,
                log_id=log_id,
            )

            content = (Path(tmpdir) / log_id).read_text(encoding="utf-8")

        self.assertIn("# 最近发布的 openai 模型", content)
        self.assertIn("## Turn: 1", content)
        self.assertIn("### 模型Input", content)
        self.assertIn("### 模型Output", content)
        self.assertIn("### 模型工具Output", content)
        self.assertIn("### 模型时间、开销", content)
        self.assertIn("### 工具执行摘要", content)
        self.assertIn('"name": "navigate"', content)
        self.assertIn('"tool": "navigate"', content)
        self.assertIn("这是整理后的回答。", content)
        self.assertNotIn("selected tools", content)
        self.assertNotIn("### Args", content)
        self.assertNotIn("### Result", content)
        self.assertNotIn("tool `navigate`", content)

    def test_extract_raw_tool_calls_preserves_provider_id_without_injecting_call_id(self) -> None:
        payload = {
            "tool_calls": [
                {
                    "id": "call_demo",
                    "function": {
                        "name": "navigate",
                        "arguments": "{\"ref\":\"4:2\",\"keep\":[\"L12-L18\"]}",
                    },
                }
            ]
        }

        rows = main._extract_raw_tool_calls(payload)

        self.assertEqual(
            rows,
            [{"id": "call_demo", "name": "navigate", "args": {"ref": "4:2", "keep": ["L12-L18"]}}],
        )

    def test_navigate_requires_valid_keep_and_rejects_old_aliases(self) -> None:
        self.assertIsNone(main._sanitize_native_tool_call("navigate", {"ref": "4:2"}))
        self.assertIsNone(main._sanitize_native_tool_call("navigate", {"keep": ["L1"]}))
        self.assertIsNone(main._sanitize_native_tool_call("navigate", {"ref": "4:2", "keep": ["bad"]}))
        self.assertIsNone(main._sanitize_native_tool_call("open", {"ref": "4:2", "keep": ["L1"]}))
        self.assertIsNone(main._sanitize_native_tool_call("close", {"ref": "4", "keep": ["L1"]}))
        self.assertEqual(
            main._sanitize_native_tool_call("navigate", {"ref": "4:2", "keep": "L0"}),
            {"name": "navigate", "args": {"ref": "4:2", "keep": "L0"}},
        )
        self.assertEqual(
            main._sanitize_native_tool_call("navigate", {"ref": "4:2", "keep": ["L12-L18"]}),
            {"name": "navigate", "args": {"ref": "4:2", "keep": ["L12-L18"]}},
        )


if __name__ == "__main__":
    unittest.main()
