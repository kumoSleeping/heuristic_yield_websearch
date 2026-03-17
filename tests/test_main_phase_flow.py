import unittest

from core.main import (
    _PHASE_STAGE1,
    _PHASE_STAGE2,
    _PhaseRuntimeState,
    _SessionRuntimeState,
    _build_phase_prompt,
    _build_stage2_phase_messages,
    _prepare_user_input_content,
    build_stage_model_config,
    _decide_turn,
    _extract_page_line_matches,
    _parse_article_skeleton,
    _parse_page_probe,
    _parse_search_rewrite_terms,
    _parse_user_need_items,
    _reset_runtime_state_for_execute,
    _select_main_loop_calls,
)


class MainPhaseFlowTests(unittest.TestCase):
    def test_parse_article_skeleton_extracts_claims_from_minimal_markdown(self):
        block = """
## Verification Outline
[1] 第一条待核验句子
[2] 第二条待核验句子
""".strip()
        title, claims = _parse_article_skeleton(block)
        self.assertEqual(title, "")
        self.assertEqual([claim.claim_id for claim in claims], ["1", "2"])
        self.assertEqual([claim.section for claim in claims], ["", ""])

    def test_select_main_loop_calls_search_phase_only_keeps_search(self):
        calls = [
            {"name": "web_search", "args": {"query": "a"}},
            {"name": "page_extract", "args": {"url": "https://example.com", "query": "x", "lines": "20"}},
            {"name": "web_search", "args": {"query": "b"}},
        ]
        selected = _select_main_loop_calls(
            calls,
            round_i=0,
            max_rounds=8,
            phase=_PHASE_STAGE1,
        )
        self.assertEqual([call["name"] for call in selected], ["web_search", "page_extract", "web_search"])
        self.assertEqual(selected[0]["args"]["query"], "a")
        self.assertEqual(selected[1]["args"]["url"], "https://example.com/")
        self.assertEqual(selected[2]["args"]["query"], "b")

    def test_decide_turn_search_phase_accepts_direct_page_fetch(self):
        phase_state = _PhaseRuntimeState()
        decision = _decide_turn(
            text="""
收到，我将查看这个链接。

<page url="https://github.com/lioensky/VCPToolBox" lines="all">VCPToolBox</page>
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=0,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.kind, "tools")
        self.assertEqual(decision.next_phase, _PHASE_STAGE1)
        self.assertEqual(len(decision.calls), 1)
        self.assertEqual(decision.calls[0]["name"], "page_extract")
        self.assertEqual(decision.calls[0]["args"]["lines"], "all")
        self.assertEqual(phase_state.first_collection_mode, "page")

    def test_decide_turn_search_calls_advance_to_skeleton(self):
        phase_state = _PhaseRuntimeState()
        decision = _decide_turn(
            text="""
先检索基础资料。
<task_list><task><status>in_progress</status><what_is>搜索基础背景</what_is></task></task_list>
<search>超时空辉夜姬</search>
<search>超时空辉夜姬 スマコン</search>
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=0,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.kind, "tools")
        self.assertEqual(decision.next_phase, _PHASE_STAGE1)
        self.assertEqual(len(decision.calls), 2)
        self.assertEqual(phase_state.first_collection_mode, "websearch")

    def test_decide_turn_search_phase_can_finish_without_tools(self):
        decision = _decide_turn(
            text="你好，今天天气不错。",
            phase=_PHASE_STAGE1,
            round_i=0,
            max_rounds=8,
            phase_state=_PhaseRuntimeState(),
            cfg={},
        )
        self.assertEqual(decision.kind, "final")
        self.assertEqual(decision.final_text, "你好，今天天气不错。")

    def test_skeleton_prompt_changes_with_first_collection_mode(self):
        page_prompt = _build_phase_prompt(
            _PHASE_STAGE1,
            phase_state=_PhaseRuntimeState(phase=_PHASE_STAGE1, first_collection_mode="page"),
        )
        mix_prompt = _build_phase_prompt(
            _PHASE_STAGE1,
            phase_state=_PhaseRuntimeState(phase=_PHASE_STAGE1, first_collection_mode="mix"),
        )
        self.assertIn("## After Page", page_prompt)
        self.assertIn("立即直接进入 skeleton", page_prompt)
        self.assertIn("## After Search", mix_prompt)

    def test_decide_turn_stage1_block_advance_to_stage2(self):
        phase_state = _PhaseRuntimeState(phase=_PHASE_STAGE1)
        decision = _decide_turn(
            text="""
整理成骨架。
## Keyword Rewrite
- スマコン
- 超时空辉夜姬

## User Need Reconstruction
- 用户真正想确认作品里该道具的价格与购买细节

## Verification Outline
[1] 第一条待核验句子
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=1,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.kind, "continue")
        self.assertEqual(decision.next_phase, _PHASE_STAGE2)
        self.assertIn("## Verification Outline", phase_state.skeleton_xml)
        self.assertEqual(phase_state.user_need_items, ["用户真正想确认作品里该道具的价格与购买细节"])
        self.assertEqual(phase_state.search_rewrite_terms, ["スマコン", "超时空辉夜姬"])
        self.assertEqual([claim.claim_id for claim in phase_state.claims], ["1"])

    def test_decide_turn_stage1_xml_skeleton_advance_to_stage2(self):
        phase_state = _PhaseRuntimeState(phase=_PHASE_STAGE1)
        decision = _decide_turn(
            text="""
<keyword_rewrite>
  <t>スマコン</t>
  <t>超时空辉夜姬</t>
</keyword_rewrite>

<user_need>
  <u>用户真正想确认作品里该道具的价格与购买细节</u>
</user_need>

<verification_outline>
  <i id="1">第一条待核验句子</i>
</verification_outline>
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=1,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.kind, "continue")
        self.assertEqual(decision.next_phase, _PHASE_STAGE2)
        self.assertEqual(phase_state.user_need_items, ["用户真正想确认作品里该道具的价格与购买细节"])
        self.assertEqual(phase_state.search_rewrite_terms, ["スマコン", "超时空辉夜姬"])
        self.assertEqual([claim.claim_id for claim in phase_state.claims], ["1"])

    def test_parse_search_rewrite_terms_extracts_terms(self):
        block = """
## Keyword Rewrite
- 官方术语A
- 官方术语B
        """.strip()
        self.assertEqual(_parse_search_rewrite_terms(block), ["官方术语A", "官方术语B"])

    def test_parse_xml_skeleton_blocks_extracts_terms_and_need(self):
        search_block = """
<keyword_rewrite>
  <t>官方术语A</t>
  <t>官方术语B</t>
</keyword_rewrite>
""".strip()
        need_block = """
<user_need>
  <u>查找 Claude 讲述渐进式上下文的原文</u>
</user_need>
""".strip()
        outline_block = """
<verification_outline>
  <i id="1">第一条待核验句子</i>
  <i id="2">第二条待核验句子</i>
</verification_outline>
""".strip()
        self.assertEqual(_parse_search_rewrite_terms(search_block), ["官方术语A", "官方术语B"])
        self.assertEqual(_parse_user_need_items(need_block), ["查找 Claude 讲述渐进式上下文的原文"])
        _, claims = _parse_article_skeleton(outline_block)
        self.assertEqual([claim.claim_id for claim in claims], ["1", "2"])

    def test_parse_user_need_items_extracts_need_restore(self):
        block = """
## User Need Reconstruction
- 查找 Claude 讲述渐进式上下文的原文
- 给出与该说法直接相关的原文段落
""".strip()
        self.assertEqual(
            _parse_user_need_items(block),
            ["查找 Claude 讲述渐进式上下文的原文", "给出与该说法直接相关的原文段落"],
        )

    def test_stage1_without_search_rewrite_stays_in_stage1(self):
        phase_state = _PhaseRuntimeState(phase=_PHASE_STAGE1)
        decision = _decide_turn(
            text="""
## User Need Reconstruction
- 只给答案

## Verification Outline
[1] 只有骨架没有搜索词重绘
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=1,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.kind, "continue")
        self.assertEqual(decision.next_phase, _PHASE_STAGE1)
        self.assertFalse(decision.store_assistant_turn)

    def test_reset_runtime_state_for_execute_clears_search_history(self):
        runtime_state = _SessionRuntimeState(
            search_history_raw=["a", "b"],
            search_history_normalized=["a", "b"],
            search_results_deduped=[{"url": "https://example.com"}],
        )
        _reset_runtime_state_for_execute(runtime_state)
        self.assertEqual(runtime_state.search_history_raw, [])
        self.assertEqual(runtime_state.search_history_normalized, [])
        self.assertEqual(runtime_state.search_results_deduped, [])

    def test_build_stage2_phase_messages_uses_minimal_context(self):
        phase_state = _PhaseRuntimeState(
            phase=_PHASE_STAGE2,
            skeleton_xml="## Verification Outline\n[1] a",
            user_need_items=["确认项目功能与风险边界"],
            search_rewrite_terms=["官方术语A", "官方术语B"],
        )
        messages = _build_stage2_phase_messages(
            cfg={},
            prompt_text="原始问题",
            phase_state=phase_state,
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("User Need Reconstruction", messages[1]["content"])
        self.assertIn("Keyword Rewrite", messages[1]["content"])

    def test_stage2_final_reply_guidance_starts_on_second_stage2_turn(self):
        first_prompt = _build_phase_prompt(
            _PHASE_STAGE2,
            phase_state=_PhaseRuntimeState(
                phase=_PHASE_STAGE2,
                stage2_turns=1,
                skeleton_xml="## Verification Outline\n[1] a",
            ),
        )
        second_prompt = _build_phase_prompt(
            _PHASE_STAGE2,
            phase_state=_PhaseRuntimeState(
                phase=_PHASE_STAGE2,
                stage2_turns=2,
                skeleton_xml="## Verification Outline\n[1] a",
            ),
        )
        self.assertNotIn("不要在这一阶段提前写完整操作指南", first_prompt)
        self.assertIn("不要在这一阶段提前写完整操作指南", second_prompt)

    def test_extract_markdown_blocks_in_decide_turn(self):
        phase_state = _PhaseRuntimeState(phase=_PHASE_STAGE1)
        decision = _decide_turn(
            text="""
## Keyword Rewrite
- 官方术语A
- 官方术语B

## User Need Reconstruction
- 验证这个说法并给出处

## Verification Outline
[1] 句子
""".strip(),
            phase=_PHASE_STAGE1,
            round_i=1,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.next_phase, _PHASE_STAGE2)
        self.assertEqual(phase_state.search_rewrite_terms, ["官方术语A", "官方术语B"])
        self.assertEqual([claim.claim_id for claim in phase_state.claims], ["1"])

    def test_stage1_plain_summary_is_not_preserved(self):
        phase_state = _PhaseRuntimeState(phase=_PHASE_STAGE1)
        decision = _decide_turn(
            text="这是一个项目总结，但没有合法的 markdown 骨架结构。",
            phase=_PHASE_STAGE1,
            round_i=1,
            max_rounds=8,
            phase_state=phase_state,
            cfg={},
        )
        self.assertEqual(decision.next_phase, _PHASE_STAGE1)
        self.assertFalse(decision.store_assistant_turn)

    def test_decide_turn_final_empty_output_falls_back_without_cfg_error(self):
        decision = _decide_turn(
            text="",
            phase="final",
            round_i=0,
            max_rounds=2,
            phase_state=_PhaseRuntimeState(phase="final"),
            cfg={},
        )
        self.assertEqual(decision.kind, "final")
        self.assertTrue(bool(decision.final_text))

    def test_build_stage_model_config_defaults_to_first_two_models(self):
        cfg = {
            "models": [
                {"model": "openrouter/google/gemini-3.1-flash-lite-preview:nitro"},
                {"model": "stepfun/step-3.5-flash:free"},
            ],
        }
        stage1_cfg = build_stage_model_config(cfg, "stage1")
        stage2_cfg = build_stage_model_config(cfg, "stage2")
        self.assertEqual(stage1_cfg["model"], "openrouter/google/gemini-3.1-flash-lite-preview:nitro")
        self.assertEqual(stage2_cfg["model"], "stepfun/step-3.5-flash:free")

    def test_build_stage_model_config_can_pick_independent_stage_indices(self):
        cfg = {
            "models": [
                {"model": "model-a"},
                {"model": "model-b"},
                {"model": "model-c"},
            ],
        }
        stage1_cfg = build_stage_model_config(cfg, "stage1", stage1_model_index=2, stage2_model_index=1)
        stage2_cfg = build_stage_model_config(cfg, "stage2", stage1_model_index=2, stage2_model_index=1)
        self.assertEqual(stage1_cfg["model"], "model-c")
        self.assertEqual(stage2_cfg["model"], "model-b")

    def test_prepare_user_input_content_with_images_returns_multimodal_content(self):
        content, error = _prepare_user_input_content(
            "请描述这张图",
            ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9l9K8AAAAASUVORK5CYII="],
            config={},
        )
        self.assertEqual(error, "")
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")

    def test_parse_page_probe_all_ignores_keywords(self):
        keywords, window = _parse_page_probe("关键词1|关键词2", lines="all")
        self.assertEqual(window, "all")
        self.assertEqual(keywords, [])

    def test_extract_page_line_matches_all_returns_full_page(self):
        matches = _extract_page_line_matches(
            "第1行\n\n第3行\n第4行",
            keywords=["不会被使用"],
            window="all",
        )
        self.assertEqual(
            matches,
            [
                {"line": 1, "text": "第1行"},
                {"line": 3, "text": "第3行"},
                {"line": 4, "text": "第4行"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
