import unittest
from io import StringIO

from rich.console import Console

from core.cli import (
    _apply_model_state,
    _cycle_model,
    _normalize_markdown_for_cli,
    _render_reply_body,
    _render_turn_transcript,
)


class CliRenderTests(unittest.TestCase):
    def test_normalize_markdown_for_cli_flattens_indented_lists(self):
        text = """
    2. 关于“彩叶笔记本”的核实

    您提到的“彩叶笔记本”很可能是对剧情细节的记忆偏差，实际情况如下：

     • 购买方式：剧情中，辉夜是通过彩叶的**笔记本电脑（ノートPC）**在网上擅自下单购买的。
     • 资金来源：辉夜使用的是彩叶的存款（貯金）。
     • 剧情影响：这一行为直接导致彩叶的存款被清空（或大幅减少），并因此“号泣”（大哭）。
""".strip("\n")
        normalized = _normalize_markdown_for_cli(text)
        self.assertIn("2. 关于“彩叶笔记本”的核实", normalized)
        self.assertIn("- 购买方式：剧情中，辉夜是通过彩叶的**笔记本电脑（ノートPC）**在网上擅自下单购买的。", normalized)
        self.assertIn("- 资金来源：辉夜使用的是彩叶的存款（貯金）。", normalized)
        self.assertIn("- 剧情影响：这一行为直接导致彩叶的存款被清空（或大幅减少），并因此“号泣”（大哭）。", normalized)

    def test_render_reply_body_does_not_ellipsis_indented_bullet_block(self):
        text = """
    2. 关于“彩叶笔记本”的核实

    您提到的“彩叶笔记本”很可能是对剧情细节的记忆偏差，实际情况如下：

     • 购买方式：剧情中，辉夜是通过彩叶的**笔记本电脑（ノートPC）**在网上擅自下单购买的。
     • 资金来源：辉夜使用的是彩叶的存款（貯金）。
     • 剧情影响：这一行为直接导致彩叶的存款被清空（或大幅减少），并因此“号泣”（大哭）。

    总结

    剧情中明确提及了该金额，但该数字属于核心剧情细节，而非公开简介中的常识性信息。在一般讨论中，该事件被定义为“辉夜因自由奔放的性格导致彩叶经济危机”的一环。
""".strip("\n")
        console = Console(width=90, record=True, file=StringIO())
        console.print(_render_reply_body(text))
        rendered = console.export_text()
        self.assertNotIn("…", rendered)
        self.assertIn("购买方式：剧情中，辉夜是通过彩叶的", rendered)
        self.assertIn("剧情中明确提及了该金额", rendered)

    def test_stage2_index_zero_is_preserved_when_stage1_cycles(self):
        state = {
            "models": [
                {"model": "openrouter/google/gemini-3.1-flash-lite-preview:nitro"},
                {"model": "openrouter/stepfun/step-3.5-flash:free"},
                {"model": "openrouter/openai/gpt-oss-120b"},
            ],
            "config": {
                "models": [
                    {"model": "openrouter/google/gemini-3.1-flash-lite-preview:nitro"},
                    {"model": "openrouter/stepfun/step-3.5-flash:free"},
                    {"model": "openrouter/openai/gpt-oss-120b"},
                ]
            },
        }
        _apply_model_state(state, 0, 1)
        _cycle_model(state, "stage2", 1)
        _cycle_model(state, "stage2", 1)
        self.assertEqual(state["stage2_model_index"], 0)
        _cycle_model(state, "stage1", 1)
        self.assertEqual(state["stage1_model_index"], 1)
        self.assertEqual(state["stage2_model_index"], 0)

    def test_render_turn_transcript_preserves_partial_reply_and_note(self):
        console = Console(width=90, record=True, file=StringIO())
        console.print(
            _render_turn_transcript(
                reply_text="这是已生成的一段内容",
                preview=True,
                note_text="已中断",
            )
        )
        rendered = console.export_text()
        self.assertIn("这是已生成的一段内容", rendered)
        self.assertIn("已中断", rendered)


if __name__ == "__main__":
    unittest.main()
