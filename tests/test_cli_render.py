from __future__ import annotations

import importlib
import unittest

from rich.markdown import Markdown

cli = importlib.import_module("core.cli")


class CliRenderTests(unittest.TestCase):
    def test_stream_preview_renders_markdown_immediately(self) -> None:
        rendered = cli._render_stream_preview("# 标题\n\n- 项目")

        self.assertIsInstance(rendered, Markdown)

    def test_stabilize_stream_markdown_closes_unfinished_code_fence(self) -> None:
        stabilized = cli._stabilize_stream_markdown("```python\nprint('x')")

        self.assertTrue(stabilized.rstrip().endswith("```"))


if __name__ == "__main__":
    unittest.main()
