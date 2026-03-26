from __future__ import annotations

import importlib
import unittest

tool_view = importlib.import_module("core.tool_view")


class ToolViewNavigateTests(unittest.TestCase):
    def test_navigate_search_text_shows_query_and_filters(self) -> None:
        text = tool_view.format_tool_view_text(
            "navigate",
            {
                "search": "entari 0.17",
                "df": "2026-03-13..2026-03-26",
                "count": 5,
            },
            max_chars=300,
        )

        self.assertEqual(
            text,
            'Navigate Search "entari 0.17" [df=2026-03-13..2026-03-26] · 5 lines',
        )

    def test_navigate_text_shows_title_target_and_stats(self) -> None:
        text = tool_view.format_tool_view_text(
            "navigate",
            {
                "url": "https://www.netflix.com/tw/title/81756595",
                "title": "Watch Cosmic Princess Kaguya! | Netflix Official Site",
                "count": 30,
                "total_lines": 266,
            },
            max_chars=300,
        )

        self.assertEqual(
            text,
            'Navigate 30 lines "Watch Cosmic Princess Kaguya! | Netflix Official Site"',
        )

    def test_navigate_text_shows_keep_lines(self) -> None:
        text = tool_view.format_tool_view_text(
            "navigate",
            {
                "ref": "44:2",
                "keep": ["L120-L180", "L220"],
            },
            max_chars=300,
        )

        self.assertEqual(
            text,
            'Navigate in "#44:2" keep L120-L180, L220',
        )


if __name__ == "__main__":
    unittest.main()
