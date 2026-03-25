from __future__ import annotations

import importlib
import unittest

tool_view = importlib.import_module("core.tool_view")


class ToolViewPageExtractTests(unittest.TestCase):
    def test_page_extract_sample_text_shows_mode_title_target_and_stats(self) -> None:
        text = tool_view.format_tool_view_text(
            "page_extract",
            {
                "url": "https://www.netflix.com/tw/title/81756595",
                "mode": "sample",
                "title": "Watch Cosmic Princess Kaguya! | Netflix Official Site",
                "count": 30,
                "total_lines": 266,
            },
            max_chars=300,
        )

        self.assertEqual(
            text,
            'Read sample 30/266 lines "Watch Cosmic Princess Kaguya! | Netflix Official Site" in "netflix.com/tw/title/81756595"',
        )

    def test_page_extract_range_text_shows_line_window_and_target(self) -> None:
        text = tool_view.format_tool_view_text(
            "page_extract",
            {
                "url": "https://realsound.jp/movie/2026/03/post-1234567.html",
                "mode": "range",
                "start_line": 120,
                "end_line": 180,
            },
            max_chars=300,
        )

        self.assertEqual(
            text,
            'Read L120-L180 in "realsound.jp/movie/2026/03/post-1234567.html"',
        )


if __name__ == "__main__":
    unittest.main()
