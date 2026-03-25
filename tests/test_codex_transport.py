from __future__ import annotations

import importlib
import unittest

codex_transport = importlib.import_module("core.codex_transport")


class CodexTransportRequestTests(unittest.TestCase):
    def test_build_request_body_maps_fast_service_tier_to_priority(self) -> None:
        body = codex_transport.build_request_body(
            cfg={
                "model": "gpt-5.4",
                "reasoning_effort": "xhigh",
                "service_tier": "fast",
            },
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            session_id="test-session",
            stream=True,
        )

        self.assertEqual(body.get("service_tier"), "priority")
        self.assertEqual(body.get("reasoning"), {"effort": "xhigh"})

    def test_build_request_body_uses_plain_messages_for_text_only_turns(self) -> None:
        body = codex_transport.build_request_body(
            cfg={"model": "gpt-5.4"},
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ],
            tools=[],
            session_id="test-session",
            stream=False,
        )

        self.assertEqual(body.get("instructions"), "Be brief.")
        self.assertEqual(body.get("input"), [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
