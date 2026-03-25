from __future__ import annotations

import importlib
import unittest

config = importlib.import_module("core.config")


class ModelConfigServiceTierTests(unittest.TestCase):
    def test_build_model_config_keeps_service_tier_from_active_profile(self) -> None:
        cfg = config.build_model_config(
            {
                "models": [
                    {
                        "name": "codex-fast",
                        "model": "gpt-5.4",
                        "model_provider": "mirror",
                        "service_tier": "fast",
                    },
                    {
                        "name": "fallback",
                        "model": "openrouter/openai/gpt-5.4-nano",
                    },
                ]
            }
        )

        self.assertEqual(cfg.get("model"), "gpt-5.4")
        self.assertEqual(cfg.get("service_tier"), "fast")


if __name__ == "__main__":
    unittest.main()
