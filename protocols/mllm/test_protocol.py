#!/usr/bin/env python3

from __future__ import annotations

import unittest

from mllm_protocol import parse_response, task_prompt


class StrictResponseTest(unittest.TestCase):
    material = ["cotton", "cottonpolyamide", "polyester"]

    def test_exact_json_label_is_accepted(self) -> None:
        self.assertEqual(
            parse_response('{"label":"cottonpolyamide"}', "material", self.material),
            ("cottonpolyamide", True),
        )

    def test_substring_is_rejected_but_embedded_json_is_accepted(self) -> None:
        self.assertEqual(
            parse_response("cottonpolyamide", "material", self.material),
            (None, False),
        )
        self.assertEqual(
            parse_response(
                '{"label":"cotton"} because it looks synthetic',
                "material",
                self.material,
            ),
            ("cotton", True),
        )

    def test_feature_negation_text_is_rejected(self) -> None:
        candidates = ["elastic", "soft", "rough"]
        self.assertEqual(
            parse_response(
                "The fabric is soft but not elastic.",
                "features",
                candidates,
            ),
            (None, False),
        )
        self.assertEqual(
            parse_response('{"labels":["soft"]}', "features", candidates),
            (["soft"], True),
        )

    def test_prompt_requires_closed_json(self) -> None:
        prompt = task_prompt("weave", ["plain", "plain-knit"])
        self.assertIn('{"label":"one exact candidate"}', prompt)
        self.assertIn("plain, plain-knit", prompt)


if __name__ == "__main__":
    unittest.main()
