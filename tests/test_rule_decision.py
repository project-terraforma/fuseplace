import unittest

from scripts.utils.conflation import decide_rule_based


class RuleDecisionTests(unittest.TestCase):
    def test_prefers_present_value(self) -> None:
        decision = decide_rule_based(
            attr="phones",
            current_value=None,
            base_value='{"primary": "+1 408 555 0100"}',
            confidence=0.9,
            base_confidence=0.7,
            current_sources='["srcA"]',
            base_sources='["srcB"]',
        )
        self.assertEqual(decision.winner, "base")

    def test_prefers_higher_confidence_when_quality_similar(self) -> None:
        decision = decide_rule_based(
            attr="websites",
            current_value='{"primary": "https://example.com"}',
            base_value='{"primary": "https://example.com"}',
            confidence=1.0,
            base_confidence=0.6,
            current_sources='["srcA", "srcB"]',
            base_sources='["srcB"]',
        )
        self.assertEqual(decision.winner, "current")


if __name__ == "__main__":
    unittest.main()
