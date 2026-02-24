import unittest

from scripts.utils.parsing import (
    extract_tokens,
    normalize_address,
    normalize_phone,
    normalize_url,
    parse_maybe_json,
)


class ParsingTests(unittest.TestCase):
    def test_parse_maybe_json_dict(self) -> None:
        value = '{"primary": "Example"}'
        parsed = parse_maybe_json(value)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["primary"], "Example")

    def test_normalize_url(self) -> None:
        self.assertEqual(normalize_url("https://www.Example.com/path/?q=1"), "example.com/path")

    def test_normalize_phone(self) -> None:
        self.assertEqual(normalize_phone("+1 (415) 555-0199"), "+14155550199")

    def test_normalize_address(self) -> None:
        raw = '{"freeform": "123 Main St"}'
        self.assertEqual(normalize_address(raw), "123 main street")

    def test_extract_tokens_websites(self) -> None:
        tokens = extract_tokens("websites", '{"primary":"https://example.com"}')
        self.assertEqual(tokens, {"example.com"})


if __name__ == "__main__":
    unittest.main()
