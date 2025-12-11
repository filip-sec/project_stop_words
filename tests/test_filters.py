"""Tests for word filtering functionality."""

import unittest
from src.core.word_filter import WordFilter


class TestWordFilter(unittest.TestCase):
    """Test cases for WordFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.stop_words = {"the", "a", "an", "is", "are"}
        self.filter = WordFilter(min_length=4, max_length=8, stop_words=self.stop_words)

    def test_filter_by_length(self):
        """Test length-based filtering."""
        words = ["a", "the", "word", "verylongword", "test", "ok"]
        result = self.filter.filter_by_length(words)
        self.assertEqual(result, ["word", "test"])

    def test_filter_stop_words(self):
        """Test stop word filtering."""
        words = ["the", "word", "is", "test", "a"]
        result = self.filter.filter_stop_words(words)
        self.assertEqual(result, ["word", "test"])

    def test_apply_all_filters(self):
        """Test combined filtering."""
        words = ["a", "the", "word", "verylongword", "is", "test"]
        result = self.filter.apply_all_filters(words)
        self.assertEqual(result, ["word", "test"])

    def test_empty_input(self):
        """Test filtering empty input."""
        result = self.filter.apply_all_filters([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
