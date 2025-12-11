"""Tests for processor implementations."""

import unittest
from src.core.data_loader import DataLoader
from src.core.word_filter import WordFilter
from src.core.statistics import StatisticsCalculator


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def test_load_stop_words(self):
        """Test loading stop words from file."""
        # Create temporary stop words file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('word1\nword2\nWORD3\n')
            temp_file = f.name

        try:
            stop_words = DataLoader.load_stop_words(temp_file)
            self.assertIn('word1', stop_words)
            self.assertIn('word2', stop_words)
            self.assertIn('word3', stop_words)  # Should be lowercase
            self.assertEqual(len(stop_words), 3)
        finally:
            import os
            os.unlink(temp_file)

    def test_tokenize_text(self):
        """Test text tokenization."""
        lines = ['Hello, world!', 'This is a test.']
        words = DataLoader.tokenize_text(lines)
        self.assertIn('hello', words)
        self.assertIn('world', words)
        self.assertIn('test', words)
        # Punctuation should be removed
        self.assertNotIn('hello,', words)


class TestStatisticsCalculator(unittest.TestCase):
    """Test cases for StatisticsCalculator class."""

    def test_calculate_frequency(self):
        """Test frequency calculation."""
        words = ['a', 'b', 'a', 'c', 'b', 'a']
        calc = StatisticsCalculator()
        freq = calc.calculate_frequency(words)
        self.assertEqual(freq['a'], 3)
        self.assertEqual(freq['b'], 2)
        self.assertEqual(freq['c'], 1)

    def test_find_most_frequent(self):
        """Test finding most frequent word."""
        freq = {'a': 3, 'b': 2, 'c': 1}
        calc = StatisticsCalculator()
        word, count = calc.find_most_frequent(freq)
        self.assertEqual(word, 'a')
        self.assertEqual(count, 3)

    def test_find_least_frequent(self):
        """Test finding least frequent word."""
        freq = {'a': 3, 'b': 2, 'c': 1}
        calc = StatisticsCalculator()
        word, count = calc.find_least_frequent(freq)
        self.assertEqual(word, 'c')
        self.assertEqual(count, 1)

    def test_compute_statistics(self):
        """Test complete statistics computation."""
        words = ['a', 'b', 'a', 'c', 'b', 'a']
        calc = StatisticsCalculator()
        stats = calc.compute_statistics(words)
        self.assertEqual(stats.most_frequent_word, 'a')
        self.assertEqual(stats.most_frequent_count, 3)
        self.assertEqual(stats.least_frequent_word, 'c')
        self.assertEqual(stats.least_frequent_count, 1)
        self.assertEqual(stats.total_words, 6)


if __name__ == '__main__':
    unittest.main()
