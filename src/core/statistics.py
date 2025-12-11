# Statistics computation
from collections import Counter
from typing import Dict, List, Tuple

from .results import Statistics


class StatisticsCalculator:
    """Calculator for word frequency statistics."""

    @staticmethod
    def calculate_frequency(words: List[str]) -> Dict[str, int]:
        """
        Calculate word frequency from a list of words.

        Args:
            words: List of words to count

        Returns:
            Dictionary mapping words to their frequency counts
        """
        return dict(Counter(words))

    @staticmethod
    def find_most_frequent(frequency: Dict[str, int]) -> Tuple[str, int]:
        """
        Find the most frequent word and its count.

        Args:
            frequency: Dictionary mapping words to their frequency counts

        Returns:
            Tuple of (word, count) for the most frequent word
        """
        if not frequency:
            return ("", 0)
        most_frequent_word = max(frequency, key=frequency.get)
        return (most_frequent_word, frequency[most_frequent_word])

    @staticmethod
    def find_least_frequent(frequency: Dict[str, int]) -> Tuple[str, int]:
        """
        Find the least frequent word and its count.

        Args:
            frequency: Dictionary mapping words to their frequency counts

        Returns:
            Tuple of (word, count) for the least frequent word
        """
        if not frequency:
            return ("", 0)
        least_frequent_word = min(frequency, key=frequency.get)
        return (least_frequent_word, frequency[least_frequent_word])

    @staticmethod
    def get_total_count(words: List[str]) -> int:
        """
        Get the total count of words.

        Args:
            words: List of words

        Returns:
            Total number of words
        """
        return len(words)

    @staticmethod
    def compute_statistics(words: List[str]) -> Statistics:
        """
        Compute all statistics from a list of words.

        Args:
            words: List of filtered words

        Returns:
            Statistics object containing all computed statistics
        """
        if not words:
            return Statistics(
                most_frequent_word="",
                most_frequent_count=0,
                least_frequent_word="",
                least_frequent_count=0,
                total_words=0,
            )

        frequency = StatisticsCalculator.calculate_frequency(words)
        most_frequent_word, most_frequent_count = (
            StatisticsCalculator.find_most_frequent(frequency)
        )
        least_frequent_word, least_frequent_count = (
            StatisticsCalculator.find_least_frequent(frequency)
        )
        total_words = StatisticsCalculator.get_total_count(words)

        return Statistics(
            most_frequent_word=most_frequent_word,
            most_frequent_count=most_frequent_count,
            least_frequent_word=least_frequent_word,
            least_frequent_count=least_frequent_count,
            total_words=total_words,
        )
