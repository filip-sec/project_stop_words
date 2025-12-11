# Filtering logic


class WordFilter:
    """Simple word filter for length and stop words."""

    def __init__(
        self, min_length: int = 4, max_length: int = 8, stop_words: set[str] = None
    ):
        """
        Initialize the word filter.

        Args:
            min_length: Minimum word length to keep (default: 4)
            max_length: Maximum word length to keep (default: 8)
            stop_words: Set of stop words to exclude (default: None)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.stop_words = stop_words or set()

    def filter_by_length(self, words: list[str]) -> list[str]:
        """
        Filter words by length (keep words with length between min_length and max_length).

        Args:
            words: List of words to filter

        Returns:
            Filtered list of words
        """
        return [
            word for word in words if self.min_length <= len(word) <= self.max_length
        ]

    def filter_stop_words(self, words: list[str]) -> list[str]:
        """
        Filter out stop words.

        Args:
            words: List of words to filter

        Returns:
            Filtered list of words (stop words removed)
        """
        return [word for word in words if word not in self.stop_words]

    def apply_all_filters(self, words: list[str]) -> list[str]:
        """
        Apply all filters: length filter first, then stop words filter.

        Args:
            words: List of words to filter

        Returns:
            Filtered list of words
        """
        filtered = self.filter_by_length(words)
        filtered = self.filter_stop_words(filtered)
        return filtered
