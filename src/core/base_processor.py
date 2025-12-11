# Abstract base class for all processors
from abc import ABC, abstractmethod
from typing import List

from .data_loader import DataLoader
from .word_filter import WordFilter
from .statistics import StatisticsCalculator
from .results import Statistics, ProcessingResults


class BaseProcessor(ABC):
    """Abstract base class for all text processing algorithms."""

    def __init__(
        self,
        data_file: str,
        stop_words_file: str,
        min_length: int = 4,
        max_length: int = 8,
    ):
        """
        Initialize the processor.

        Args:
            data_file: Path to the data text file
            stop_words_file: Path to the stop words file
            min_length: Minimum word length to keep
            max_length: Maximum word length to keep
        """
        self.data_file = data_file
        self.stop_words_file = stop_words_file
        self.min_length = min_length
        self.max_length = max_length

        # Initialize components
        self.data_loader = DataLoader()
        self.word_filter = None  # Will be initialized in process()
        self.statistics_calculator = StatisticsCalculator()

    def load_data(self) -> List[str]:
        """
        Load text data from file.

        Returns:
            List of text lines
        """
        return self.data_loader.load_text_file(self.data_file)

    def load_stop_words(self) -> set[str]:
        """
        Load stop words from file.

        Returns:
            Set of stop words (lowercase)
        """
        return self.data_loader.load_stop_words(self.stop_words_file)

    def tokenize_text(self, lines: List[str]) -> List[str]:
        """
        Tokenize text lines into words.

        Args:
            lines: List of text lines

        Returns:
            List of normalized words
        """
        return self.data_loader.tokenize_text(lines)

    def filter_by_length(self, words: List[str]) -> List[str]:
        """
        Filter words by length.

        Args:
            words: List of words to filter

        Returns:
            Filtered list of words
        """
        if self.word_filter is None:
            stop_words = self.load_stop_words()
            self.word_filter = WordFilter(
                min_length=self.min_length,
                max_length=self.max_length,
                stop_words=stop_words,
            )
        return self.word_filter.filter_by_length(words)

    def filter_stop_words(self, words: List[str]) -> List[str]:
        """
        Filter out stop words.

        Args:
            words: List of words to filter

        Returns:
            Filtered list of words
        """
        if self.word_filter is None:
            stop_words = self.load_stop_words()
            self.word_filter = WordFilter(
                min_length=self.min_length,
                max_length=self.max_length,
                stop_words=stop_words,
            )
        return self.word_filter.filter_stop_words(words)

    def compute_statistics(self, words: List[str]) -> Statistics:
        """
        Compute statistics from filtered words.

        Args:
            words: List of filtered words

        Returns:
            Statistics object
        """
        return self.statistics_calculator.compute_statistics(words)

    @abstractmethod
    def process(self) -> List[str]:
        """
        Process the data and return filtered words.
        This method must be implemented by subclasses.

        Returns:
            List of filtered words
        """
        pass

    def execute(self) -> ProcessingResults:
        """
        Template method that defines the processing workflow.
        Subclasses can override process() to customize the algorithm.

        Returns:
            ProcessingResults object with statistics and timing
        """
        import time

        start_time = time.time()

        # Process the data (algorithm-specific implementation)
        filtered_words = self.process()

        # Compute statistics (common for all algorithms)
        statistics = self.compute_statistics(filtered_words)

        processing_time = time.time() - start_time

        return ProcessingResults(
            statistics=statistics,
            processing_time=processing_time,
            algorithm_name=self.__class__.__name__,
            filtered_words=filtered_words,
        )
