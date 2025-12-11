# Results data structure
from dataclasses import dataclass
from typing import List


@dataclass
class Statistics:
    """Statistics about processed words."""

    most_frequent_word: str
    most_frequent_count: int
    least_frequent_word: str
    least_frequent_count: int
    total_words: int


@dataclass
class ProcessingResults:
    """Results from text processing algorithm."""

    statistics: Statistics
    processing_time: float  # seconds
    algorithm_name: str
    filtered_words: List[str] = None  # Optional, for debugging
