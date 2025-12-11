"""
Data loading utilities for text files and stop words.

This module handles file I/O operations, text tokenization, and word normalization.
All text is normalized to lowercase to ensure consistent matching between source
text and stop words.
"""


class DataLoader:
    """Simple data loader for text files and stop words."""

    @staticmethod
    def load_text_file(filepath: str) -> list[str]:
        """
        Load a text file and return all lines.

        Args:
            filepath: Path to the text file

        Returns:
            List of lines from the file
        """
        with open(filepath, "r", encoding="utf-8-sig") as file:
            return file.read().splitlines()

    @staticmethod
    def load_stop_words(filepath: str) -> set[str]:
        """
        Load stop words from a file (one word per line).

        Args:
            filepath: Path to the stop words file

        Returns:
            Set of stop words (lowercase)
        """
        with open(filepath, "r") as file:
            return set(word.strip().lower() for word in file.readlines())

    @staticmethod
    def tokenize_text(lines: list[str]) -> list[str]:
        """
        Tokenize text lines into words.

        Args:
            lines: List of text lines

        Returns:
            List of normalized words (lowercase, punctuation removed)
        """
        words = []
        for line in lines:
            for word in line.split():
                # Normalize: lowercase and remove punctuation
                word = word.lower().strip('.,!?;:"()[]{}')
                if word:
                    words.append(word)
        return words
