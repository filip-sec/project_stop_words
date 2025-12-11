# Test visualization of the filtering process
from src.core.data_loader import DataLoader
from src.core.word_filter import WordFilter
from collections import Counter
from src.visualization.chart_generator import ChartGenerator

# Load and process data
print("Loading and processing data...")
data_lines = DataLoader.load_text_file("data.txt")
stop_words = DataLoader.load_stop_words("stop_words.txt")
all_words = DataLoader.tokenize_text(data_lines)

# Apply filters
word_filter = WordFilter(min_length=4, max_length=8, stop_words=stop_words)
filtered_by_length = word_filter.filter_by_length(all_words)
final_words = word_filter.apply_all_filters(all_words)

# Compute statistics
word_counts = Counter(final_words)
most_frequent_word, most_frequent_count = word_counts.most_common(1)[0]
least_frequent_word, least_frequent_count = word_counts.most_common()[-1]

print(f"\nProcessing Summary:")
print(f"  Total words before filter: {len(all_words):,}")
print(f"  After length filter: {len(filtered_by_length):,}")
print(f"  After stop words filter: {len(final_words):,}")
print(f"\nStatistics:")
print(f"  Most frequent: '{most_frequent_word}' ({most_frequent_count:,} times)")
print(f"  Least frequent: '{least_frequent_word}' ({least_frequent_count:,} time)")

# Create visualizations
print("\nGenerating visualizations...")

# 1. Filtering process visualization
ChartGenerator.visualize_filtering_process(
    before_filter=len(all_words),
    after_length_filter=len(filtered_by_length),
    after_stop_words_filter=len(final_words),
    output_file="filtering_process.png"
)

# 2. Top word frequencies
ChartGenerator.visualize_word_frequencies(
    word_counts=dict(word_counts),
    top_n=20,
    output_file="word_frequencies.png"
)

print("\nVisualization complete!")

