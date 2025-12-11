# Demonstration: How words from data are matched with stop words

from src.core.data_loader import DataLoader

print("=" * 70)
print("HOW WORDS FROM DATA ARE MATCHED WITH STOP WORDS")
print("=" * 70)

# Step 1: Load stop words
print("\n1. LOADING STOP WORDS:")
print("-" * 70)
stop_words = DataLoader.load_stop_words("stop_words.txt")
print(f"Stop words loaded: {stop_words}")
print(f"Type: {type(stop_words)} (set for fast lookup)")
print(f"All converted to lowercase: {[w for w in stop_words]}")

# Step 2: Load and tokenize data
print("\n2. LOADING AND TOKENIZING DATA:")
print("-" * 70)
data_lines = DataLoader.load_text_file("data.txt")
print(f"Loaded {len(data_lines)} lines from data.txt")
print(f"First line: {data_lines[0][:60]}...")

# Step 3: Show normalization
print("\n3. NORMALIZATION (Both data words AND stop words):")
print("-" * 70)
print("Stop words file contains: 'VERSION', 'Gutenberg', 'WARRANTY'")
print("After normalization: 'version', 'gutenberg', 'warranty'")
print("\nData words are also normalized:")
sample_line = data_lines[0]
sample_words = sample_line.split()[:5]
print(f"Original: {sample_words}")
normalized = [w.lower().strip('.,!?;:"()[]{}') for w in sample_words]
print(f"Normalized: {normalized}")

# Step 4: Show the matching process
print("\n4. MATCHING PROCESS:")
print("-" * 70)
print("For each word from data:")
print("  1. Normalize: word.lower().strip(punctuation)")
print("  2. Check: word not in stop_words_set")
print("  3. Keep word if NOT in stop words")

# Step 5: Demonstrate with actual examples
print("\n5. ACTUAL EXAMPLES:")
print("-" * 70)

# Get some words from data
all_words = DataLoader.tokenize_text(data_lines[:100])  # First 100 lines
sample_words_to_check = ['the', 'project', 'gutenberg', 'ebook', 'version', 'electronic']

print("\nChecking sample words:")
for word in sample_words_to_check:
    normalized_word = word.lower()
    is_stop_word = normalized_word in stop_words
    status = "❌ REMOVED (stop word)" if is_stop_word else "✓ KEPT"
    print(f"  '{word}' -> '{normalized_word}' -> {status}")

# Step 6: Show the filtering
print("\n6. FILTERING OPERATION:")
print("-" * 70)
print("Code: [word for word in words if word not in stop_words]")
print("\nThis means:")
print("  • Iterate through each word")
print("  • Check if word is NOT in stop_words set")
print("  • Keep word only if it's NOT a stop word")
print("  • Set lookup is O(1) - very fast!")

# Step 7: Show actual filtering
print("\n7. ACTUAL FILTERING RESULT:")
print("-" * 70)
from src.core.word_filter import WordFilter

word_filter = WordFilter(min_length=4, max_length=8, stop_words=stop_words)
filtered = word_filter.filter_stop_words(all_words[:50])

print(f"Before filtering: {len(all_words[:50])} words")
print(f"After stop words filter: {len(filtered)} words")
print(f"Removed: {len(all_words[:50]) - len(filtered)} stop words")
print(f"\nSample filtered words: {filtered[:10]}")

print("\n" + "=" * 70)
print("KEY POINT: Both data words and stop words are normalized")
print("to lowercase, so matching works correctly!")
print("=" * 70)

