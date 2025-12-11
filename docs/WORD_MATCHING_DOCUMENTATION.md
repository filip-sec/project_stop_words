# Word Matching Documentation

## Normalization

Both source words and stop words are normalized to lowercase for matching.

**Stop words**: `word.strip().lower()` → stored in set  
**Source words**: `word.lower().strip('.,!?;:"()[]{}')` → stored in list

**Implementation**: `src/core/data_loader.py`

## Set-Based Matching

Stop words stored in Python `set` for O(1) membership testing.

**Time Complexity**:
- Set: O(1) per lookup
- List: O(m) per lookup where m = stop words count

**For n words and m stop words**:
- Set: O(n) total
- List: O(n×m) total

## Filtering Algorithm

```python
def filter_stop_words(self, words: list[str]) -> list[str]:
    return [word for word in words if word not in self.stop_words]
```

**Implementation**: `src/core/word_filter.py`

**Complexity**: O(n) time, O(n) space

## Performance

Set-based approach provides 10× performance improvement for typical use case (m=10, n=200,000).
