# Text Processing Project - OOP Structure Plan

## 1. Project Overview

This project implements 4 different algorithms for processing text files:
- CPU Single-threaded
- CPU Multi-threaded
- GPU-based (using CUDA/OpenCL)
- Apache Spark distributed processing

All algorithms perform the same operations but with different execution strategies.

## 2. Core Requirements Analysis

### Input Processing
- Load `data.txt` (large text file, ~22K lines)
- Load `stop_words.txt` (10 stop words, one per line)

### Processing Operations
1. Word-by-word processing
2. Filter words: length > 8 OR length < 4 (exclude these)
3. Filter stop words from `stop_words.txt`
4. Compute statistics:
   - Most frequent word and its count
   - Least frequent word and its count
   - Total word count after filtering

### Output Requirements
- Print all results to console
- Measure processing time for each algorithm
- Visualize timing results in pie chart using Matplotlib

## 3. OOP Architecture Design

### 3.1 Design Patterns

**Strategy Pattern**: Different processing algorithms (CPU single, CPU multi, GPU, Spark)
**Factory Pattern**: Create appropriate processor instances
**Template Method Pattern**: Common processing workflow with algorithm-specific steps
**Observer Pattern**: For progress tracking (optional)

### 3.2 Package Structure

```
project_stop_words/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_processor.py          # Abstract base class
│   │   ├── data_loader.py              # File loading utilities
│   │   ├── word_filter.py              # Filtering logic
│   │   ├── statistics.py               # Statistics computation
│   │   └── results.py                  # Results data structure
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── cpu_single_thread.py        # CPU single-threaded
│   │   ├── cpu_multi_thread.py         # CPU multi-threaded
│   │   ├── gpu_processor.py            # GPU implementation
│   │   └── spark_processor.py           # Apache Spark
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── word_mapper.py              # Word-to-ID mapping for GPU
│   │   ├── text_preprocessor.py        # Text cleaning/normalization
│   │   └── performance_timer.py        # Timing utilities
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── chart_generator.py          # Matplotlib pie chart
│   │
│   └── main.py                         # Entry point
│
├── tests/
│   ├── __init__.py
│   ├── test_processors.py
│   └── test_filters.py
│
├── requirements.txt
├── README.md
└── PROJECT_STRUCTURE_PLAN.md
```

## 4. Core Classes Design

### 4.1 Base Processor (Abstract Class)

**File**: `core/base_processor.py`

```python
# Abstract base class for all processors
class BaseProcessor(ABC):
    - data_file: str
    - stop_words_file: str
    - min_length: int = 4
    - max_length: int = 8
    
    + load_data() -> List[str]
    + load_stop_words() -> Set[str]
    + process() -> ProcessingResults
    + filter_by_length(words) -> List[str]
    + filter_stop_words(words) -> List[str]
    + compute_statistics(words) -> Statistics
    + execute() -> ProcessingResults  # Template method
```

**Responsibilities**:
- Define common interface for all processors
- Implement template method pattern for processing workflow
- Provide default implementations for common operations
- Define abstract methods for algorithm-specific implementations

### 4.2 Data Loader

**File**: `core/data_loader.py`

```python
class DataLoader:
    + load_text_file(filepath: str) -> List[str]
    + load_stop_words(filepath: str) -> Set[str]
    + tokenize_text(text: List[str]) -> List[str]
    + normalize_word(word: str) -> str
```

**Responsibilities**:
- File I/O operations
- Text tokenization (split into words)
- Word normalization (lowercase, remove punctuation)

### 4.3 Word Filter

**File**: `core/word_filter.py`

```python
class WordFilter:
    - min_length: int
    - max_length: int
    - stop_words: Set[str]
    
    + filter_by_length(words: List[str]) -> List[str]
    + filter_stop_words(words: List[str]) -> List[str]
    + apply_all_filters(words: List[str]) -> List[str]
```

**Responsibilities**:
- Length-based filtering (4 <= length <= 8)
- Stop word filtering
- Combined filtering logic

### 4.4 Statistics Calculator

**File**: `core/statistics.py`

```python
class StatisticsCalculator:
    + calculate_frequency(words: List[str]) -> Dict[str, int]
    + find_most_frequent(frequency: Dict[str, int]) -> Tuple[str, int]
    + find_least_frequent(frequency: Dict[str, int]) -> Tuple[str, int]
    + get_total_count(words: List[str]) -> int
    + compute_all_statistics(words: List[str]) -> Statistics
```

**Responsibilities**:
- Word frequency counting
- Finding min/max frequency words
- Computing total counts

### 4.5 Results Data Structure

**File**: `core/results.py`

```python
@dataclass
class Statistics:
    most_frequent_word: str
    most_frequent_count: int
    least_frequent_word: str
    least_frequent_count: int
    total_words: int

@dataclass
class ProcessingResults:
    statistics: Statistics
    processing_time: float  # seconds
    algorithm_name: str
    filtered_words: List[str]  # Optional, for debugging
```

## 5. Processor Implementations

### 5.1 CPU Single-Threaded Processor

**File**: `processors/cpu_single_thread.py`

```python
class CPUSingleThreadProcessor(BaseProcessor):
    + process() -> ProcessingResults
    - _process_sequential(words: List[str]) -> List[str]
    - _count_frequencies_sequential(words: List[str]) -> Dict[str, int]
```

**Algorithm Flow**:
1. Load data sequentially
2. Tokenize text sequentially
3. Filter words sequentially (length + stop words)
4. Count frequencies sequentially
5. Compute statistics sequentially

**Optimization Strategies**:
- Use list comprehensions for filtering
- Use Counter from collections for frequency counting
- Minimize memory allocations

### 5.2 CPU Multi-Threaded Processor

**File**: `processors/cpu_multi_thread.py`

```python
class CPUMultiThreadProcessor(BaseProcessor):
    - num_threads: int  # Auto-detect CPU cores
    - chunk_size: int
    
    + process() -> ProcessingResults
    - _split_into_chunks(data: List[str], num_chunks: int) -> List[List[str]]
    - _process_chunk(chunk: List[str]) -> List[str]
    - _merge_results(chunk_results: List[List[str]]) -> List[str]
    - _count_frequencies_parallel(words: List[str]) -> Dict[str, int]
```

**Algorithm Flow**:
1. Load data (sequential - I/O bound)
2. Split text into chunks (one per CPU core)
3. Parallel processing:
   - Each thread: tokenize → filter → count frequencies
4. Merge results from all threads
5. Aggregate frequency counts
6. Compute final statistics

**Parallelization Strategy**:
- Use `concurrent.futures.ThreadPoolExecutor` or `multiprocessing.Pool`
- Split `data.txt` into N chunks (N = CPU cores)
- Each thread processes its chunk independently
- Merge frequency dictionaries at the end

**Thread Safety Considerations**:
- Each thread works on separate data chunk
- Merge phase requires synchronization (use locks or merge after all threads complete)

### 5.3 GPU Processor

**File**: `processors/gpu_processor.py`

**Dependencies**: `cupy` or `numba.cuda` or `pycuda`

```python
class GPUProcessor(BaseProcessor):
    - word_to_id: Dict[str, int]
    - id_to_word: Dict[int, str]
    - stop_word_ids: Set[int]
    
    + process() -> ProcessingResults
    - _create_word_dictionary(words: List[str]) -> Dict[str, int]
    - _convert_to_integer_vector(words: List[str]) -> np.ndarray
    - _filter_by_length_cpu(words: List[str]) -> List[str]  # Pre-GPU step
    - _filter_stop_words_gpu(word_ids: np.ndarray) -> np.ndarray
    - _count_frequencies_gpu(word_ids: np.ndarray) -> Dict[int, int]
    - _convert_ids_back_to_words(word_id_counts: Dict[int, int]) -> Dict[str, int]
```

**Algorithm Flow**:
1. Load data (CPU)
2. Tokenize text (CPU)
3. **Filter by length on CPU** (as per requirements - too complex for GPU)
4. Create word-to-ID dictionary from filtered words
5. Convert words to integer vector (numpy array)
6. Transfer to GPU memory
7. Filter stop words on GPU (using stop word IDs)
8. Count frequencies on GPU (parallel reduction)
9. Transfer results back to CPU
10. Convert IDs back to words
11. Compute statistics (CPU)

**GPU Operations**:
- Use CUDA kernels for:
  - Stop word filtering (parallel element-wise comparison)
  - Frequency counting (parallel reduction/histogram)
- Use GPU-accelerated libraries (CuPy) for array operations

**Memory Management**:
- Batch processing if data doesn't fit in GPU memory
- Efficient GPU-CPU memory transfers

### 5.4 Apache Spark Processor

**File**: `processors/spark_processor.py`

**Dependencies**: `pyspark`

```python
class SparkProcessor(BaseProcessor):
    - spark_session: SparkSession
    - num_partitions: int  # Based on CPU cores
    
    + process() -> ProcessingResults
    - _create_spark_session() -> SparkSession
    - _load_data_rdd(filepath: str) -> RDD[str]
    - _tokenize_rdd(rdd: RDD[str]) -> RDD[str]
    - _filter_rdd(rdd: RDD[str]) -> RDD[str]
    - _count_frequencies_rdd(rdd: RDD[str]) -> Dict[str, int]
    - _collect_statistics(frequency_rdd: RDD[Tuple[str, int]]) -> Statistics
```

**Algorithm Flow**:
1. Initialize SparkSession (local mode, use all cores)
2. Load `data.txt` as RDD (Resilient Distributed Dataset)
3. Transformations:
   - `flatMap`: Split lines into words
   - `map`: Normalize words (lowercase, remove punctuation)
   - `filter`: Remove words with length < 4 or > 8
   - `filter`: Remove stop words
4. Actions:
   - `countByValue()`: Count word frequencies
   - Collect results to driver
5. Compute statistics from frequency dictionary

**Spark Optimizations**:
- Repartition RDD for better parallelism
- Use broadcast variables for stop words (efficient distribution)
- Cache intermediate RDDs if needed
- Minimize shuffles

**RDD Operations**:
```python
# Pseudo-code structure
rdd = spark_context.textFile("data.txt")
words = rdd.flatMap(lambda line: tokenize(line))
normalized = words.map(lambda w: normalize(w))
filtered_length = normalized.filter(lambda w: 4 <= len(w) <= 8)
stop_words_broadcast = spark_context.broadcast(stop_words_set)
filtered_stop = filtered_length.filter(lambda w: w not in stop_words_broadcast.value)
frequencies = filtered_stop.countByValue()
```

## 6. Utility Classes

### 6.1 Word Mapper (for GPU)

**File**: `utils/word_mapper.py`

```python
class WordMapper:
    - word_to_id: Dict[str, int]
    - id_to_word: Dict[int, str]
    - next_id: int
    
    + create_dictionary(words: List[str]) -> Dict[str, int]
    + words_to_ids(words: List[str]) -> np.ndarray
    + ids_to_words(ids: np.ndarray) -> List[str]
    + get_stop_word_ids(stop_words: Set[str]) -> Set[int]
```

### 6.2 Text Preprocessor

**File**: `utils/text_preprocessor.py`

```python
class TextPreprocessor:
    + normalize_word(word: str) -> str
    + tokenize_line(line: str) -> List[str]
    + remove_punctuation(word: str) -> str
    + to_lowercase(word: str) -> str
```

### 6.3 Performance Timer

**File**: `utils/performance_timer.py`

```python
class PerformanceTimer:
    + start() -> None
    + stop() -> float  # Returns elapsed time in seconds
    + context_manager() -> ContextManager
```

**Usage**:
```python
timer = PerformanceTimer()
with timer:
    results = processor.process()
processing_time = timer.elapsed_time
```

## 7. Visualization Module

### 7.1 Chart Generator

**File**: `visualization/chart_generator.py`

```python
class ChartGenerator:
    + create_pie_chart(results: List[ProcessingResults]) -> None
    + save_chart(filename: str) -> None
    + display_chart() -> None
```

**Chart Design**:
- Pie chart showing processing time distribution
- Each slice: algorithm name + time (seconds)
- Legend with percentage and absolute time
- Title: "Text Processing Algorithm Performance Comparison"

## 8. Main Application

**File**: `main.py`

```python
class TextProcessingApplication:
    - processors: List[BaseProcessor]
    - results: List[ProcessingResults]
    
    + run_all_algorithms() -> List[ProcessingResults]
    + print_results(results: List[ProcessingResults]) -> None
    + generate_chart(results: List[ProcessingResults]) -> None
    + main() -> None
```

**Execution Flow**:
1. Initialize all 4 processors
2. Run each processor sequentially (with timing)
3. Collect all results
4. Print statistics for each algorithm to console
5. Generate pie chart comparing processing times
6. Display chart

## 9. Configuration Management

**File**: `config.py` (optional)

```python
@dataclass
class Config:
    data_file: str = "data.txt"
    stop_words_file: str = "stop_words.txt"
    min_word_length: int = 4
    max_word_length: int = 8
    num_threads: int = None  # Auto-detect if None
    gpu_device_id: int = 0
    spark_master: str = "local[*]"
    output_chart_file: str = "performance_chart.png"
```

## 10. Error Handling Strategy

### Exception Hierarchy
```python
class TextProcessingError(Exception): pass
class FileLoadError(TextProcessingError): pass
class ProcessingError(TextProcessingError): pass
class GPUError(ProcessingError): pass
class SparkError(ProcessingError): pass
```

### Error Handling Points
- File loading: Handle missing files, encoding issues
- GPU: Handle GPU availability, memory errors
- Spark: Handle Spark initialization failures
- Processing: Handle empty results, division by zero

## 11. Testing Strategy

### Unit Tests
- Test each processor independently
- Mock file I/O
- Test filtering logic
- Test statistics computation

### Integration Tests
- Test full pipeline for each processor
- Compare results across processors (should be identical)

### Performance Tests
- Benchmark each algorithm
- Test with different data sizes

## 12. Dependencies

### requirements.txt
```
# Core dependencies
numpy>=1.21.0
matplotlib>=3.5.0

# CPU multi-threading
# (built-in: concurrent.futures, multiprocessing)

# GPU processing
cupy>=10.0.0  # or numba>=0.56.0, or pycuda>=2021.1

# Apache Spark
pyspark>=3.2.0

# Utilities
tqdm>=4.62.0  # Optional: progress bars
```

## 13. Algorithm-Specific Considerations

### 13.1 CPU Single-Threaded
- **Strengths**: Simple, no overhead, easy to debug
- **Weaknesses**: Slow for large datasets
- **Use Case**: Baseline comparison, small datasets

### 13.2 CPU Multi-Threaded
- **Strengths**: Utilizes all CPU cores, good speedup
- **Weaknesses**: GIL limitations in Python (use multiprocessing if needed)
- **Optimization**: Use `multiprocessing.Pool` instead of threads for true parallelism
- **Chunking Strategy**: Divide file into equal-sized chunks per core

### 13.3 GPU
- **Strengths**: Massive parallelism for numerical operations
- **Weaknesses**: 
  - String operations not GPU-friendly (requires mapping)
  - Memory transfer overhead
  - Complex setup
- **Optimization**:
  - Batch processing for large datasets
  - Minimize CPU-GPU transfers
  - Use efficient CUDA kernels

### 13.4 Apache Spark
- **Strengths**: Distributed processing, fault tolerance, scalable
- **Weaknesses**: 
  - Overhead for small datasets
  - Requires JVM
- **Optimization**:
  - Use broadcast variables for stop words
  - Proper partitioning
  - Minimize shuffles

## 14. Data Flow Diagram

```
Input Files
    ↓
[DataLoader] → Tokenized Words
    ↓
[WordFilter] → Filtered Words (length + stop words)
    ↓
[StatisticsCalculator] → Statistics
    ↓
[ProcessingResults] → Output
```

**For each processor**:
- Same logical flow
- Different implementation strategies
- Same final results (should be identical)

## 15. Performance Measurement Strategy

### Timing Approach
1. **Warm-up run**: Run each algorithm once (discard results) to account for JIT compilation, GPU initialization, Spark startup
2. **Actual measurement**: Run each algorithm and measure time
3. **Multiple runs**: Optionally run multiple times and average (for more accurate results)

### What to Measure
- **Total processing time**: From start to finish
- **Breakdown** (optional):
  - Data loading time
  - Processing time
  - Statistics computation time

### Chart Visualization
- Pie chart with 4 slices (one per algorithm)
- Each slice labeled with algorithm name and time
- Colors: Different color per algorithm
- Include percentage and absolute time in legend

## 16. Output Format

### Console Output Example
```
=== Text Processing Results ===

Algorithm: CPU Single-Threaded
Processing Time: 2.45 seconds
Most Frequent Word: "the" (occurrences: 1234)
Least Frequent Word: "xyz" (occurrences: 1)
Total Words: 45678

Algorithm: CPU Multi-Threaded
Processing Time: 0.67 seconds
...

Algorithm: GPU
Processing Time: 0.89 seconds
...

Algorithm: Apache Spark
Processing Time: 1.23 seconds
...

=== Performance Comparison Chart ===
Chart saved to: performance_chart.png
```

## 17. Implementation Priority

1. **Phase 1**: Core infrastructure
   - Base classes, data loading, filtering, statistics
   - CPU single-threaded (baseline)

2. **Phase 2**: Parallel CPU
   - Multi-threaded implementation
   - Testing and optimization

3. **Phase 3**: GPU
   - Word mapping utilities
   - GPU processing implementation
   - Memory management

4. **Phase 4**: Apache Spark
   - Spark setup and configuration
   - RDD transformations and actions
   - Optimization

5. **Phase 5**: Integration
   - Main application
   - Visualization
   - Testing and validation

## 18. Key Design Decisions

1. **Abstract Base Class**: Ensures consistent interface across all processors
2. **Template Method Pattern**: Common workflow with algorithm-specific steps
3. **Separation of Concerns**: Data loading, filtering, statistics are separate modules
4. **Strategy Pattern**: Easy to add new algorithms
5. **Result Objects**: Structured data for results and statistics
6. **GPU Preprocessing**: Length filtering on CPU (as per requirements)
7. **Spark Broadcast**: Efficient stop word distribution

## 19. Potential Challenges & Solutions

### Challenge 1: Python GIL for Multi-threading
**Solution**: Use `multiprocessing` instead of `threading` for true parallelism

### Challenge 2: GPU String Operations
**Solution**: Map words to integers, process integers on GPU, map back

### Challenge 3: Memory Management (GPU)
**Solution**: Batch processing, efficient memory transfers

### Challenge 4: Spark Overhead
**Solution**: Use local mode efficiently, proper partitioning

### Challenge 5: Result Consistency
**Solution**: All algorithms should produce identical results (validation tests)

## 20. Extension Points

- Add more algorithms (e.g., Dask, Ray)
- Add progress tracking
- Add detailed performance metrics
- Add memory usage tracking
- Add distributed Spark (multi-node)
- Add real-time processing
- Add web interface

---

## Summary

This OOP design provides:
- ✅ Clear separation of concerns
- ✅ Extensible architecture (easy to add new algorithms)
- ✅ Consistent interface across all processors
- ✅ Reusable components (filters, statistics, loaders)
- ✅ Comprehensive error handling
- ✅ Performance measurement and visualization
- ✅ Testable components

The structure follows SOLID principles and common design patterns, making it maintainable and scalable.

