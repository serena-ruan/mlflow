# ThreadPoolExecutor Benchmark Results for OTel Span Logging

## Summary

Benchmark comparing sequential vs parallel span logging using ThreadPoolExecutor for the OTel endpoint.

**Conclusion**: ThreadPoolExecutor **does not provide meaningful performance improvement** and in most cases **degrades performance** compared to sequential logging when using SQLite as the backend store.

## Test Configuration

- **Backend**: SQLite database (default for development)
- **Spans per trace**: 10
- **Test scenarios**: Various trace counts (1, 5, 10, 20, 50, 100)

## Results

### Test 1: Default Configuration (max_workers=5)

| Traces | Total Spans | Sequential (s) | Parallel (s) | Speedup | Result |
|--------|-------------|----------------|--------------|---------|--------|
| 1      | 10          | 0.0112         | 0.0055       | 2.06x   | ✓ Parallel faster |
| 5      | 50          | 0.0356         | 0.0542       | 0.66x   | ✗ Sequential faster (1.52x) |
| 10     | 100         | 0.0688         | 0.1076       | 0.64x   | ✗ Sequential faster (1.57x) |
| 20     | 200         | 0.1817         | 0.1602       | 1.13x   | ~ Marginal parallel improvement |

**Average speedup: 1.12x** - Moderate improvement overall, but **sequential is faster for typical batch sizes**.

### Test 2: Higher Parallelism (max_workers=10)

| Traces | Total Spans | Sequential (s) | Parallel (s) | Speedup | Result |
|--------|-------------|----------------|--------------|---------|--------|
| 1      | 10          | 0.0105         | 0.0063       | 1.68x   | ✓ Parallel faster |
| 10     | 100         | 0.0827         | 0.1697       | 0.49x   | ✗ Sequential faster (2.05x) |
| 50     | 500         | 0.3457         | 0.3746       | 0.92x   | ✗ Sequential faster (1.08x) |
| 100    | 1000        | 0.6654         | 0.7922       | 0.84x   | ✗ Sequential faster (1.19x) |

**Average speedup: 0.98x** - ThreadPoolExecutor provides **no meaningful performance improvement**.

## Analysis

### Why ThreadPoolExecutor Doesn't Help (with SQLite)

1. **Database Write Locking**: SQLite uses a single-writer model. When multiple threads try to write concurrently, they must wait for locks, negating the benefits of parallelism.

2. **Thread Overhead**: Creating and managing threads introduces overhead:
   - Thread creation cost
   - Context switching between threads
   - Synchronization overhead

3. **Python GIL**: The Global Interpreter Lock limits true parallelism for CPU-bound operations in Python.

4. **Small Operation Size**: Each `log_spans()` call is relatively fast (~3-10ms). The thread overhead becomes significant relative to the actual work.

### When Would ThreadPoolExecutor Help?

ThreadPoolExecutor would provide benefits in these scenarios:

1. **Remote/Network Backend**: If the backend store is remote (e.g., MySQL, PostgreSQL on another server), parallel requests could overlap network I/O latency.

2. **I/O-Bound Operations**: If span logging involves significant I/O wait time (network calls, slow disk), parallelism could help.

3. **Very Large Batches**: With hundreds of traces, there might be modest improvements, but contention would still be an issue.

4. **Backend with Better Concurrency**: Databases like PostgreSQL handle concurrent writes better than SQLite.

## Recommendation

For the OTel endpoint with typical usage patterns:

1. **Remove ThreadPoolExecutor**: Sequential logging is simpler, faster, and more predictable for SQLite backends.

2. **Alternative Approaches**:
   - **Batch all spans together**: Instead of grouping by trace_id, log all spans in a single transaction if the backend supports it.
   - **Conditional parallelism**: Only use ThreadPoolExecutor for remote backends (not SQLite).
   - **Async I/O**: Consider using async/await patterns instead of threads if the backend supports it.

3. **If keeping ThreadPoolExecutor**:
   - Add a configuration option to disable it for SQLite
   - Consider lowering max_workers to 2-3 to reduce contention
   - Document that it's primarily for remote backend stores

## Reproduction

To reproduce these benchmarks:

```bash
# Default configuration
python benchmark_thread_pool.py

# Custom configuration
python benchmark_thread_pool.py --traces 1,10,50,100 --workers 10 --spans-per-trace 10
```

## Related PR Discussion

This benchmark was created in response to: https://github.com/mlflow/mlflow/pull/18536#discussion_r2471544516
