"""
Benchmark to compare ThreadPoolExecutor vs sequential logging for multiple traces.

This benchmark simulates the OTEL endpoint behavior where spans from multiple traces
need to be logged. It compares the performance of:
1. Sequential logging: logging each trace's spans one after another
2. Parallel logging: using ThreadPoolExecutor to log traces in parallel
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan

import mlflow
from mlflow.entities import Span
from mlflow.server.handlers import _get_tracking_store


def create_test_spans(num_traces: int, spans_per_trace: int) -> dict[str, list]:
    """Create OTel protobuf spans grouped by trace_id."""
    spans_by_trace = defaultdict(list)

    for trace_idx in range(num_traces):
        # Generate a proper 16-byte trace ID
        trace_id_hex = f"{trace_idx:032x}"
        trace_id_bytes = bytes.fromhex(trace_id_hex)

        for span_idx in range(spans_per_trace):
            span = OTelProtoSpan()
            span.trace_id = trace_id_bytes
            # Generate a proper 8-byte span ID
            span_id_hex = f"{(trace_idx * 1000 + span_idx):016x}"
            span.span_id = bytes.fromhex(span_id_hex)
            span.name = f"test-span-{trace_idx}-{span_idx}"
            span.start_time_unix_nano = int(time.time() * 1e9)
            span.end_time_unix_nano = int(time.time() * 1e9) + 1000000

            # Convert to MLflow span for storage
            mlflow_span = Span.from_otel_proto(span)
            spans_by_trace[mlflow_span.trace_id].append(mlflow_span)

    return spans_by_trace


def sequential_logging(experiment_id: str, spans_by_trace: dict[str, list]) -> float:
    """Log spans sequentially, one trace at a time."""
    store = _get_tracking_store()

    start = time.perf_counter()
    for trace_id, trace_spans in spans_by_trace.items():
        store.log_spans(experiment_id, trace_spans)
    elapsed = time.perf_counter() - start

    return elapsed


def parallel_logging(
    experiment_id: str, spans_by_trace: dict[str, list], max_workers: int = 5
) -> float:
    """Log spans in parallel using ThreadPoolExecutor."""
    store = _get_tracking_store()

    def log_trace_spans(trace_id: str, trace_spans: list) -> str:
        store.log_spans(experiment_id, trace_spans)
        return trace_id

    start = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="BenchmarkWorker",
    ) as executor:
        future_to_trace = {
            executor.submit(log_trace_spans, trace_id, trace_spans): trace_id
            for trace_id, trace_spans in spans_by_trace.items()
        }

        # Wait for all tasks to complete
        for future in as_completed(future_to_trace):
            future.result()
    elapsed = time.perf_counter() - start

    return elapsed


def run_benchmark(
    num_traces_list: list[int] = [1, 5, 10, 20],
    spans_per_trace: int = 10,
    max_workers: int = 5,
):
    """Run benchmark comparing sequential vs parallel logging."""
    print("=" * 80)
    print("Benchmark: Sequential vs Parallel Span Logging")
    print("=" * 80)
    print(f"Spans per trace: {spans_per_trace}")
    print(f"ThreadPoolExecutor max_workers: {max_workers}")
    print("=" * 80)
    print()

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///benchmark.db")
    experiment = mlflow.set_experiment("benchmark-thread-pool")
    experiment_id = experiment.experiment_id

    results = []

    for num_traces in num_traces_list:
        print(f"\n{'=' * 80}")
        print(f"Number of traces: {num_traces} (total spans: {num_traces * spans_per_trace})")
        print(f"{'=' * 80}")

        # Create test data for sequential
        spans_by_trace_seq = create_test_spans(num_traces, spans_per_trace)

        # Benchmark sequential logging
        seq_time = sequential_logging(experiment_id, spans_by_trace_seq)
        print(f"Sequential logging ({num_traces} traces): {seq_time:.4f} seconds")

        # Create fresh test data for parallel test
        spans_by_trace_par = create_test_spans(num_traces, spans_per_trace)

        # Benchmark parallel logging
        par_time = parallel_logging(experiment_id, spans_by_trace_par, max_workers)
        print(f"Parallel logging ({num_traces} traces, {max_workers} workers): {par_time:.4f} seconds")

        # Calculate speedup
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"✓ Parallel is {speedup:.2f}x faster")
        elif speedup < 1:
            print(f"✗ Sequential is {1/speedup:.2f}x faster")
        else:
            print("≈ No significant difference")

        results.append({
            "num_traces": num_traces,
            "total_spans": num_traces * spans_per_trace,
            "sequential_time": seq_time,
            "parallel_time": par_time,
            "speedup": speedup,
        })

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Traces':<10} {'Total Spans':<15} {'Sequential (s)':<18} {'Parallel (s)':<18} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['num_traces']:<10} {r['total_spans']:<15} "
            f"{r['sequential_time']:<18.4f} {r['parallel_time']:<18.4f} "
            f"{r['speedup']:<10.2f}x"
        )
    print("=" * 80)

    # Provide analysis
    print("\nAnalysis:")
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}x")

    if avg_speedup > 1.5:
        print("✓ ThreadPoolExecutor provides significant performance improvement")
    elif avg_speedup > 1.1:
        print("~ ThreadPoolExecutor provides moderate performance improvement")
    else:
        print("✗ ThreadPoolExecutor does not provide meaningful performance improvement")


if __name__ == "__main__":
    import sys

    # Allow customization via command line
    num_traces_list = [1, 5, 10, 20]
    spans_per_trace = 10
    max_workers = 5

    if len(sys.argv) > 1:
        # Parse custom parameters
        if "--help" in sys.argv or "-h" in sys.argv:
            print("Usage: python benchmark_thread_pool.py [--traces N1,N2,N3] [--spans-per-trace N] [--workers N]")
            print()
            print("Options:")
            print("  --traces N1,N2,N3     Comma-separated list of trace counts to test (default: 1,5,10,20)")
            print("  --spans-per-trace N   Number of spans per trace (default: 10)")
            print("  --workers N           Max workers for ThreadPoolExecutor (default: 5)")
            sys.exit(0)

        for i, arg in enumerate(sys.argv):
            if arg == "--traces" and i + 1 < len(sys.argv):
                num_traces_list = [int(x) for x in sys.argv[i + 1].split(",")]
            elif arg == "--spans-per-trace" and i + 1 < len(sys.argv):
                spans_per_trace = int(sys.argv[i + 1])
            elif arg == "--workers" and i + 1 < len(sys.argv):
                max_workers = int(sys.argv[i + 1])

    run_benchmark(num_traces_list, spans_per_trace, max_workers)
