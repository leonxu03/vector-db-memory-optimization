"""
Query one or more Qdrant collections with the SIFT1M query set and measure recall and latency.

Usage:
    python query.py                                     # defaults to "baseline"
    python query.py --c <collection_name> [<collection_name> ...]
    python query.py --c <collection_name> --batch-size 100
    python query.py --c <collection_name> --progress-file <path>   # for monitor_query.py
    python query.py --c <collection_name> --save-csv false         # don't save to the CSV
    python query.py --c <collection_name> --restart false          # keep Qdrant + page cache warm

For each collection, in order:
    1. Cool down for --cooldown seconds (skipped for the first collection).
    2. Cold start (--restart true, the default): restart Qdrant, drop the OS page cache.
    3. Send all 10,000 queries, --batch-size queries per request, and time each request.
    4. Print the results and append one row to results/query_results.csv (--save-csv true, the default).
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models

from qdrant_restart import QDRANT_URL, RestartError, drop_caches, restart_qdrant

DEFAULT_COLLECTION = "baseline"
QUERY_FILE_PATH = "dataset/sift_query.fvecs"
GROUNDTRUTH_FILE_PATH = "dataset/sift_groundtruth.ivecs"
RESULTS_CSV_PATH = "results/query_results.csv"

TOP_K = 10
EXPECTED_QUERIES = 10_000
EXPECTED_POINTS = 1_000_000
BATCH_SIZES = [1, 10, 100, 1000, 10000]

CLIENT_TIMEOUT_S = 600
COOLDOWN_S = 30  
PROGRESS_EVERY = 10 

CSV_FIELDS = [
    "timestamp", "collection", "batch_size", "queries", "requests", "top_k",
    f"recall_at_{TOP_K}_pct",
    "avg_query_ms", "median_query_ms", "p95_query_ms", "p99_query_ms",
    "avg_request_ms", "median_request_ms", "p95_request_ms", "p99_request_ms",
    "total_s", "qps",
]


class StepError(Exception):
    """Any failure; carries the reason to print."""


def read_vecs(path, dtype):
    """Read an .fvecs / .ivecs file into an (n, dim) array.

    Each record is [dim: int32][dim values: float32 or int32], so every row is dim + 1 int32s.
    """
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    values = raw.reshape(-1, dim + 1)[:, 1:].copy()  # drop the leading dim of every record
    return values.view(dtype)


def load_dataset():
    """Return (queries, groundtruth): 10,000 query vectors and each one's true nearest ids."""
    queries = read_vecs(QUERY_FILE_PATH, np.float32)
    groundtruth = read_vecs(GROUNDTRUTH_FILE_PATH, np.int32)
    if len(queries) != EXPECTED_QUERIES or len(groundtruth) != EXPECTED_QUERIES:
        raise StepError(f"Expected {EXPECTED_QUERIES} queries and ground-truth rows, "
                        f"got {len(queries)} and {len(groundtruth)}.")
    return queries, groundtruth


def check_collection(client, collection):
    """Raise StepError unless the collection exists and holds the full dataset."""
    if not client.collection_exists(collection):
        raise StepError(f"Collection '{collection}' does not exist. Run main.py first.")
    points = client.count(collection, exact=True).count
    if points != EXPECTED_POINTS:
        raise StepError(f"'{collection}' holds {points} points, expected {EXPECTED_POINTS}.")


def cold_start(collection):
    """Restart Qdrant, then drop the OS page cache."""
    print(f"\n=== Cold start before querying {collection} ===", flush=True)
    try:
        restart_qdrant()
        drop_caches()
    except RestartError as exc:
        raise StepError(str(exc)) from exc


def build_requests(batch):
    """One QueryRequest per query vector (built before the timer starts)."""
    return [models.QueryRequest(query=vector.tolist(), limit=TOP_K,
                                with_payload=False, with_vector=False)
            for vector in batch]


def send(client, collection, requests):
    """Send the requests and return one response per query.

    A single query uses the plain search endpoint; several go in one batch request.
    """
    if len(requests) == 1:
        return [client.query_points(collection_name=collection, query=requests[0].query,
                                    limit=TOP_K, with_payload=False, with_vectors=False)]
    return client.query_batch_points(collection_name=collection, requests=requests)


def count_hits(responses, truths):
    """Number of returned ids that are among each query's true TOP_K nearest neighbours."""
    hits = 0
    for response, truth in zip(responses, truths):
        returned = {point.id for point in response.points}
        hits += len(returned & set(truth[:TOP_K].tolist()))
    return hits


def write_progress(path, done, total):
    """Atomically write 'done/total' so a monitor never reads a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(f"{done}/{total}\n")
    os.replace(tmp, path)


@dataclass
class RunResult:
    collection: str
    batch_size: int
    request_ms: np.ndarray  # latency of each request
    query_ms: np.ndarray    # latency of each query = its request's latency / queries in it
    recall_pct: float       # Recall@TOP_K, in %
    total_s: float          # wall time for all queries

    @property
    def queries(self):
        return len(self.query_ms)

    @property
    def requests(self):
        return len(self.request_ms)

    @property
    def qps(self):
        return self.queries / self.total_s


def run_queries(client, collection, queries, groundtruth, batch_size, progress_file=None):
    """Send every query in requests of batch_size and time each request."""
    total = len(queries)
    request_ms, query_ms = [], []
    hits = done = 0
    if progress_file:
        write_progress(progress_file, 0, total)

    run_start = time.perf_counter()
    with tqdm(total=total, desc=f"Querying {collection} (batch {batch_size})") as bar:
        for first in range(0, total, batch_size):
            batch = queries[first:first + batch_size]
            requests = build_requests(batch)

            start = time.perf_counter()
            responses = send(client, collection, requests)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            request_ms.append(elapsed_ms)
            query_ms.extend([elapsed_ms / len(batch)] * len(batch))
            hits += count_hits(responses, groundtruth[first:first + batch_size])

            # Update the progress file each time another PROGRESS_EVERY queries finish
            previous, done = done, done + len(batch)
            bar.update(len(batch))
            crossed_step = done // PROGRESS_EVERY > previous // PROGRESS_EVERY
            if progress_file and (crossed_step or done == total):
                write_progress(progress_file, done, total)
    total_s = time.perf_counter() - run_start

    return RunResult(collection=collection, batch_size=batch_size,
                     request_ms=np.array(request_ms), query_ms=np.array(query_ms),
                     recall_pct=hits / (total * TOP_K) * 100.0, total_s=total_s)


def latency_stats(ms):
    """(mean, median, P95, P99) of a latency array, in ms."""
    return (float(ms.mean()), float(np.median(ms)),
            float(np.percentile(ms, 95)), float(np.percentile(ms, 99)))


def csv_row(result):
    query_avg, query_median, query_p95, query_p99 = latency_stats(result.query_ms)
    request_avg, request_median, request_p95, request_p99 = latency_stats(result.request_ms)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "collection": result.collection,
        "batch_size": result.batch_size,
        "queries": result.queries,
        "requests": result.requests,
        "top_k": TOP_K,
        f"recall_at_{TOP_K}_pct": round(result.recall_pct, 4),
        "avg_query_ms": round(query_avg, 4),
        "median_query_ms": round(query_median, 4),
        "p95_query_ms": round(query_p95, 4),
        "p99_query_ms": round(query_p99, 4),
        "avg_request_ms": round(request_avg, 4),
        "median_request_ms": round(request_median, 4),
        "p95_request_ms": round(request_p95, 4),
        "p99_request_ms": round(request_p99, 4),
        "total_s": round(result.total_s, 3),
        "qps": round(result.qps, 2),
    }


def append_csv(row):
    """Append one row to RESULTS_CSV_PATH, writing the header first if the file is new."""
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    new_file = not os.path.exists(RESULTS_CSV_PATH) or os.path.getsize(RESULTS_CSV_PATH) == 0
    with open(RESULTS_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def print_summary(result):
    query_avg, query_median, query_p95, query_p99 = latency_stats(result.query_ms)
    print(f"\nCollection:   {result.collection}")
    print(f"Queries:      {result.queries} (top-{TOP_K}, Euclidean), "
          f"batch size {result.batch_size} -> {result.requests} requests")
    print(f"Recall@{TOP_K}:    {result.recall_pct:.2f}%")
    print(f"Avg latency:  {query_avg:.3f} ms/query")
    print(f"Median:       {query_median:.3f} ms/query")
    print(f"P95:          {query_p95:.3f} ms/query")
    print(f"P99:          {query_p99:.3f} ms/query")
    if result.batch_size > 1:
        request_avg, _, _, request_p99 = latency_stats(result.request_ms)
        print(f"Avg request:  {request_avg:.3f} ms (P99 {request_p99:.3f} ms)")
    print(f"Throughput:   {result.qps:.1f} queries/s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c", nargs="+", default=[DEFAULT_COLLECTION], metavar="COLLECTION",
                        help="One or more Qdrant collections to test, run in the order given "
                             f"(default: {DEFAULT_COLLECTION})")
    parser.add_argument("--batch-size", type=int, default=1, choices=BATCH_SIZES,
                        help="Queries sent per request (default: 1, i.e. sequential)")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S,
                        help=f"Seconds to wait between collections (default: {COOLDOWN_S})")
    parser.add_argument("--progress-file",
                        help="Write 'completed/total' queries to this file while running")
    parser.add_argument("--restart", choices=["true", "false"], default="true",
                        help="Restart Qdrant and drop the OS page cache before each collection "
                             "(default: true)")
    parser.add_argument("--save-csv", choices=["true", "false"], default="true",
                        help=f"Append results to {RESULTS_CSV_PATH} "
                             "(default: true; false e.g. under valgrind)")
    return parser.parse_args()


def main():
    args = parse_args()
    collections = list(dict.fromkeys(args.c))  # de-duplicate, keep order
    queries, groundtruth = load_dataset()

    client = QdrantClient(url=QDRANT_URL, timeout=CLIENT_TIMEOUT_S)
    for collection in collections:  # fail fast, before spending time querying
        check_collection(client, collection)

    for i, collection in enumerate(collections):
        if i > 0 and args.cooldown > 0:
            print(f"\nCooling down {args.cooldown:g}s before querying {collection}...", flush=True)
            time.sleep(args.cooldown)
        if args.restart == "true":
            cold_start(collection)

        result = run_queries(client, collection, queries, groundtruth,
                             args.batch_size, args.progress_file)
        if args.save_csv == "true":
            append_csv(csv_row(result))
        print_summary(result)

    if args.save_csv == "true":
        print(f"\nResults appended to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except StepError as exc:
        sys.exit(f"ERROR: {exc}")
