"""
Usage:
    python test_collection.py                              # sift1m_baseline
    python test_collection.py --collection sift1m_pq_x4    # TODO: needs to be added
"""

import argparse
import sys
import time

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient

# --- CONFIGURATION ---------------------------------------------------------
DEFAULT_COLLECTION = "sift1m_baseline"
QDRANT_URL = "http://localhost:6333"
QUERY_FILE_PATH = "dataset/sift_query.fvecs"
GROUNDTRUTH_FILE_PATH = "dataset/sift_groundtruth.ivecs"
TOP_K = 10
EXPECTED_QUERIES = 10_000
EXPECTED_POINTS = 1_000_000


class StepError(Exception):
    """Any failure; carries the reason to print."""


def read_fvecs(path):
    """Read an .fvecs file ([dim:int32][dim x float32] records) into an (n, dim) array."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path):
    """Read an .ivecs file ([dim:int32][dim x int32] records) into an (n, dim) array."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


def run_experiment(client, collection, queries, groundtruth):
    """Send every query one at a time; return per-query latencies (ms) and Recall@10 (%)."""
    latencies_ms = []
    hits = 0

    for query, truth in tqdm(zip(queries, groundtruth), total=len(queries),
                             desc=f"Querying {collection}"):
        vector = query.tolist()  # conversion kept outside the timed region

        start = time.perf_counter()
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=TOP_K,
            with_payload=False,
            with_vectors=False,
        )
        elapsed = time.perf_counter() - start

        latencies_ms.append(elapsed * 1000.0)
        retrieved = {point.id for point in response.points}
        hits += len(retrieved & set(truth[:TOP_K].tolist()))

    recall = hits / (len(queries) * TOP_K) * 100.0
    return np.array(latencies_ms), recall


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection to test (default: {DEFAULT_COLLECTION})")
    args = parser.parse_args()

    queries = read_fvecs(QUERY_FILE_PATH)
    groundtruth = read_ivecs(GROUNDTRUTH_FILE_PATH)
    if len(queries) != EXPECTED_QUERIES or len(groundtruth) != EXPECTED_QUERIES:
        raise StepError(f"Expected {EXPECTED_QUERIES} queries and ground-truth rows, "
                        f"got {len(queries)} and {len(groundtruth)}.")

    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(args.collection):
        raise StepError(f"Collection '{args.collection}' does not exist. Run main.py first.")
    count = client.count(args.collection, exact=True).count
    if count != EXPECTED_POINTS:
        raise StepError(f"'{args.collection}' holds {count} points, expected {EXPECTED_POINTS}.")

    latencies, recall = run_experiment(client, args.collection, queries, groundtruth)

    print(f"\nCollection:   {args.collection}")
    print(f"Queries:      {len(queries)} (top-{TOP_K}, Euclidean)")
    print(f"Recall@{TOP_K}:    {recall:.2f}%")
    print(f"Avg latency:  {latencies.mean():.3f} ms")
    print(f"Median:       {np.median(latencies):.3f} ms")
    print(f"P95:          {np.percentile(latencies, 95):.3f} ms")
    print(f"P99:          {np.percentile(latencies, 99):.3f} ms")


if __name__ == "__main__":
    try:
        main()
    except StepError as exc:
        sys.exit(f"ERROR: {exc}")