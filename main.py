#!/usr/bin/env python3
"""Idempotently create several SIFT1M collections (one per quantization) and load them."""

import argparse
import os
import struct
import sys
from itertools import count

from tqdm import tqdm
from qdrant_client import QdrantClient, models

# --- CONFIGURATION ---------------------------------------------------------
FVECS_FILE_PATH = "dataset/sift_base.fvecs"
QDRANT_URL = "http://localhost:6333"
DIM = 128
EXPECTED_COUNT = 1_000_000
DISTANCE = models.Distance.EUCLID
BATCH_SIZE = 256
PARALLEL = 2

# --- QUANTIZATION JOBS QUEUE -------------------------------------------------
QUANTIZATION_JOBS = {
    # 1. Baseline: No compression
    "sift1m_baseline": None,

    # 2. Product Quantization
    "pq_x4": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X4,
        )
    ),
    "pq_x8": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X8,
        )
    ),
    "pq_x16": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X16,
        )
    ),
    "pq_x32": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X32,
        )
    ),
    "pq_x64": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X64,
        )
    ),

    "scalar": models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
        )
    )
    # TODO: 4. Binary Quantization
}


class StepError(Exception):
    """Any failure; carries the reason to print."""


def stream_fvecs(filename):
    """Yield vectors from an .fvecs file one at a time ([dim:int32][dim x float32] records)."""
    with open(filename, "rb") as f:
        while True:
            dim_bin = f.read(4)
            if not dim_bin:
                break
            (dim,) = struct.unpack("<i", dim_bin)
            yield list(struct.unpack(f"<{dim}f", f.read(dim * 4)))


def _quant_name(q):
    if q is None:
        return "none"
    if isinstance(q, models.ScalarQuantization):
        return "scalar"
    if isinstance(q, models.ProductQuantization):
        comp = q.product.compression
        return f"product:{getattr(comp, 'value', comp)}"
    if isinstance(q, models.BinaryQuantization):
        return "binary"
    return type(q).__name__


def create_collection_if_not_exists(client, collection_name, quantization_config):
    if not client.collection_exists(collection_name):
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=DIM, distance=DISTANCE),
                quantization_config=quantization_config,
            )
            return "created"
        except Exception as exc:
            if not client.collection_exists(collection_name):
                raise StepError(f"Could not create '{collection_name}': {exc}") from exc

    # CHECK IF COLLECTION IS CONFIGURED CORRECTLY
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict) or vectors.size != DIM or vectors.distance != DISTANCE:
        raise StepError(f"'{collection_name}' exists with a different vector configuration.")
    have, want = _quant_name(info.config.quantization_config), _quant_name(quantization_config)
    if have != want:
        raise StepError(
            f"'{collection_name}' exists with quantization '{have}', expected '{want}'. "
            f"Delete it yourself or use another name."
        )
    return "exists"


def load_collection(client, collection_name):
    """Upload SIFT1M with ids = row index, so re-running overwrites instead of duplicating."""
    existing = client.count(collection_name, exact=True).count
    if existing == EXPECTED_COUNT:
        return "skipped (already loaded)"
    if existing > EXPECTED_COUNT:
        raise StepError(f"'{collection_name}' holds {existing} points (> {EXPECTED_COUNT}).")

    vectors = tqdm(stream_fvecs(FVECS_FILE_PATH), total=EXPECTED_COUNT,
                   desc=f"Uploading {collection_name}")
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        ids=count(),
        batch_size=BATCH_SIZE,
        parallel=PARALLEL,
    )

    final = client.count(collection_name, exact=True).count
    if final != EXPECTED_COUNT:
        raise StepError(f"'{collection_name}' holds {final} points after upload, expected {EXPECTED_COUNT}.")
    return "uploaded"


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--quantization", choices=QUANTIZATION_JOBS.keys())
    args = parser.parse_args()

    if not os.path.exists(FVECS_FILE_PATH):
        raise StepError(f"Could not find SIFT1M file at '{FVECS_FILE_PATH}'.")

    client = QdrantClient(url=QDRANT_URL)
    client.get_collections()

    jobs = QUANTIZATION_JOBS if args.all else {
        args.quantization: QUANTIZATION_JOBS[args.quantization]
    }

    for name, quant in jobs.items():
        result = create_collection_if_not_exists(client, name, quant)
        print(f"Collection {name}: {result}")
        result = load_collection(client, name)
        print(f"Upload result: {result}")

    print("\nSUCCESS: selected collections are ready. Verify with:\n")
    for name in jobs:
        print(f"curl -s {QDRANT_URL}/collections/{name} | python -m json.tool")
        print(f"curl -s -X POST {QDRANT_URL}/collections/{name}/points/count "
              f"-H 'Content-Type: application/json' -d '{{\"exact\": true}}'\n")


if __name__ == "__main__":
    main()