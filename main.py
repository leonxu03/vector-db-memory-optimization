#!/usr/bin/env python3

import argparse
import os
import shlex
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count

from tqdm import tqdm
from qdrant_client import QdrantClient, models

from qdrant_restart import RestartError, drop_caches, restart_qdrant, QDRANT_URL

# --- CONFIGURATION -----------------------------------------------------------
FVECS_FILE_PATH = "dataset/sift_base.fvecs"
ALIASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_aliases.sh")

DIM = 128
EXPECTED_COUNT = 1_000_000
DISTANCE = models.Distance.EUCLID
UPLOAD_BATCH_SIZE = 256

GREEN_POLL_S = 2        # seconds between collection status checks
GREEN_TIMEOUT_S = 3600  # give up waiting for status green after this long
RESTART_DELAY_S = 5     # pause after all collections are green, before restarting Qdrant


# --- QUANTIZATION CONFIGS ----------------------------------------------------
# pinned=True keeps the quantized vectors in RAM; the original vectors are then stored cold (on disk)

def _memory(pinned):
    return models.Memory.PINNED if pinned else None


def product(ratio, pinned=True):
    return models.ProductQuantization(product=models.ProductQuantizationConfig(
        compression=ratio, memory=_memory(pinned)))


def scalar_int8(pinned=True):
    return models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8, quantile=0.99, memory=_memory(pinned)))


def binary(encoding, pinned=True):
    return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
        encoding=encoding, memory=_memory(pinned)))


def turbo(bits, pinned=True):
    return models.TurboQuantization(turbo=models.TurboQuantQuantizationConfig(
        bits=bits, memory=_memory(pinned)))


Ratio = models.CompressionRatio
Encoding = models.BinaryQuantizationEncoding
Bits = models.TurboQuantBitSize

QUANTIZATION_JOBS = {
    "baseline":   {"qvectors": None, "dvectors": models.Memory.COLD},

    "4x_pq":      {"qvectors": product(Ratio.X4), "dvectors": models.Memory.COLD},
    "8x_pq":      {"qvectors": product(Ratio.X8), "dvectors": models.Memory.COLD},
    "16x_pq":     {"qvectors": product(Ratio.X16), "dvectors": models.Memory.COLD},
    "32x_pq":     {"qvectors": product(Ratio.X32), "dvectors": models.Memory.COLD},
    "64x_pq":     {"qvectors": product(Ratio.X64), "dvectors": models.Memory.COLD},

    "4x_scalar":  {"qvectors": scalar_int8(), "dvectors": models.Memory.COLD},

    "32x_binary": {"qvectors": binary(Encoding.ONE_BIT), "dvectors": models.Memory.COLD},
    "21x_binary": {"qvectors": binary(Encoding.ONE_AND_HALF_BITS), "dvectors": models.Memory.COLD},
    "16x_binary": {"qvectors": binary(Encoding.TWO_BITS), "dvectors": models.Memory.COLD},

    "4x_tq":      {"qvectors": turbo(Bits.BITS8), "dvectors": models.Memory.COLD},
    "8x_tq":      {"qvectors": turbo(Bits.BITS4), "dvectors": models.Memory.COLD},
    "16x_tq":     {"qvectors": turbo(Bits.BITS2), "dvectors": models.Memory.COLD},
    "21x_tq":     {"qvectors": turbo(Bits.BITS1_5), "dvectors": models.Memory.COLD},
    "32x_tq":     {"qvectors": turbo(Bits.BITS1), "dvectors": models.Memory.COLD},
}


class StepError(Exception):
    """Any failure; carries the reason to print."""


def check_existing_config(client, name, dvectors, qvectors):
    """Raise StepError if an existing collection doesn't match the config we would create."""
    existing_collection = client.get_collection(name).config

    vectors_size = existing_collection.params.vectors.size
    vectors_distance = existing_collection.params.vectors.distance

    if vectors_size != DIM or vectors_distance != DISTANCE:
        raise StepError(f"'{name}' exists with a different vector configuration.")

    curr_dvectors_memory = existing_collection.params.vectors.memory
    if curr_dvectors_memory != dvectors:
        raise StepError(f"'{name}' exists with vector memory '{curr_dvectors_memory}', expected "
                        f"'{dvectors}'. Delete it yourself or use another name.")

    curr_qvectors_memory = existing_collection.quantization_config
    if curr_qvectors_memory != qvectors:
        raise StepError(f"'{name}' exists with quantization '{curr_qvectors_memory}', expected "
                        f"'{qvectors}'. Delete it yourself or use another name.")


def create_collection_if_not_exists(client, name, quant_config):
    """Create the collection, or check an existing one matches. Returns 'created' or 'exists'."""
    dvectors = quant_config.get("dvectors", models.Memory.COLD)
    qvectors = quant_config.get("qvectors", None)

    if not client.collection_exists(name):
        try:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=DIM, distance=DISTANCE,
                                                   memory=dvectors),
                quantization_config=qvectors,
            )
            return "created"
        except Exception as exc:
            # Fine if it was created in the meantime; it gets checked below
            if not client.collection_exists(name):
                raise StepError(f"Could not create '{name}': {exc}") from exc

    check_existing_config(client, name, dvectors, qvectors)
    return "exists"


# --- LOAD --------------------------------------------------------------------

def stream_fvecs(filename):
    """Yield vectors from an .fvecs file one at a time ([dim:int32][dim x float32] records)."""
    with open(filename, "rb") as f:
        while dim_bytes := f.read(4):
            (dim,) = struct.unpack("<i", dim_bytes)
            yield list(struct.unpack(f"<{dim}f", f.read(dim * 4)))


def load_collection(client, name, progress_bar_row=0):
    """Upload SIFT1M with ids = row index, so re-running overwrites instead of duplicating."""
    existing = client.count(name, exact=True).count
    if existing == EXPECTED_COUNT:
        return "skipped (already loaded)"
    if existing > EXPECTED_COUNT:
        raise StepError(f"'{name}' holds {existing} points (> {EXPECTED_COUNT}).")

    vectors = tqdm(stream_fvecs(FVECS_FILE_PATH), total=EXPECTED_COUNT,
                   desc=f"Uploading {name}", position=progress_bar_row)
    # parallel=1: each collection already uploads in its own process, and worker
    # processes can't start upload_collection's own process pool
    client.upload_collection(collection_name=name, vectors=vectors, ids=count(),
                             batch_size=UPLOAD_BATCH_SIZE, parallel=1, wait=True)

    final = client.count(name, exact=True).count
    if final != EXPECTED_COUNT:
        raise StepError(f"'{name}' holds {final} points after upload, expected {EXPECTED_COUNT}.")
    return "uploaded"


def prepare_collection(name, quantization, progress_bar_row):
    """Create and load one collection. Runs in its own process, so it opens its own client."""
    client = QdrantClient(url=QDRANT_URL)
    created = create_collection_if_not_exists(client, name, quantization)
    loaded = load_collection(client, name, progress_bar_row)
    return f"{created}, {loaded}"


def prepare_all_in_parallel(jobs):
    """Create + load every collection at the same time, one process each.

    Returns {name: error message} for the collections that failed.
    """
    failed = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(prepare_collection, name, quant_config, row): name
                   for row, (name, quant_config) in enumerate(jobs.items())}
        for future in as_completed(futures):
            name = futures[future]
            try:
                print(f"Collection {name}: {future.result()}")
            except Exception as exc:
                failed[name] = str(exc)
                print(f"Collection {name}: FAILED: {exc}")
    return failed


def print_status_line(text):
    """Overwrite the live 'Waiting ...' line with a permanent one."""
    print(f"\r{' ' * 120}\r{text}")


def wait_until_all_green(client, names):
    """Poll every collection until Qdrant reports status green (indexing/optimization finished).

    Returns {name: error message} for collections that turned red or timed out.
    """
    start = time.monotonic()
    pending = list(names)
    failed = {}

    while pending:
        elapsed = time.monotonic() - start
        still_waiting = []
        for name in pending:
            info = client.get_collection(name)
            status = info.status
            if status == models.CollectionStatus.GREEN:
                print_status_line(f"Status {name}: green after {elapsed:.0f}s")
            elif status == models.CollectionStatus.RED:
                print_status_line(f"Status {name}: red")
                failed[name] = "status red"
            elif elapsed > GREEN_TIMEOUT_S:
                failed[name] = f"still {status.value} after {GREEN_TIMEOUT_S}s"
            else:
                still_waiting.append((name, status, info))

        pending = [name for name, _, _ in still_waiting]
        if pending:
            progress = " | ".join(
                f"{name} {status.value} "
                f"{info.indexed_vectors_count or 0:,}/{info.points_count or 0:,}"
                for name, status, info in still_waiting)
            print(f"\rWaiting ({elapsed:.0f}s): {progress}", end="", flush=True)
            time.sleep(GREEN_POLL_S)

    return failed


def run(cmd):
    """Run a command, echoing it first; raise StepError if it fails."""
    print(f"$ {shlex.join(cmd)}", flush=True)  # flush so it shows before the command's output
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StepError(f"'{shlex.join(cmd)}' failed: {exc}") from exc


def restart_and_report(names):
    """Restart Qdrant, drop the OS page cache, then print each collection's memory table."""
    print(f"\nWaiting {RESTART_DELAY_S}s before restarting Qdrant...", flush=True)
    time.sleep(RESTART_DELAY_S)
    try:
        print("\n=== Restarting Qdrant ===")
        restart_qdrant()
        print("\n=== Dropping page cache ===")
        drop_caches()
    except RestartError as exc:
        raise StepError(str(exc)) from exc

    for name in names:
        print(f"\n=== qmemtable --c {name} ===")
        run(["bash", "-c", 'source "$1" && qmemtable --c "$2"', "bash", ALIASES_PATH, name])


def parse_jobs():
    """Return {name: quant_config} for the collections chosen on the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every collection in QUANTIZATION_JOBS")
    group.add_argument("--c", nargs="+", choices=QUANTIZATION_JOBS.keys(), metavar="COLLECTION",
                       help="one or more collections, created and loaded in parallel")
    args = parser.parse_args()

    if args.all:
        return dict(QUANTIZATION_JOBS)
    # dict.fromkeys de-duplicates while keeping the order given
    return {name: QUANTIZATION_JOBS[name] for name in dict.fromkeys(args.c)}


def main():
    jobs = parse_jobs()

    if not os.path.exists(FVECS_FILE_PATH):
        raise StepError(f"Could not find SIFT1M file at '{FVECS_FILE_PATH}'.")
    client = QdrantClient(url=QDRANT_URL)
    client.get_collections()  # fail fast if Qdrant isn't running

    # 1. Create + load all collections in parallel
    failed = prepare_all_in_parallel(jobs)

    # 2. Wait until every loaded collection is green
    loaded = [name for name in jobs if name not in failed]
    if loaded:
        print()
        failed.update(wait_until_all_green(client, loaded))

    if failed:
        print("\nFAILED:")
        for name, reason in failed.items():
            print(f"  {name}: {reason}")
        sys.exit(1)
    print(f"\nSUCCESS: all {len(jobs)} collections are green")

    # 3. Restart Qdrant + drop caches so the memory tables show a cold start
    restart_and_report(list(jobs))


if __name__ == "__main__":
    try:
        main()
    except StepError as exc:
        sys.exit(f"ERROR: {exc}")
