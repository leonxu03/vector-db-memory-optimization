"""
Run query.py against one or more collections and record memory, storage and block I/O over time.

Usage:
    python monitor_query.py                             # defaults to "baseline"
    python monitor_query.py --c <collection_name> [--interval 0.5] [--out-dir results]
    python monitor_query.py --c 4x_pq 4x_scalar 4x_tq [--cooldown 30] [--batch-size 100]
    python monitor_query.py --c <collection_name> --flush-cache false   # keep Qdrant + page cache warm

For each collection, in order:
    1. Cool down for --cooldown seconds (skipped for the first collection).
    2. Cold start (--flush-cache true, the default): restart Qdrant, check the collection, drop
       the OS page cache, so every run starts from the same flat baseline.
       (query.py itself runs with --no-restart, so nothing restarts mid-recording.)
    3. Sample for IDLE_SECONDS, run query.py while sampling, then sample IDLE_SECONDS more.
    4. Save the samples and graphs.

Outputs, one folder per collection (in <out-dir>/<YYYYMMDD>/<collection>[_b<batch>]_<timestamp>/):
    memory.png    pinned + page cache (MB) stacked per structure from GET /collections/<c>/memory
    storage.png   Disk used (MB) stacked per structure         from GET /collections/<c>/memory
                  Structures: id_tracker, payload, vectors.index, vectors.quantized, vectors.dense, ...
    block_io.png  Disk read + write op counts stacked (= total ops)  from the container's cgroup io.stat
    samples.csv   Raw samples

The x-axis of every graph is query completion (0-100% of queries answered, as reported by
query.py --progress-file), so runs of different collections line up and can be overlaid.
The idle sampling before and after the run is drawn as fixed-width sections
(-IDLE_PCT..0 and 100..100+IDLE_PCT, placed by time), so every run has the same x-axis.
samples.csv has phase (before / query / after), progress_pct and the plotted x (plot_x).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qdrant_restart import (
    QDRANT_CONTAINER, QDRANT_URL, RestartError, drop_caches, restart_qdrant)

DEFAULT_COLLECTION = "baseline"
DEFAULT_INTERVAL = 0.5   # seconds between samples
DEFAULT_OUT_DIR = "results"
DEFAULT_COOLDOWN = 30    # seconds of rest between collections so Qdrant settles
BATCH_SIZES = [1, 10, 100, 1000, 10000] 

STORAGE_DIR = "qdrant_storage"
CGROUP_ROOT = "/sys/fs/cgroup/system.slice"

IDLE_SECONDS = 2.0  # sampling before and after query.py runs
IDLE_PCT = 10       # x-axis width of each idle section, in % units

# samples.csv columns; per-structure columns (see column()) follow these
FIELDS = ["t", "phase", "progress_pct", "plot_x", "pinned_mb", "cached_mb", "total_mb",
          "disk_mb", "read_ops", "write_ops", "total_ops"]

# measure name -> field in Qdrant's /memory response ("pinned" = held in RAM, from ram_bytes)
MEASURES = {"disk": "disk_bytes", "pinned": "ram_bytes", "cached": "cached_bytes"}


class StepError(Exception):
    """Any failure; carries the reason to print."""


def fetch_memory(collection):
    """Return the 'result' object of Qdrant's /collections/<c>/memory endpoint."""
    url = f"{QDRANT_URL}/collections/{collection}/memory"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.load(resp)["result"]


def components_of(result):
    """Turn Qdrant's nested /memory response into a flat {structure name: MB per measure}.

    Qdrant's response (bytes)                     What this returns (MB)
    -------------------------                     ----------------------
    "vectors": [{"name": "",
                 "storage":   {"disk_bytes": ...}  ->  "vectors.dense":     {"disk", "pinned", "cached"}
                 "index":     {"disk_bytes": ...}  ->  "vectors.index":     {...}
                 "quantized": {"disk_bytes": ...}  ->  "vectors.quantized": {...}
               }],
    "payload":              {"disk_bytes": ...}    ->  "payload":           {...}
    "other": {"id_tracker": {"disk_bytes": ...}}   ->  "id_tracker":        {...}
    "total":                {"disk_bytes": ...}        (skipped: the graphs add up their own total)

    It isn't hardcoded to these names: anything in the response holding "disk_bytes" is
    picked up, so named vectors, sparse vectors or payload indexes would get their own entries.
    """
    components = {}
    for section, value in result.items():
        if section == "total":
            continue
        for path, numbers in find_structures(value, child_path([], section)):
            name = ".".join(path)
            entry = components.setdefault(name, dict.fromkeys(MEASURES, 0.0))
            for measure, field in MEASURES.items():  # e.g. "pinned" <- "ram_bytes"
                entry[measure] += numbers.get(field, 0) / 1e6
    return components


def find_structures(node, path):
    """Yield (path, numbers) for every structure inside node.

    A structure is a dict holding "disk_bytes"; path is the list of names leading to it,
    e.g. (["vectors", "index"], {"disk_bytes": ..., "ram_bytes": ..., "cached_bytes": ...}).
    """
    if isinstance(node, dict) and "disk_bytes" in node:
        # Found one: this dict holds the numbers
        yield path, node

    elif isinstance(node, dict):
        # A group of structures: look inside each one
        for key, child in node.items():
            yield from find_structures(child, child_path(path, key))

    elif isinstance(node, list):
        # A list, e.g. one entry per vector: add the entry's name (if any) to the path
        for item in node:
            name = entry_name(item)
            yield from find_structures(item, path + [name] if name else path)


def child_path(path, key):
    """The path to child `key`, with two renames to keep the names short and clear."""
    if key == "other":
        return path  # just a grouping: "other.id_tracker" -> "id_tracker"
    if key == "storage" and path[:1] == ["vectors"]:
        return path + ["dense"]  # the original vectors: "vectors.storage" -> "vectors.dense"
    return path + [key]


def entry_name(item):
    """A list entry's name: its first non-empty text field, e.g. a named vector's "name".

    The default vector has name "", so it adds nothing: "vectors.index", not "vectors..index".
    """
    return next((value for value in item.values() if isinstance(value, str) and value), None)


def column(measure, component):
    """samples.csv column for one measure of one structure, e.g. 'pinned_vectors.index_mb'."""
    return f"{measure}_{component}_mb"


def components_in(samples):
    """Every structure seen in any sample, in stacking order (bottom first)."""
    seen = dict.fromkeys(key[len("disk_"):-len("_mb")] for sample in samples for key in sample
                         if key.startswith("disk_") and key != "disk_mb")
    return stack_order(seen)


def stack_order(components):
    """Static structures at the bottom, quantized vectors next, dense vectors on top."""
    def rank(component):
        if component == "id_tracker":
            return 0
        if component.startswith("payload"):
            return 1
        if component.endswith(".index"):
            return 2
        if component.endswith(".quantized"):
            return 4
        if component.endswith(".dense"):
            return 5
        return 3  # anything else sits between the index and the quantized vectors
    return sorted(components, key=rank)


# --- DISK I/O ----------------------------------------------------------------

def find_io_stat(container):
    """Locate the container's cgroup io.stat file (Docker is only needed to get the ID)."""
    try:
        container_id = subprocess.run(
            ["sudo", "-n", "docker", "inspect", "-f", "{{.Id}}", container],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise StepError(f"Could not inspect container '{container}'. Needs passwordless sudo "
                        f"for docker, or add your user to the docker group. ({exc})") from exc
    path = f"{CGROUP_ROOT}/docker-{container_id}.scope/io.stat"
    if not os.path.exists(path):
        raise StepError(f"cgroup file not found: {path}")
    return path


def storage_device():
    """'major:minor' of the disk holding qdrant_storage, so io.stat is not double-counted."""
    dev = os.stat(STORAGE_DIR).st_dev
    return f"{os.major(dev)}:{os.minor(dev)}"


def read_io_ops(path, dev):
    """Return cumulative (read ops, write ops) for one device from io.stat."""
    with open(path) as f:
        for line in f:
            parts = line.split()  # e.g. "259:0 rbytes=... wbytes=... rios=12 wios=3 ..."
            if parts and parts[0] == dev:
                stats = dict(p.split("=") for p in parts[1:])
                return int(stats["rios"]), int(stats["wios"])
    return 0, 0  # device has no I/O yet


# --- SAMPLING ----------------------------------------------------------------

def read_progress(path):
    """Percent of queries completed according to query.py's progress file (0 until it exists)."""
    try:
        with open(path) as f:
            done, total = f.read().split("/")
        return int(done) / int(total) * 100
    except (OSError, ValueError, ZeroDivisionError):
        return 0.0


def take_sample(collection, io_path, dev, t0, phase, progress_pct):
    """One row of samples.csv: totals, cumulative I/O ops, and MB per structure."""
    memory = fetch_memory(collection)
    total = memory["total"]
    pinned_mb, cached_mb = total["ram_bytes"] / 1e6, total["cached_bytes"] / 1e6
    read_ops, write_ops = read_io_ops(io_path, dev)

    row = {
        "t": time.perf_counter() - t0,
        "phase": phase,
        "progress_pct": progress_pct,
        "pinned_mb": pinned_mb,
        "cached_mb": cached_mb,
        "total_mb": pinned_mb + cached_mb,
        "disk_mb": total["disk_bytes"] / 1e6,
        "read_ops": read_ops,
        "write_ops": write_ops,
    }
    for component, measures in components_of(memory).items():
        for measure, mb in measures.items():
            row[column(measure, component)] = mb
    return row


def cold_start(collection):
    """Restart Qdrant, check the collection is readable again, then drop the page cache.

    Returns the container's io.stat path, looked up again in case the restart changed it.
    """
    print(f"Cold start before monitoring {collection}", flush=True)
    try:
        restart_qdrant()
    except RestartError as exc:
        raise StepError(str(exc)) from exc

    try:
        fetch_memory(collection)
    except Exception as exc:
        raise StepError(f"Cannot read memory for '{collection}' after restart: {exc}") from exc
    io_path = find_io_stat(QDRANT_CONTAINER)

    try:
        drop_caches()
    except RestartError as exc:
        raise StepError(str(exc)) from exc
    return io_path

def run_and_sample(collection, interval, io_path, dev, batch_size=1):
    """Sample before, during and after query.py; return (samples, query.py's exit code)."""
    samples = []
    t0 = time.perf_counter()
    progress_dir = tempfile.mkdtemp(prefix="monitor_query_")
    progress_file = os.path.join(progress_dir, "progress")

    def sample_idle(seconds, phase, progress_pct):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            samples.append(take_sample(collection, io_path, dev, t0, phase, progress_pct))
            time.sleep(interval)

    # 1. Idle before
    sample_idle(IDLE_SECONDS, "before", 0.0)

    # 2. query.py running
    t_start = time.perf_counter() - t0
    query_cmd = [sys.executable, "query.py", "--c", collection,
                 "--batch-size", str(batch_size), "--progress-file", progress_file,
                 "--no-restart"]  # cold start was already done above
    proc = subprocess.Popen(query_cmd)
    try:
        while proc.poll() is None:
            samples.append(take_sample(collection, io_path, dev, t0, "query",
                                       read_progress(progress_file)))
            time.sleep(interval)
    finally:
        if proc.poll() is None:  # we were interrupted; don't leave query.py running
            proc.terminate()
            proc.wait()
        final_pct = read_progress(progress_file)
        shutil.rmtree(progress_dir, ignore_errors=True)
    t_end = time.perf_counter() - t0

    # 3. Idle after
    sample_idle(IDLE_SECONDS, "after", final_pct)

    # I/O counters are cumulative since container start; make them relative to the first sample
    reads_0, writes_0 = samples[0]["read_ops"], samples[0]["write_ops"]
    for s in samples:
        s["read_ops"] -= reads_0
        s["write_ops"] -= writes_0
        s["total_ops"] = s["read_ops"] + s["write_ops"]
        s["plot_x"] = plot_x(s, t_start, t_end)
    return samples, proc.returncode


def plot_x(sample, t_start, t_end):
    """x position of a sample: query completion % while query.py runs.

    Idle time maps onto the fixed-width sections before 0% and after 100%
    (IDLE_SECONDS of idle = IDLE_PCT of axis).
    """
    if sample["phase"] == "before":
        return -IDLE_PCT * min(1.0, (t_start - sample["t"]) / IDLE_SECONDS)
    if sample["phase"] == "after":
        return 100 + IDLE_PCT * min(1.0, (sample["t"] - t_end) / IDLE_SECONDS)
    return sample["progress_pct"]

#
# --- GRAPHS ------------------------------------------------------------------
#
# Style of each part of a group: part 0 (e.g. pinned) solid, part 1 (e.g. page cache) hatched
PART_STYLES = [{"alpha": 0.85, "hatch": None}, {"alpha": 0.35, "hatch": "///"}]
LABEL_MIN = 0.05  # values below this would print as 0.0, so they get no label


def _draw_bands(ax, samples, x, groups):
    """Fill one band per (group, part), stacked bottom-up.

    Returns [(label, color, bottom, [value per part])] at the last sample, for the labels.
    """
    bottom = [0.0] * len(x)
    extents = []
    for label, color, columns in groups:
        group_bottom = bottom[-1]
        ends = []
        for style, col in zip(PART_STYLES, columns):
            values = [s.get(col, 0) for s in samples]
            upper = [b + v for b, v in zip(bottom, values)]
            solid = style["hatch"] is None
            ax.fill_between(x, bottom, upper, color=color, alpha=style["alpha"],
                            hatch=style["hatch"], edgecolor="white" if solid else color,
                            linewidth=0.6 if solid else 0, zorder=1)
            ends.append(values[-1])
            bottom = upper
        extents.append((label, color, group_bottom, ends))
    return extents


def _draw_idle_sections(ax, top):
    """Lightly grey the idle sections before/after the queries and dash the run's edges."""
    for lo, hi, text in ((-IDLE_PCT, 0, "idle\nbefore"), (100, 100 + IDLE_PCT, "idle\nafter")):
        ax.axvspan(lo, hi, color="white", alpha=0.45, zorder=2)
        ax.annotate(text, ((lo + hi) / 2, top), xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="0.3")
    for x in (0, 100):
        ax.axvline(x, color="black", linestyle="--", linewidth=1, zorder=3)


def _draw_labels(ax, extents, parts, end_total, top, unit, fmt):
    """Label each group at the right edge (size + share of total), nudged apart to not overlap."""
    labels = []  # [y, text, color, weight], bottom first
    for label, color, group_bottom, ends in extents:
        value = sum(ends)
        if value < LABEL_MIN:
            continue
        share = value / end_total * 100 if end_total else 0
        text = f"{label}  {value:{fmt}} {unit} ({share:.0f}%)"
        if len(parts) > 1:
            text += "  = " + " + ".join(f"{p} {e:{fmt}}" for p, e in zip(parts, ends)
                                        if e >= LABEL_MIN)
        labels.append([group_bottom + value / 2, text, color, "normal"])

    total_text = f"Total  {end_total:{fmt}} {unit}"
    if len(parts) > 1:
        part_totals = [sum(ends[i] for *_, ends in extents) for i in range(len(parts))]
        total_text += "  = " + " + ".join(f"{p} {v:{fmt}}" for p, v in zip(parts, part_totals))
    labels.append([end_total, total_text, "black", "bold"])

    # Push each label up until it's at least `gap` above the one below it
    gap = top * 0.05
    for below, label in zip(labels, labels[1:]):
        label[0] = max(label[0], below[0] + gap)

    for y, text, color, weight in labels:
        ax.annotate(text, (100 + IDLE_PCT, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=color, fontweight=weight,
                    annotation_clip=False)


def _draw_parts_legend(fig, parts):
    """Key for solid vs hatched, in the empty top-right corner of the figure."""
    handles = [Patch(facecolor="0.4", alpha=style["alpha"], hatch=style["hatch"],
                     edgecolor="0.4" if style["hatch"] else "white", label=part)
               for style, part in zip(PART_STYLES, parts)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=len(parts), fontsize=9, frameon=False)


def _format_axes(ax, title, ylabel, top):
    ax.set_title(title, loc="left")
    ax.set_xlabel(f"Query completion (%)  —  with {IDLE_SECONDS:g} s idle before and after")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-IDLE_PCT, 100 + IDLE_PCT)
    percent_ticks = list(range(0, 101, 20))
    ax.set_xticks([-IDLE_PCT] + percent_ticks + [100 + IDLE_PCT],
                  [f"-{IDLE_SECONDS:g} s"] + [str(x) for x in percent_ticks]
                  + [f"+{IDLE_SECONDS:g} s"])
    ax.set_ylim(0, top)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.3)


def plot_stacked(samples, total_column, groups, parts, title, ylabel, unit, path, fmt=",.1f"):
    """Save a stacked-area graph whose top edge is the total, with each group labelled.

    groups: [(label, color, [column per part])], bottom of the stack first.
    parts:  part names, e.g. ["pinned", "page cache"]; part 0 is drawn solid, part 1 hatched.
    """
    x = [s["plot_x"] for s in samples]
    totals = [s[total_column] for s in samples]
    top = max(totals) * 1.1 if max(totals) > 0 else 1
    fig, ax = plt.subplots(figsize=(11, 5.5))

    extents = _draw_bands(ax, samples, x, groups)
    ax.plot(x, totals, color="black", linewidth=1.8, zorder=3)
    _draw_idle_sections(ax, top)
    _draw_labels(ax, extents, parts, totals[-1], top, unit, fmt)
    if len(parts) > 1:
        _draw_parts_legend(fig, parts)
    _format_axes(ax, title, ylabel, top)

    fig.tight_layout()
    fig.subplots_adjust(right=0.62 if len(parts) > 1 else 0.72)  # room for the group labels
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- OUTPUT ------------------------------------------------------------------

def save_csv(samples, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
        writer.writeheader()
        writer.writerows(samples)


def output_dir(out_dir, collection, batch_size):
    """<out_dir>/<YYYYMMDD>/<collection>[_b<batch>]_<YYYYMMDD-HHMMSS>/"""
    now = datetime.now()  # one timestamp so the date folder and run folder always agree
    batch_tag = f"_b{batch_size}" if batch_size > 1 else ""
    return os.path.join(out_dir, f"{now:%Y%m%d}", f"{collection}{batch_tag}_{now:%Y%m%d-%H%M%S}")


def save_results(samples, collection, dev, out_dir):
    """Write samples.csv, memory.png, storage.png and block_io.png into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    components = components_in(samples)
    save_csv(samples, os.path.join(out_dir, "samples.csv"),
             FIELDS + [column(m, c) for c in components for m in MEASURES])

    colors = {c: plt.cm.tab10(i % 10) for i, c in enumerate(components)}  # same on both graphs
    plot_stacked(samples, "total_mb",
                 [(c, colors[c], [column("pinned", c), column("cached", c)]) for c in components],
                 ["pinned", "page cache"],
                 f"{collection} — Memory by structure (stacked = Total)",
                 "Memory (MB)", "MB", os.path.join(out_dir, "memory.png"))
    plot_stacked(samples, "disk_mb",
                 [(c, colors[c], [column("disk", c)]) for c in components],
                 ["disk"],
                 f"{collection} — Storage by structure (stacked = Total)",
                 "Storage (MB)", "MB", os.path.join(out_dir, "storage.png"))
    plot_stacked(samples, "total_ops",
                 [("Reads", "tab:blue", ["read_ops"]), ("Writes", "tab:orange", ["write_ops"])],
                 ["ops"],
                 f"{collection} — Block I/O, device {dev} (stacked = Total = Reads + Writes)",
                 "Operations (cumulative count)", "ops", os.path.join(out_dir, "block_io.png"),
                 fmt=",.0f")


def monitor_collection(collection, args, io_path, dev):
    """Monitor one query.py run and save its CSV and graphs; return query.py's exit code."""
    samples, returncode = run_and_sample(collection, args.interval, io_path, dev, args.batch_size)
    out_dir = output_dir(args.out_dir, collection, args.batch_size)
    save_results(samples, collection, dev, out_dir)

    print(f"\nSamples:      {len(samples)} every {args.interval}s")
    print(f"Saved to:     {out_dir}/")
    return returncode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c", nargs="+", default=[DEFAULT_COLLECTION], metavar="COLLECTION",
                        help="One or more Qdrant collections to test, run in the order given "
                             f"(default: {DEFAULT_COLLECTION})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"Seconds to wait between collections (default: {DEFAULT_COOLDOWN})")
    parser.add_argument("--batch-size", type=int, default=1, choices=BATCH_SIZES,
                        help="Queries per request, passed to query.py (default: 1, sequential)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Sampling interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Parent directory for results (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--flush-cache", choices=["true", "false"], default="true",
                        help="Restart Qdrant and drop the OS page cache before each collection "
                             "(default: true)")
    return parser.parse_args()


def main():
    args = parse_args()
    collections = list(dict.fromkeys(args.c))  # de-duplicate, keep order

    io_path = find_io_stat(QDRANT_CONTAINER)
    dev = storage_device()
    for collection in collections:  # fail fast, before spending time on earlier runs
        try:
            fetch_memory(collection)
        except Exception as exc:
            raise StepError(f"Cannot read memory for collection '{collection}': {exc}") from exc

    failed = []
    for i, collection in enumerate(collections):
        if i > 0 and args.cooldown > 0:
            print(f"\nCooling down {args.cooldown:g}s before monitoring {collection}...", flush=True)
            time.sleep(args.cooldown)
        print(f"\n=== [{i + 1}/{len(collections)}] {collection} ===", flush=True)
        if args.flush_cache == "true":
            io_path = cold_start(collection)
        if monitor_collection(collection, args, io_path, dev) != 0:
            failed.append(collection)

    if failed:
        sys.exit(f"ERROR: query.py failed for: {', '.join(failed)}")


if __name__ == "__main__":
    try:
        main()
    except StepError as exc:
        sys.exit(f"ERROR: {exc}")
