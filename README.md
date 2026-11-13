# vector-db-memory-optimization
Researching compression algorithms on Qdrant and Chimera vector databases

# Setup
Run setup script

```
source ./setup.sh
```

`setup.sh` installs Docker, starts the Qdrant container, and installs these Python dependencies into `.venv`:

| Package | Used by |
|---|---|
| `qdrant-client` | `main.py`, `query.py` |
| `tqdm` | progress bars |
| `matplotlib` | `monitor_query.py` graphs |

To install them manually:

```
pip install qdrant-client tqdm matplotlib
```

# Dataset Setup
Due to GitHub's restrictions on allowed uploaded file sizes, please see the "Release" section of repo and download datasets tagged v1.0.0. Put those 4 files into the `/dataset` folder

# Create and Populate Collections
Creates and loads all requested collections in parallel (one process each), then waits until every one is green before exiting. Collections that already exist and are fully loaded are skipped. If any collection fails, the others still finish, and the script lists the failures and exits with code 1.

Once every collection is green, the script restarts the `qdrant` Docker container, waits for it to be ready, drops the OS page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`), and prints `qmemtable --c <name>` for each collection, so you can see memory use from a cold start. This step uses `sudo` and needs `jq`.

```
qcreate [flags]
```

####  Flags
```
--all                Create every collection in QUANTIZATION_JOBS (see main.py)
--c name [name ...]  Create one or more QUANTIZATION_JOBS collections, in parallel
```
One of `--all` or `--c` is required.

Collection names are `<compression>x_<method>`. Quantized vectors are pinned in RAM and the original vectors are stored cold on disk (the baseline's original vectors are cold too):

| Method | Names |
|---|---|
| None | `baseline` |
| Product | `4x_pq` `8x_pq` `16x_pq` `32x_pq` `64x_pq` |
| Scalar | `4x_scalar` |
| Binary | `32x_binary` `21x_binary` `16x_binary` |
| TurboQuant | `4x_tq` `8x_tq` `16x_tq` `21x_tq` `32x_tq` |

To add a variant, add a line to `QUANTIZATION_JOBS` in `main.py`, e.g. `"pq_x4": product(Ratio.X4, pinned=False)` for product quantization with Qdrant's default memory setting.

```
python main.py --c baseline 4x_pq 4x_scalar
qcreate --all
```

# Run Query Script
```
python query.py [flags]
```


####  Flags
```
--c name [name ...]  Optional. One or more Qdrant collection names, queried in the order given. Defaults to 'baseline'
--batch-size int     Optional. Queries per request: 1, 10, 100, 1000 or 10000. Defaults to 1 (sequential)
--cooldown float     Optional. Seconds to wait between collections. Defaults to 30
--progress-file path Optional. Write 'completed/total' queries to this file while running (used by monitor_query.py)
--save-csv bool      Optional. true or false: append results to results/query_results.csv. Defaults to true (cachegrind.py passes false)
--restart bool       Optional. true or false: restart Qdrant and drop the page cache before each collection. Defaults to true (monitor_query.py does its own cold start and passes false; cachegrind.py passes false too)
```

Before each collection, `query.py` restarts the `qdrant` Docker container and checks Qdrant  `/readyz`, then drops the OS page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`). Every collection is queried from a cold start. This uses `sudo`, and the script stops with an error if any step fails.

Each collection appends one row to `results/query_results.csv` (recall, per-query and per-request latency mean/median/P95/P99, total time, QPS). Per-query latency is the request time divided by the batch size.

```
python query.py --c 4x_pq 8x_pq 4x_scalar --batch-size 100
```

# Run Monitored Query
Runs `query.py` for each collection and samples resource usage before, during and after it, then saves a graph for each metric with its peak marked. Collections run one after another, with `--cooldown` seconds of rest in between.

```
python monitor_query.py [flags]
```

####  Flags
```
--c name [name ...]  Optional. One or more Qdrant collections, monitored in the order given. Defaults to baseline'
--batch-size int     Optional. Queries per request, passed to query.py: 1, 10, 100, 1000 or 10000. Defaults to 1
--cooldown float     Optional. Seconds to wait between collections. Defaults to 30
--interval float     Optional. Sampling interval in seconds. Defaults to 0.5
--out-dir string     Optional. Parent folder for results. Defaults to 'results'
--flush-cache bool   Optional. true or false: restart Qdrant and drop the page cache before each collection. Defaults to true
```

Before sampling each collection, the script restarts the `qdrant` container, checks it's running and ready, then drops the OS page cache. Every run starts from flushed cache baseline. `query.py` then runs with `--restart false`, so nothing restarts mid-recording.

Output goes to `results/<YYYYMMDD>/<collection>[_b<batch>]_<timestamp>/` (one folder per day; `_b<batch>` only when `--batch-size` is above 1):

| File | Metric | Source |
|---|---|---|
| `memory.png` | Pinned (held in RAM) and page cache (MB), stacked per structure | `GET /collections/<name>/memory` → `total.ram_bytes`, `total.cached_bytes` |
| `storage.png` | Disk used (MB), stacked per structure | `GET /collections/<name>/memory` → `total.disk_bytes` |
| `block_io.png` | Disk read and write operations (cumulative count) | Container cgroup `io.stat` → `rios`, `wios` |
| `samples.csv` | Raw samples | |

The x-axis is query completion (0–100% of queries answered, from `query.py --progress-file`), so runs of different collections line up. The 2 seconds of idle sampling before and after the run are drawn as fixed-width sections on either side.

Notes:
- The script runs `sudo -n docker inspect` once to find the `qdrant` container, so it needs passwordless sudo for Docker, or your user in the `docker` group.
- `docker stats` only reports Block I/O in bytes, so operation counts are read from the container's cgroup file `io.stat`, which is the same kernel data `docker stats` uses.
- Block I/O covers the whole Qdrant container, not just one collection. On a warm run (data already cached in RAM) it may stay flat at 0, meaning no disk I/O happened.

#### Example
```
python monitor_query.py --c 4x_pq
python monitor_query.py --c 4x_pq 4x_scalar 4x_tq --batch-size 100
```

# Qdrant local API commands
 Check your default shell 
```
echo $SHELL
```

If your default shell is:

-  Bash: add ```source ~/<path_to_this_project>/qdrant_aliases.sh``` to your ```~/.bashrc```
-  Zshrc: add ```source ~/<path_to_this_project>/qdrant_aliases.sh``` to your ```~/.zshrc```



| Command | Description | Qdrant endpoint |
|---|---|---|
| `qlist` | List all collections | `GET /collections` |
| `qver` | Show Qdrant version info | `GET /` |
| `qinfo --c <name>` | Show collection config, status, vector count, segments, and quantization settings | `GET /collections/<name>` |
| `qmem --c <name>` | Show memory usage for a collection (disk, RAM, and page cache per component) | `GET /collections/<name>/memory` |
| `qcount --c <name>` | Show exact point count | `POST /collections/<name>/points/count` |
| `qdelete --c <name>` | Delete a collection (asks for `y` to confirm) | `DELETE /collections/<name>` |
| `qmemall` | Show the total memory summary for every collection | `GET /collections` + `/memory` for each |
| `qcreate [flags]` | Same as `python main.py [flags]` (create and populate collections); runs from the project directory | — |
| `qmemtable --c <name>` | Show memory usage for a collection (disk, RAM, and page cache per component) in table format. Also accepts `qmem`-style JSON piped in | `GET /collections/<name>/memory` |

### Examples

    qlist
    qinfo --c baseline
    qmem --c 4x_pq
    qcount --c 8x_pq
    qdelete --c 16x_pq
    qmemall
    qmemtable --c 4x_pq
    qcreate --c baseline 4x_pq
