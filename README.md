# vector-db-memory-optimization
Researching compression algorithms on Qdrant and Chimera vector databases

# Setup
Run setup script

```
source ./setup.sh
```

# Dataset Setup
Due to GitHub's restrictions on allowed uploaded file sizes, please see the "Release" section of repo and downloaded datasets tagged v1.0.0. Put those 4 files into the `/dataset` folder

# Create and Populate Collections
Please comment/uncomment the interested quantization collections BEFORE running this script (currently set to only create baseline collection)

```Setup Collections (please check which ones you want to bring up)
python main.py [flags]
```

####  Flags
```
--c string           Supply a Qdrant QUANTIZATION_JOBS name (see main.py)
```

# Run Query Script
```
python query.py [flags]
```


####  Flags
```
--c string           Optional. Supply a Qdrant collection name. Defaults to 'sift1m_baseline'
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
| `qmemtable --c <name>` | Show memory usage for a collection (disk, RAM, and page cache per component) in table format | `GET /collections/<name>` |

### Examples

    qlist
    qinfo --c sift1m_baseline
    qmem --c pq_x4
    qcount --c pq_x8
    qdelete --c pq_x16
    qmemall