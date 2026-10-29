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
python main.py
```

# Run Query Script
```
python query.py [flags]
```


####  Flags
```
--collection string           Optional. Supply a Qdrant collection name. Defaults to 'sift1m_baseline'
```