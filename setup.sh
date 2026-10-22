#!/usr/bin/env bash
set -euo pipefail
 
echo "=== TRY: Installing Docker ===" 
# 1. Install Docker if missing
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable --now docker
fi
echo "=== SUCCESS: Installing Docker ==="
 
echo "=== TRY: Running Qdrant service ==="
# 2. Run Qdrant (re-running replaces the container; data persists in ./qdrant_storage)
sudo docker rm -f qdrant >/dev/null 2>&1 || true
sudo docker run -d --name qdrant --restart unless-stopped \
    -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
echo "=== SUCCESS: Running Qdrant service ==="

echo "=== TRY: Installing Python deps ==="
# 3. Python dependencies
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q qdrant-client tqdm
echo "=== SUCCESS: Installing Python deps ==="

echo "=== TRY: Virtual environment activated ==="
source .venv/bin/activate
echo "=== SUCCESS: Virtual environment activated ==="
 
# 4. Wait for Qdrant to be ready (fails after ~30s)
curl -fsS --retry 30 --retry-delay 1 --retry-connrefused http://localhost:6333/readyz >/dev/null
 
echo "Success! Qdrant: REST localhost:6333 | UI localhost:6333/dashboard | gRPC localhost:6334"