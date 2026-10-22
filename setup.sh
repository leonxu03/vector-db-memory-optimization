
#!/bin/bash
set -e
if command -v docker &> /dev/null; then
    echo "=== Docker is already installed (Version: $(docker --version)) ==="
else
    echo "=== Installing Docker... ==="
    curl -fsSL https://docker.com | sh
    sudo systemctl enable --now docker
    echo "=== Docker installation completed successfully! ==="
fi
