#!/bin/bash
# Run Cowrie training in Docker with MPS support for Apple Silicon

# Ensure script exits on error
set -e

# Check if running on macOS with Apple Silicon
if [[ "$(uname)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: This script is designed for macOS on Apple Silicon."
  echo "MPS acceleration will not be available on other platforms."
fi

# Build the Docker image
echo "Building Docker training image..."
docker build --target training -t logbert:training-mps .

# Create required directories
mkdir -p ./data/normalized
mkdir -p ./data/dataset
mkdir -p ./models/logbert-mlm

# Run the training container with proper volume mounts
echo "Starting training container with MPS support..."
docker run --rm -it \
  -v "$(pwd)/cowrie-honeypot:/home/app/cowrie-honeypot" \
  -v "$(pwd)/configs:/home/app/configs" \
  -v "$(pwd)/data:/home/app/data" \
  -v "$(pwd)/models:/home/app/models" \
  -v "$(pwd)/normalization:/home/app/normalization" \
  -v "$(pwd)/dataset:/home/app/dataset" \
  -v "$(pwd)/eval:/home/app/eval" \
  -v "$(pwd)/train_cowrie.py:/home/app/train_cowrie.py" \
  -v "$(pwd)/docker_train.py:/home/app/docker_train.py" \
  -e PYTHONUNBUFFERED=1 \
  logbert:training-mps \
  python docker_train.py --config /home/app/configs/config.yaml

echo "Training complete!"