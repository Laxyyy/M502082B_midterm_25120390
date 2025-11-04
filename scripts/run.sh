#!/bin/bash
# This script runs the main training pipeline.

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory of this script
DIR_PATH=$(dirname "$0")
# Get the absolute path of the project root (one level up from scripts)
PROJECT_ROOT=$(realpath "$DIR_PATH/..")

echo "Project Root: $PROJECT_ROOT"
echo "Starting training..."

# Run the training script from the project root
python3 "$PROJECT_ROOT/run_train.py"

echo "Training finished."
