#!/bin/bash

# Check if minimum required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name> <expid> [additional_args...]"
    echo "Example: $0 graplus test_run --batch_size 64 --d_k 64"
    exit 1
fi

# Store model name and experiment ID
MODEL_NAME="$1"
EXPID="$2"
shift 2  # Remove first two arguments (model_name and expid) from $@

# Get the absolute path to the main project directory
PROJECT_ROOT="$(pwd)"

# Get absolute path to dataset
DATASET_PATH="$PROJECT_ROOT/dataset/OPA"

# Check if model directory exists
if [ ! -d "models/$MODEL_NAME" ]; then
    echo "Error: Model directory 'models/$MODEL_NAME' does not exist"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "dataset/OPA" ]; then
    echo "Error: Dataset directory 'dataset/OPA' does not exist"
    exit 1
fi

# Change to the specific model directory
cd "models/$MODEL_NAME" || {
    echo "Error: Could not change to models/$MODEL_NAME directory"
    exit 1
}

# Run the training script with all provided arguments
python main.py \
    --expid "$EXPID" \
    --data_root "$DATASET_PATH" \
    "$@"

# Return to original directory
cd "$PROJECT_ROOT"