#!/bin/bash

# Check if minimum required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name> <config_args...>"
    echo "Example: $0 graplus --epoch 21 --eval_type eval"
    exit 1
fi

# Store experiment name
EXPID="$1"
shift  # Remove first argument (model_name) from $@

# Get the absolute path to the main project directory
PROJECT_ROOT="$(pwd)"

# Get absolute path to dataset
DATASET_PATH="$PROJECT_ROOT/dataset/OPA"


# Check if model directory exists
if [ ! -d "models/$EXPID" ]; then
    echo "Error: Model directory 'models/$EXPID' does not exist"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "dataset/OPA" ]; then
    echo "Error: Dataset directory 'dataset/OPA' does not exist"
    exit 1
fi


# Change to the specific model directory
cd "models/$EXPID" || {
    echo "Error: Could not change to models/$EXPID directory"
    exit 1
}

# Run the inference script with all provided arguments
# Use absolute path for data_root
python infer.py \
    --expid "$EXPID" \
    --data_root "$DATASET_PATH" \
    "$@"

# Return to original directory
cd "$ORIGINAL_DIR"