#!/bin/bash

# Check if minimum required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <eval_type> <model_name> <expid> <epoch> [additional_args...]"
    echo "Example: $0 eval graplus graplus_test_run 5"
    exit 1
fi

# Store evaluation type, model name, epoch, and experiment ID
EVAL_TYPE="$1"
MODEL_NAME="$2"
EXPID="$3"
EPOCH="$4"
shift 4  # Remove first four arguments from $@

# Get the absolute path to the main project directory
PROJECT_ROOT="$(pwd)"

# Get absolute path to dataset
OPA_PATH="$PROJECT_ROOT/dataset/OPA"
SG_PATH="$PROJECT_ROOT/dataset/OPA_SG"
GPT2_PATH="$PROJECT_ROOT/gpt2_embeddings"

# Check if model directory exists
if [ ! -d "models/$MODEL_NAME" ]; then
    echo "Error: Model directory 'models/$MODEL_NAME' does not exist"
    exit 1
fi

# Check if base dataset directory exists
if [ ! -d "dataset/OPA" ]; then
    echo "Error: Dataset directory 'dataset/OPA' does not exist"
    exit 1
fi

# Additional checks for GraPLUS-specific directories
if [ "$MODEL_NAME" = "graplus" ]; then
    # Check for scene graph directory
    if [ ! -d "dataset/OPA_SG" ]; then
        echo "Error: Scene graph directory 'dataset/OPA_SG' does not exist"
        echo "This directory is required for GraPLUS model"
        exit 1
    fi
    
    # Check for GPT-2 embeddings directory
    if [ ! -d "gpt2_embeddings" ]; then
        echo "Error: GPT-2 embeddings directory 'gpt2_embeddings' does not exist"
        echo "This directory is required for GraPLUS model"
        exit 1
    fi
fi

# Change to the specific model directory
cd "models/$MODEL_NAME" || {
    echo "Error: Could not change to models/$MODEL_NAME directory"
    exit 1
}

# Run the inference script with model-specific arguments
if [ "$MODEL_NAME" = "graplus" ]; then
    # Special case for graplus model with additional required paths
    python infer.py \
        --expid "$EXPID" \
        --data_root "$OPA_PATH" \
        --sg_root "$SG_PATH" \
        --gpt2_path "$GPT2_PATH" \
        --epoch "$EPOCH" \
        --eval_type "$EVAL_TYPE" \
        "$@"
else
    # Standard case for other models
    python infer.py \
        --expid "$EXPID" \
        --data_root "$OPA_PATH" \
        --epoch "$EPOCH" \
        --eval_type "$EVAL_TYPE" \
        "$@"
fi

# Return to original directory
cd "$PROJECT_ROOT"
