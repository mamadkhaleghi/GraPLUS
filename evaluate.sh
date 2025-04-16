#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <epoch> <metric>"
    echo "Example: $0 graplus 21 accuracy"
    echo "Example: $0 graplus 21 fid"
    echo "Example: $0 graplus 21 lpips"
    echo "Example: $0 graplus 21 sp"
    echo "Example: $0 graplus 21 all"
    exit 1
fi

# Store arguments in named variables
MODEL_NAME="$1"
EPOCH="$2"
METRIC="$3"

# Print execution information
echo "Running evaluations for $METRIC:"
echo "Model: $MODEL_NAME"
echo "Epoch: $EPOCH"
echo "----------------------------------------"

# Conditional execution based on METRIC
if [ "$METRIC" = "accuracy" ]; then
    # Run accuracy evaluation
    echo "Running accuracy evaluation..."
    sh script/eval_acc.sh "$MODEL_NAME" "$EPOCH"

elif [ "$METRIC" = "fid" ]; then
    # Run FID evaluation
    echo "Running FID evaluation..."
    sh script/eval_fid.sh "$MODEL_NAME" "$EPOCH"

elif [ "$METRIC" = "lpips" ]; then
    # Run LPIPS evaluation
    echo "Running LPIPS evaluation..."
    sh script/eval_lpips.sh "$MODEL_NAME" "$EPOCH"

elif [ "$METRIC" = "sp" ]; then
    # Run Spatial Precision evaluation
    echo "Running Spatial Precision evaluation..."
    sh script/eval_spatial_precision.sh "$MODEL_NAME" "$EPOCH"

elif [ "$METRIC" = "all" ]; then
    # Run all evaluations
    echo "Running all evaluations..."
    echo "Running accuracy evaluation..."
    sh script/eval_acc.sh "$MODEL_NAME" "$EPOCH"
    echo "Running FID evaluation..."
    sh script/eval_fid.sh "$MODEL_NAME" "$EPOCH"
    echo "Running LPIPS evaluation..."
    sh script/eval_lpips.sh "$MODEL_NAME" "$EPOCH"
    echo "Running Spatial Precision evaluation..."
    sh script/eval_spatial_precision.sh "$MODEL_NAME" "$EPOCH"

else
    echo "Error: Invalid metric specified. Must be 'accuracy', 'fid', 'lpips', 'sp', or 'all'"
    exit 1
fi