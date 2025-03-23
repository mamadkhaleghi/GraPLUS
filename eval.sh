#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <epoch>"
    echo "Example: $0 graplus 21"
    exit 1
fi

# Store arguments in named variables
MODEL_NAME="$1"
EPOCH="$2"

# Print execution information
echo "Running evaluations for:"
echo "Model: $MODEL_NAME"
echo "Epoch: $EPOCH"
echo "----------------------------------------"

# Run accuracy evaluation
echo "Running accuracy evaluation..."
sh script/eval_acc.sh "$MODEL_NAME" "$EPOCH" ../BINARY_CLASSIFIER/best-acc.pth

# # Run FID evaluation
echo "Running FID evaluation..."
sh script/eval_fid.sh "$MODEL_NAME" "$EPOCH" ../OPA_dataset/com_pic_testpos299

# Run LPIPS evaluation
echo "Running LPIPS evaluation..."
sh script/eval_lpips.sh "$MODEL_NAME" "$EPOCH"

echo "----------------------------------------"
echo "All evaluations completed!"