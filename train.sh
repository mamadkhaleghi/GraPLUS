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

# Build and display the command before execution
if [ "$MODEL_NAME" = "graplus" ]; then
    # Special case for graplus model with additional required paths
    CMD="cd models/$MODEL_NAME && python main.py \\
    --expid \"$EXPID\" \\
    --data_root \"$OPA_PATH\" \\
    --sg_root \"$SG_PATH\" \\
    --gpt2_path \"$GPT2_PATH\""
    
    # Add any additional arguments
    for arg in "$@"; do
        CMD+=" \\
    $arg"
    done
else
    # Standard case for other models
    CMD="cd models/$MODEL_NAME && python main.py \\
    --expid \"$EXPID\" \\
    --data_root \"$OPA_PATH\""
    
    # Add any additional arguments
    for arg in "$@"; do
        CMD+=" \\
    $arg"
    done
fi

# Print the command for visibility
echo "########################################################### Executing command:"
echo "$CMD"
echo "cd \"$PROJECT_ROOT\""
echo "########################################################### Training files will be saved to:"
echo " - Model checkpoints: $PROJECT_ROOT/result/$EXPID/models/*.pth"
echo " - Log file:          $PROJECT_ROOT/result/$EXPID/"
echo " - sample files:      $PROJECT_ROOT/result/$EXPID/sample/"
echo " - TensorBoard:       $PROJECT_ROOT/result/$EXPID/tblog/"
echo "###########################################################"
echo ""

# Change to the specific model directory
cd "models/$MODEL_NAME" || {
    echo "Error: Could not change to models/$MODEL_NAME directory"
    exit 1
}

# Run the training script with model-specific arguments
if [ "$MODEL_NAME" = "graplus" ]; then
    # Special case for graplus model with additional required paths
    python main.py \
        --expid "$EXPID" \
        --data_root "$OPA_PATH" \
        --sg_root "$SG_PATH" \
        --gpt2_path "$GPT2_PATH" \
        "$@"
else
    # Standard case for other models
    python main.py \
        --expid "$EXPID" \
        --data_root "$OPA_PATH" \
        "$@"
fi

# Return to original directory
cd "$PROJECT_ROOT"
