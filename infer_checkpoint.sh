#!/bin/bash

# Check if minimum required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <eval_type> <model_name>"
    echo "Example: $0 eval graplus"
    exit 1
fi

# Store model name and evaluation type
EVAL_TYPE="$1"
MODEL_NAME="$2"

# Automatically set the epoch based on model name
case "$MODEL_NAME" in
    "graplus")
        EPOCH=21
        ;;
    "csanet")
        EPOCH=18
        ;;
    "cagan")
        EPOCH=15
        ;;
    "graconet")
        EPOCH=11
        ;;
    "placenet")
        EPOCH=9
        ;;
    "terse")
        EPOCH=11
        ;;
    *)
        echo "Error: Unknown model name '$MODEL_NAME'. Cannot determine epoch."
        echo "Model Name should be one of following names:"
        echo "graplus, csanet, cagan, graconet, placenet, terse"
        exit 1
        ;;
esac

echo "Using $MODEL_NAME checkpoint (epoch $EPOCH) for $EVAL_TYPE evaluation"

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

# Build and display the command before execution
if [ "$EVAL_TYPE" = "evaluni" ]; then
    # For evaluni, add the --repeat 10 flag
    if [ "$MODEL_NAME" = "graplus" ]; then
        # Special case for graplus model with additional required paths
        CMD="cd models/$MODEL_NAME && python infer.py \\
    --expid \"$MODEL_NAME\" \\
    --data_root \"$OPA_PATH\" \\
    --sg_root \"$SG_PATH\" \\
    --gpt2_path \"$GPT2_PATH\" \\
    --epoch \"$EPOCH\" \\
    --eval_type \"$EVAL_TYPE\" \\
    --repeat 10"
    else
        # Standard case for other models
        CMD="cd models/$MODEL_NAME && python infer.py \\
    --expid \"$MODEL_NAME\" \\
    --data_root \"$OPA_PATH\" \\
    --epoch \"$EPOCH\" \\
    --eval_type \"$EVAL_TYPE\" \\
    --repeat 10"
    fi
else
    # For all other eval_types (including "eval"), use the original command
    if [ "$MODEL_NAME" = "graplus" ]; then
        # Special case for graplus model with additional required paths
        CMD="cd models/$MODEL_NAME && python infer.py \\
    --expid \"$MODEL_NAME\" \\
    --data_root \"$OPA_PATH\" \\
    --sg_root \"$SG_PATH\" \\
    --gpt2_path \"$GPT2_PATH\" \\
    --epoch \"$EPOCH\" \\
    --eval_type \"$EVAL_TYPE\""
    else
        # Standard case for other models
        CMD="cd models/$MODEL_NAME && python infer.py \\
    --expid \"$MODEL_NAME\" \\
    --data_root \"$OPA_PATH\" \\
    --epoch \"$EPOCH\" \\
    --eval_type \"$EVAL_TYPE\""
    fi
fi

# Print the command for visibility
echo "###########################################################"
echo "Executing command:"
echo "$CMD"
echo "cd "$PROJECT_ROOT""
echo "###########################################################"
echo ""

# Execute the actual command (we still use the original cd and python execution for this)
if [ "$EVAL_TYPE" = "evaluni" ]; then
    # For evaluni, add the --repeat 10 flag
    if [ "$MODEL_NAME" = "graplus" ]; then
        # Special case for graplus model with additional required paths
        python infer.py \
            --expid "$MODEL_NAME" \
            --data_root "$OPA_PATH" \
            --sg_root "$SG_PATH" \
            --gpt2_path "$GPT2_PATH" \
            --epoch "$EPOCH" \
            --eval_type "$EVAL_TYPE" \
            --repeat 10
    else
        # Standard case for other models
        python infer.py \
            --expid "$MODEL_NAME" \
            --data_root "$OPA_PATH" \
            --epoch "$EPOCH" \
            --eval_type "$EVAL_TYPE" \
            --repeat 10
    fi
else
    # For all other eval_types (including "eval"), use the original command
    if [ "$MODEL_NAME" = "graplus" ]; then
        # Special case for graplus model with additional required paths
        python infer.py \
            --expid "$MODEL_NAME" \
            --data_root "$OPA_PATH" \
            --sg_root "$SG_PATH" \
            --gpt2_path "$GPT2_PATH" \
            --epoch "$EPOCH" \
            --eval_type "$EVAL_TYPE"
    else
        # Standard case for other models
        python infer.py \
            --expid "$MODEL_NAME" \
            --data_root "$OPA_PATH" \
            --epoch "$EPOCH" \
            --eval_type "$EVAL_TYPE"
    fi
fi

# Return to original directory
cd "$PROJECT_ROOT"
