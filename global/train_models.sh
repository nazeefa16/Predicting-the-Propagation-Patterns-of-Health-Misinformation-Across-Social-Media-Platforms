#!/bin/bash

# Script for training health misinformation detection models

# Default values
CONFIG="config/default_config.yaml"
MODE="both"
OUTPUT_DIR="output"
DATA_PATH="data/merged_dataset.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    -h|--help)
      echo "Usage: ./train.sh [options]"
      echo "Options:"
      echo "  --config CONFIG      Configuration file path"
      echo "  --mode MODE          Operation mode: train, evaluate, or both"
      echo "  --model TYPE         Model type: traditional, transformer, or rag"
      echo "  --model-name NAME    Specific model name"
      echo "  --output DIR         Output directory"
      echo "  --data PATH          Dataset path"
      echo "  --debug              Enable debug mode"
      echo "  -h, --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

PYTHON_PATH="/home/aledel/repos/aphasia-inference/aphasia/bin/python3.13"

# Prepare command
CMD="/home/aledel/repos/aphasia-inference/aphasia/bin/python3.13 main.py --config ${CONFIG} --mode ${MODE}"

# Add optional arguments
if [ ! -z "$MODEL_TYPE" ]; then
  CMD="${CMD} --model_type ${MODEL_TYPE}"
fi

if [ ! -z "$MODEL_NAME" ]; then
  CMD="${CMD} --model_name ${MODEL_NAME}"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="${CMD} --output_dir ${OUTPUT_DIR}"
fi

if [ ! -z "$DATA_PATH" ]; then
  CMD="${CMD} --data_path ${DATA_PATH}"
fi

if [ ! -z "$DEBUG" ]; then
  CMD="${CMD} --debug"
fi

# Run command
echo "Running: ${CMD}"
eval $CMD

# Check for errors
if [ $? -ne 0 ]; then
  echo "Error: Training failed."
  exit 1
fi

echo "Training completed successfully!"
echo "Results are saved in ${OUTPUT_DIR}"

# Print performance metrics if available
MODEL_DIRS=$(ls -d ${OUTPUT_DIR}/*/ 2>/dev/null)
if [ $? -eq 0 ]; then
  echo "Model performance:"
  for DIR in $MODEL_DIRS; do
    MODEL=$(basename "$DIR")
    METRICS_FILE="${DIR}/${MODEL}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
      echo "----- $MODEL -----"
      # Extract and print key metrics
      ACCURACY=$(grep -o '"accuracy": [0-9.]*' "$METRICS_FILE" | cut -d ' ' -f2)
      F1=$(grep -o '"f1": [0-9.]*' "$METRICS_FILE" | cut -d ' ' -f2)
      PRECISION=$(grep -o '"precision": [0-9.]*' "$METRICS_FILE" | cut -d ' ' -f2)
      RECALL=$(grep -o '"recall": [0-9.]*' "$METRICS_FILE" | cut -d ' ' -f2)
      
      echo "Accuracy:  $ACCURACY"
      echo "F1 Score:  $F1"
      echo "Precision: $PRECISION"
      echo "Recall:    $RECALL"
      echo ""
    fi
  done
fi