#!/bin/bash

# Path to your Python environment
PYTHON_PATH="/home/aledel/repos/aphasia-inference/aphasia/bin/python3.13"

# Base directory
BASE_DIR="/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms"

# Output directory for comparison
OUTPUT_DIR="$BASE_DIR/output/complete_evaluation"
mkdir -p "$OUTPUT_DIR"

# Data path
DATA_PATH="$BASE_DIR/global/data/merged_dataset.csv"

# Run training and evaluation for each model
echo "Running models..."

# BERT model
echo "==== Training and evaluating BERT ===="
$PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type bert --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/bert"

# RoBERTa model
echo "==== Training and evaluating RoBERTa ===="
$PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type roberta --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/roberta"

# Logistic Regression model
echo "==== Training and evaluating Logistic Regression ===="
$PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type lr --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/lr"

# Optional: RAG models
if [ "$1" = "--rag" ]; then
  echo "==== Training and evaluating BERT with RAG ===="
  $PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type bert_rag --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/bert_rag"
  
  echo "==== Training and evaluating RoBERTa with RAG ===="
  $PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type roberta_rag --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/roberta_rag"
fi

# Optional: LLM models
if [ "$1" = "--llm" ] || [ "$2" = "--llm" ]; then
  echo "==== Training and evaluating LLM ===="
  $PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type llm --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/llm"
  
  if [ "$1" = "--rag" ] || [ "$2" = "--rag" ]; then
    echo "==== Training and evaluating LLM with RAG ===="
    $PYTHON_PATH "$BASE_DIR/global/main.py" --mode both --model_type llm_rag --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR/llm_rag"
  fi
fi

# Run model comparison
echo "==== Comparing all models ===="

# Make the comparison script executable
chmod +x "$BASE_DIR/global/evaluation/compare_all_models.py"

# Run the comparison script
mkdir -p "$OUTPUT_DIR/comparison"
$PYTHON_PATH "$BASE_DIR/global/evaluation/compare_all_models.py" --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/comparison"

echo "All done! Results are saved in $OUTPUT_DIR/comparison"
echo "Key results files:"
echo "- $OUTPUT_DIR/comparison/model_summary.csv - CSV with all model metrics"
echo "- $OUTPUT_DIR/comparison/model_summary.png - Visual comparison of models"
echo "- $OUTPUT_DIR/comparison/model_comparison.csv - Detailed comparison"
echo "- $OUTPUT_DIR/comparison/model_comparison.png - Detailed visual comparison"

# Print a summary of the results
echo "===================== RESULTS SUMMARY ====================="
if [ -f "$OUTPUT_DIR/comparison/model_summary.csv" ]; then
  echo "Model performance metrics:"
  cat "$OUTPUT_DIR/comparison/model_summary.csv"
  
  # Find best model for F1 score
  BEST_F1=$(cat "$OUTPUT_DIR/comparison/model_summary.csv" | tail -n+2 | sort -t, -k5 -nr | head -1)
  echo ""
  echo "Best model (by F1 score): $BEST_F1"
else
  echo "Results summary not available. Check for errors in the logs."
fi
echo "=========================================================="