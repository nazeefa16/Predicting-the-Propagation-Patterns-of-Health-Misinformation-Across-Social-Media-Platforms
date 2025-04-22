# Health Misinformation Detection

A modular and efficient system for detecting health misinformation in text content.

## Overview

This project implements different machine learning approaches to identify health misinformation in social media content and other text sources. The system supports traditional ML, transformer-based, and retrieval-augmented generation (RAG) approaches.

## Features

- Multiple model architectures (traditional ML, transformers, RAG)
- Standardized evaluation metrics and visualizations
- Knowledge-enhanced classification with RAG
- Simple configuration and training interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/health-misinfo-detection.git
cd health-misinfo-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
# Train a basic transformer model
./train.sh --model transformer --model-name bert-base-uncased

# Train a traditional ML model
./train.sh --model traditional --model-name logistic_regression

# Train a RAG model
./train.sh --model rag --model-name bert-base-uncased
```

### Evaluating a Model

```bash
# Evaluate only
./train.sh --mode evaluate --model transformer --model-name bert-base-uncased
```

### Using Custom Data

```bash
./train.sh --data path/to/your/data.csv --model transformer
```

## Configuration

The system can be configured via `config/default_config.yaml` or by passing configuration parameters directly:

```bash
./train.sh --config custom_config.yaml
```

Key configuration options:
- `data.path`: Path to dataset
- `model.type`: Model type (traditional, transformer, rag)
- `model.name`: Specific model name
- `training.epochs`: Number of training epochs
- `training.batch_size`: Batch size for training
- `output_dir`: Directory for saving outputs

## Models

### Traditional ML Models
- Logistic Regression
- Naive Bayes
- SVM

### Transformer Models
- BERT
- RoBERTa
- DistilBERT

### RAG Models
- Knowledge-enhanced transformers

## Project Structure

```
health_misinfo/
  ├── config/
  │   └── default_config.yaml      # Configuration
  ├── data/
  │   ├── loader.py                # Data loading
  │   ├── dataset.py               # Dataset classes
  │   └── knowledge_base.py        # Knowledge base for RAG
  ├── models/
  │   ├── base.py                  # Base model interface
  │   ├── traditional.py           # Traditional ML models
  │   ├── transformer.py           # Transformer models
  │   └── rag.py                   # RAG wrapper
  ├── evaluation/
  │   ├── metrics.py               # Metrics calculation
  │   └── visualization.py         # Visualization
  ├── utils/
  │   └── helpers.py               # Helper functions
  ├── main.py                      # Main entry point
  └── train.sh                     # Training script
```

## Results

The system generates comprehensive evaluation metrics and visualizations for each model:

- Accuracy, precision, recall, F1 score
- ROC curve and AUC
- Precision-recall curve
- Confusion matrix
- Threshold analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using PyTorch and Hugging Face Transformers
- Knowledge base derived from public health resources