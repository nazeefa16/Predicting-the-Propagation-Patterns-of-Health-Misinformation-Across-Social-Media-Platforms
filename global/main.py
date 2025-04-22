#!/usr/bin/env python3
"""
Main module for health misinformation detection system.
This is the entry point for training and evaluating models.
"""

import os
import logging
import argparse
import yaml
from typing import Dict, Any, List, Tuple

from utils.helpers import set_seed, setup_logging, get_abs_path
from data.loader import DataLoader
from models.traditional import TraditionalModel
from models.transformer import TransformerModel
from models.rag import RAGModel, LLMRAGModel
from evaluation.metrics import evaluate_model, compare_models
from models.llm import VLLMModel
# Optional import for knowledge base
try:
    from data.knowledge_base import KNOWLEDGE_BASE
except ImportError:
    KNOWLEDGE_BASE = {}

logger = logging.getLogger(__name__)

# Define available models and their variants
AVAILABLE_MODELS = {
    "traditional": [
        "logistic_regression", 
        "naive_bayes",
        "random_forest", 
        "svm"
    ],
    "transformer": [
        "distilbert-base-uncased", 
        "distilroberta-base",
    ],
    "rag": [
        "distilroberta-base"
    ],
    "llm_rag": [
        "Qwen/Qwen2.5-7B-Instruct-1M",
    ],
    "llm": [
        "Qwen/Qwen2.5-7B-Instruct-1M",
    ]
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Health Misinformation Detection")
    
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="both",
                        help="Operation mode: train, evaluate, or both")
    parser.add_argument("--model_type", type=str, 
                        choices=["traditional", "transformer","llm" ,"rag", "llm_rag", "all"],
                        help="Type of model to use (overrides config)")
    parser.add_argument("--model_name", type=str,
                        help="Name of specific model to use (overrides config)")
    parser.add_argument("--data_path", type=str, help="Path to dataset (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
    """
    abs_path = get_abs_path(config_path)
    
    with open(abs_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate and convert numeric values
    if "training" in config:
        # Convert numeric values in training section
        if "learning_rate" in config["training"]:
            config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
        if "batch_size" in config["training"]:
            config["training"]["batch_size"] = int(config["training"]["batch_size"])
        if "epochs" in config["training"]:
            config["training"]["epochs"] = int(config["training"]["epochs"])
        if "test_size" in config["training"]:
            config["training"]["test_size"] = float(config["training"]["test_size"])
        if "val_size" in config["training"]:
            config["training"]["val_size"] = float(config["training"]["val_size"])
        if "seed" in config["training"]:
            config["training"]["seed"] = int(config["training"]["seed"])
    
    # Set sensible defaults for missing values
    if "model" not in config:
        config["model"] = {}
    if "type" not in config["model"]:
        config["model"]["type"] = "transformer"
    if "name" not in config["model"]:
        config["model"]["name"] = "bert-base-uncased"
    
    if "training" not in config:
        config["training"] = {}
    if "epochs" not in config["training"]:
        config["training"]["epochs"] = 3
    if "batch_size" not in config["training"]:
        config["training"]["batch_size"] = 128
    if "learning_rate" not in config["training"]:
        config["training"]["learning_rate"] = 5e-5
    
    return config

def create_model(model_type: str, model_name: str) -> Any:
    """Create model based on type and name"""
    print(f"Create model received model_name : {model_name}")
    if model_type == "traditional":
        return TraditionalModel(model_name=model_name)
    elif model_type == "transformer":
        return TransformerModel(config={"model_name": model_name})
    elif model_type == "rag":
        return RAGModel(model_name=model_name, knowledge_base=KNOWLEDGE_BASE)
    elif model_type == "llm_rag":
        return LLMRAGModel(model_name=model_name, knowledge_base=KNOWLEDGE_BASE)
    elif model_type == "llm":
        return VLLMModel(model_name=model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_and_evaluate_model(
    model_type: str,
    model_name: str,
    data_splits: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    mode: str
) -> Dict[str, Any]:
    """
    Train and evaluate a single model
    
    Args:
        model_type: Type of model
        model_name: Name of model
        data_splits: Data splits from DataLoader
        config: Configuration dictionary
        output_dir: Output directory
        mode: Operation mode (train, evaluate, both)
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Processing {model_type} model: {model_name}")
    
    # Create model
    model = create_model(model_type, model_name)
    
    # Create model directory
    model_dir = os.path.join(output_dir, f"{model_type}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Process data for model
    processed_data = model.prepare_data(data_splits)
    
    # Training
    if mode in ["train", "both"]:
        logger.info(f"Training {model_name}")
        
        # Ensure learning_rate is a float
        learning_rate = float(config["training"]["learning_rate"])
        
        # Train model
        train_results = model.train(
            train_data={k: v for k, v in processed_data.items() if k.startswith("train")},
            val_data={k: v for k, v in processed_data.items() if k.startswith("val")},
            num_epochs=int(config["training"]["epochs"]),
            learning_rate=learning_rate
        )
        
        # Save model
        model.save(model_dir)
        
        logger.info(f"Training completed: {train_results}")
    
    # Evaluation
    eval_results = None
    if mode in ["evaluate", "both"]:
        logger.info(f"Evaluating {model_name}")
        
        def get_metric(results, name):
            if name in results:
                return results[name]
            elif f"eval_{name}" in results:
                return results[f"eval_{name}"]
            else:
                logger.warning(f"Metric {name} not found in results")
                return 0.0 
        # Load model if only evaluating
        if mode == "evaluate":
            model.load(model_dir)
        
        # Evaluate model
        test_data = {k: v for k, v in processed_data.items() if k.startswith("test")}
        eval_results = model.evaluate(test_data)
        
        # Generate evaluation plots and metrics
        metrics = {
            "accuracy": get_metric(eval_results, "accuracy"),
            "precision": get_metric(eval_results, "precision"),
            "recall": get_metric(eval_results, "recall"),
            "f1": get_metric(eval_results, "f1")
        }
        eval_results.update(metrics)

        evaluation_metrics = evaluate_model(
            predictions=eval_results["predictions"],
            true_labels=eval_results["true_labels"],
            probabilities=eval_results["probabilities"],
            output_dir=model_dir,
            model_name=model_name
        )

        # Print summary
        logger.info(f"Evaluation results for {model_type}_{model_name}:")
        logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"  F1 Score: {eval_results['f1']:.4f}")
        logger.info(f"  Precision: {eval_results['precision']:.4f}")
        logger.info(f"  Recall: {eval_results['recall']:.4f}")
    
    return eval_results

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type and args.model_type != "all":
        config["model"]["type"] = args.model_type
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Create output directory
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    data_loader = DataLoader(
        file_path=get_abs_path(config["data"]["path"]),
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"]
    )
    
    data_splits = data_loader.prepare_data(
        test_size=float(config["training"]["test_size"]),
        val_size=float(config["training"]["val_size"]) if "val_size" in config["training"] else None,
        random_state=int(config["training"]["seed"])
    )
    
    # Determine which models to train
    models_to_train = []
    
    if args.model_type == "all":
        # Train all available models
        for model_type, model_variants in AVAILABLE_MODELS.items():
            for model_name in model_variants:
                models_to_train.append((model_type, model_name))
    elif args.model_name:
        # Train specific model
        models_to_train.append((config["model"]["type"], args.model_name))
    else:
        # Train all variants of a specific model type
        model_type = config["model"]["type"]
        if model_type in AVAILABLE_MODELS:
            for model_name in AVAILABLE_MODELS[model_type]:
                models_to_train.append((model_type, model_name))
        else:
            # Fallback to default model
            models_to_train.append((model_type, config["model"]["name"]))
    
    # Train and evaluate each model
    all_metrics = {}
    
    for model_type, model_name in models_to_train:
        logger.info(f"Processing model: {model_type}_{model_name}")
        
        try:
            eval_results = train_and_evaluate_model(
                model_type=model_type,
                model_name=model_name,
                data_splits=data_splits,
                config=config,
                output_dir=output_dir,
                mode=args.mode
            )
            
            if eval_results:
                model_key = f"{model_type}_{model_name}"
                all_metrics[model_key] = {
                    "accuracy": eval_results["accuracy"],
                    "precision": eval_results["precision"],
                    "recall": eval_results["recall"],
                    "f1": eval_results["f1"]
                }
                
                if "roc_auc" in eval_results:
                    all_metrics[model_key]["roc_auc"] = eval_results["roc_auc"]
                if "pr_auc" in eval_results:
                    all_metrics[model_key]["pr_auc"] = eval_results["pr_auc"]
                
        except Exception as e:
            logger.error(f"Error processing {model_type}_{model_name}: {e}")
            logger.exception(e)
    
    # Generate model comparison
    if len(all_metrics) > 1 and args.mode in ["evaluate", "both"]:
        logger.info("Generating model comparison")
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Compare models
        compare_models(all_metrics, comparison_dir)
        
        # Log best model for each metric
        for metric in ["accuracy", "f1", "precision", "recall"]:
            best_model = max(all_metrics.items(), key=lambda x: x[1].get(metric, 0))
            logger.info(f"Best model by {metric}: {best_model[0]} ({best_model[1].get(metric, 0):.4f})")
    
    logger.info("All models processed")

if __name__ == "__main__":
    main()