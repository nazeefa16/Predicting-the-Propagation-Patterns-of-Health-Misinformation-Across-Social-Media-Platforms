#!/usr/bin/env python3
"""
Main module for health misinformation detection system.

This script serves as the entry point for training and evaluating
misinformation detection models with a modular architecture.
"""

import os
import logging
import argparse
import yaml
import pandas as pd
from typing import Dict, Any, Optional

from models.traditional_models import LogisticRegressionModel
from models.transformer_models import BERTModel, RoBERTaModel
from models.llm_models import QwenModel
from models.rag_models import RAGTransformerModel, RAGLLMModel, BatchRetriever
from data.knowledge_base import KNOWLEDGE_BASE
from evaluation.metrics import evaluate_model, compare_models
from utils.helpers import set_seed, setup_logging
from models.traditional_models import *
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Health Misinformation Detection System")
    
    parser.add_argument("--config", type=str, default="/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms/global/config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="both",
                        help="Operation mode: train, evaluate, or both")
    parser.add_argument("--model_type", type=str, 
                        choices=["lr", "bert", "roberta", "bert_rag", "roberta_rag", "llm", "llm_rag"],
                        help="Type of model to use (overrides config)")
    parser.add_argument("--data_path", type=str, help="Path to dataset (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with dataset
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} samples from {data_path}")
        
        # Check required columns
        required_columns = ["content", "label"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Standardize label column
        if df["label"].dtype == object:
            # Map string labels to integers
            label_map = {"Reliable": 0, "Misinformation": 1}
            df["label_encoded"] = df["label"].map(label_map)
        else:
            # Already numeric
            df["label_encoded"] = df["label"]
        
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def create_model(model_config: Dict[str, Any], use_rag: bool = False) -> Any:
    """
    Create model based on configuration
    
    Args:
        model_config: Model configuration dictionary
        use_rag: Whether to use RAG enhancement
        
    Returns:
        Initialized model
    """
    model_type = model_config["type"].lower()
    
    # Common parameters
    common_params = {
        k: v for k, v in model_config.items() 
        if k not in ["type", "name", "use_rag"]
    }
    
    try:
        # Create model based on type
        if model_type == "traditional":
            model_name = model_config.get("name", "logistic_regression")
            
            if "logistic" in model_name.lower():
                model = LogisticRegressionModel(**common_params)
            elif "naive" in model_name.lower():
                base_model = NaiveBayesModel(**common_params)
            else:
                raise ValueError(f"Unsupported traditional model: {model_name}")
                
        elif model_type == "transformer":
            model_name = model_config.get("name", "bert-base-uncased")
            
            if "bert" in model_name.lower() and "roberta" not in model_name.lower():
                base_model = BERTModel(model_name=model_name, **common_params)
            elif "roberta" in model_name.lower():
                base_model = RoBERTaModel(model_name=model_name, **common_params)
            else:
                raise ValueError(f"Unsupported transformer model: {model_name}")
                
            # Apply RAG wrapper if requested
            if use_rag:
                logger.info(f"Creating RAG-enhanced transformer model with {model_name}")
                
                # Initialize retriever
                retriever = BatchRetriever(embedding_model=model_config.get("embedding_model", 
                                                         "sentence-transformers/all-MiniLM-L6-v2"))
                retriever.index_knowledge_base(KNOWLEDGE_BASE)
                
                model = RAGTransformerModel(
                    base_model_name=model_name,
                    transformer_type="bert" if "bert" in model_name.lower() else "roberta",
                    retriever=retriever,
                    top_k=model_config.get("top_k", 2),
                    **common_params
                )
            else:
                model = base_model
                
        elif model_type == "llm":
            model_name = model_config.get("name", "Qwen/Qwen2.5-7B-Instruct-1M")
            
            if "qwen" in model_name.lower():
                base_model = QwenModel(model_name=model_name, **common_params)
            else:
                # Default to Qwen for other LLMs
                base_model = QwenModel(model_name=model_name, **common_params)
                
            # Apply RAG wrapper if requested
            if use_rag:
                logger.info(f"Creating RAG-enhanced LLM model with {model_name}")
                
                # Initialize retriever
                retriever = BatchRetriever(embedding_model=model_config.get("embedding_model", 
                                                         "sentence-transformers/all-MiniLM-L6-v2"))
                retriever.index_knowledge_base(KNOWLEDGE_BASE)
                
                model = RAGLLMModel(
                    base_model_name=model_name,
                    llm_type="qwen",  # Adjust based on model name if needed
                    retriever=retriever,
                    top_k=model_config.get("top_k", 2),
                    **common_params
                )
            else:
                model = base_model
                
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        logger.info(f"Created model: {model}")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise



def train_model(
    model: Any, 
    data: pd.DataFrame, 
    train_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train model with given data"""
    try:
        logger.info(f"Training model {model}")
        
        # Prepare data
        prepared_data = model.prepare_data(
            data,
            text_column=train_config.get("text_column", "content"),
            label_column=train_config.get("label_column", "label"),  # Use "label" not "label_encoded"
            test_size=train_config.get("test_size", 0.2)
        )
        
        # Make sure learning_rate is a float
        learning_rate = float(train_config.get("learning_rate", 5e-5))
        
        # Check model type to handle different training approaches
        if isinstance(model, TraditionalModel):
            # Traditional model training
            train_results = model.train(
                train_data=prepared_data,
                val_data=prepared_data,  # Pass the same data for validation
                num_epochs=train_config.get("epochs", 1),
                batch_size=train_config.get("batch_size", 256),
                learning_rate=learning_rate,
                **train_config.get("extra_params", {})
            )
        elif any(cls_name in model.__class__.__name__ for cls_name in ["QwenModel", "LlamaModel", "MixtralModel", "RAGLLMModel"]):
            # LLM model - just perform inference, no training needed
            logger.info(f"LLM model detected: {model.__class__.__name__} - skipping training, will perform inference only")
            # Just return empty results - LLMs don't need training
            train_results = {}
        else:
            # Transformer model training
            train_results = model.train(
                train_data={"train_dataset": prepared_data["train_dataset"], "train_loader": prepared_data["train_loader"]},
                val_data={"val_dataset": prepared_data["val_dataset"], "val_loader": prepared_data["val_loader"]},
                num_epochs=train_config.get("epochs", 1),
                batch_size=train_config.get("batch_size", 256),
                learning_rate=learning_rate,
                **train_config.get("extra_params", {})
            )
        
        return {
            "model": model,
            "prepared_data": prepared_data,
            "train_results": train_results
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_models(
    models_data: Dict[str, Dict[str, Any]],
    eval_config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate and compare multiple models
    
    Args:
        models_data: Dictionary mapping model names to their data
        eval_config: Evaluation configuration
        output_dir: Output directory
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        logger.info(f"Evaluating {len(models_data)} models")
        
        # Evaluate each model
        all_results = {}
        for model_name, model_data in models_data.items():
            model = model_data["model"]
            prepared_data = model_data["prepared_data"]
            
            # Evaluate model
            eval_results = model.evaluate(
                test_data=prepared_data,
                **eval_config.get("extra_params", {})
            )
            
            # Add to results
            all_results[model_name] = eval_results
            
            # Check for different key names in evaluation results
            accuracy = eval_results.get('accuracy', eval_results.get('eval_accuracy', None))
            f1 = eval_results.get('f1', eval_results.get('eval_f1', None))
            
            if accuracy is not None and f1 is not None:
                logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            else:
                # Just log the available metrics
                logger.info(f"Model {model_name} - Evaluation results: {eval_results}")
            
        # Compare models
        compare_dir = os.path.join(output_dir, "comparison")
        os.makedirs(compare_dir, exist_ok=True)
        
        comparison = compare_models(
            models_results=all_results,
            output_dir=compare_dir
        )
        
        return {
            "individual_results": all_results,
            "comparison": comparison
        }
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise

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
    if args.model_type:
        if args.model_type == "lr":
            config["model"]["type"] = "traditional"
            config["model"]["name"] = "logistic_regression"
        elif args.model_type == "bert":
            config["model"]["type"] = "transformer"
            config["model"]["name"] = "bert-base-uncased"
        elif args.model_type == "roberta":
            config["model"]["type"] = "transformer"
            config["model"]["name"] = "roberta-base"
        elif args.model_type == "bert_rag":
            config["model"]["type"] = "transformer"
            config["model"]["name"] = "bert-base-uncased"
            config["model"]["use_rag"] = True
        elif args.model_type == "roberta_rag":
            config["model"]["type"] = "transformer"
            config["model"]["name"] = "roberta-base"
            config["model"]["use_rag"] = True
        elif args.model_type == "llm":
            config["model"]["type"] = "llm"
            config["model"]["name"] = "Qwen/Qwen2.5-7B-Instruct-1M"
        elif args.model_type == "llm_rag":
            config["model"]["type"] = "llm"
            config["model"]["name"] = "Qwen/Qwen2.5-7B-Instruct-1M"
            config["model"]["use_rag"] = True
    
    if args.data_path:
        config["data"]["path"] = args.data_path
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Create output directory
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data(config["data"]["path"])
    print(f"Dta size {len(data)}")
    # Training and evaluation
    models_data = {}
    
    # Training/evaluation for the main model
    model_config = config["model"]
    use_rag = model_config.get("use_rag", False)
    model_name = f"{model_config['type']}_{model_config['name'].split('/')[-1]}"
    if use_rag:
        model_name += "_rag"
    
    model = create_model(model_config, use_rag=use_rag)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    if args.mode in ["train", "both"]:
        logger.info(f"Training {model_name}")
        train_results = train_model(model, data, config["training"])
        
        # Save model
        model.save(model_dir)
        
        models_data[model_name] = train_results
    
    # Load model if only evaluating
    if args.mode == "evaluate" and not models_data:
        logger.info(f"Loading {model_name} from {model_dir}")
        model.load(model_dir)
        
        # Prepare data
        prepared_data = model.prepare_data(
            data,
            text_column=config["training"].get("text_column", "content"),
            label_column=config["training"].get("label_column", "label_encoded")
        )
        
        models_data[model_name] = {
            "model": model,
            "prepared_data": prepared_data
        }
    
    # Evaluate models
    if args.mode in ["evaluate", "both"]:
        logger.info("Evaluating models")
        eval_results = evaluate_models(models_data, config["evaluation"], output_dir)
        
        # Print summary
        if model_name in eval_results["individual_results"]:
            metrics = eval_results["individual_results"][model_name]
            print(f"Accuracy: {metrics.get('accuracy') or metrics.get('eval_accuracy'):.4f}")
            print(f"F1 Score: {metrics.get('f1') or metrics.get('eval_f1'):.4f}")
            print(f"Precision: {metrics.get('precision') or metrics.get('eval_precision'):.4f}")
            print(f"Recall: {metrics.get('recall') or metrics.get('eval_recall'):.4f}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()