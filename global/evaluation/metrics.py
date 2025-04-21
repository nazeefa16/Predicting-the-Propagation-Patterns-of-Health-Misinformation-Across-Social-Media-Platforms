import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve, 
    auc, 
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        return numpy_to_python_type(obj)

def numpy_to_python_type(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that might contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle numpy scalar types
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
        
    if isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
        return float(obj)
        
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    
    if isinstance(obj, (np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python_type(i) for i in obj]
    
    # Return other types as is
    return obj

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file with proper handling of numpy types.
    
    Args:
        data: Data to save (may contain numpy types)
        filepath: Path to save JSON file
        indent: JSON indentation level
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=indent)

def calculate_basic_metrics(
    true_labels: np.ndarray, 
    predictions: np.ndarray
) -> Dict[str, float]:
    """
    Calculate basic classification metrics.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        
    Returns:
        Dictionary with basic metrics
    """
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0)
    }
    
    return metrics

def calculate_advanced_metrics(
    true_labels: np.ndarray, 
    predictions: np.ndarray, 
    probabilities: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate advanced classification metrics.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities (N x C) where C is the number of classes
        
    Returns:
        Dictionary with advanced metrics
    """
    metrics = {}
    
    # Confidence metrics
    correct_mask = (predictions == true_labels)
    if correct_mask.sum() > 0:
        metrics["mean_confidence_correct"] = np.mean(probabilities[np.arange(len(predictions)), true_labels][correct_mask])
    else:
        metrics["mean_confidence_correct"] = 0.0
    
    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        metrics["mean_confidence_incorrect"] = np.mean(probabilities[np.arange(len(predictions)), true_labels][incorrect_mask])
    else:
        metrics["mean_confidence_incorrect"] = 0.0
    
    # ROC and PR curves
    fpr, tpr, roc_thresholds = roc_curve(true_labels, probabilities[:, 1])
    metrics["roc_auc"] = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(true_labels, probabilities[:, 1])
    metrics["pr_auc"] = auc(recall, precision)
    
    # Calibration
    metrics["brier_score"] = brier_score_loss(true_labels, probabilities[:, 1])
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()
    if cm.size == 4:  # Binary case
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = int(tn), int(fp), int(fn), int(tp)
    
    # Curve data for plotting
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist()
    }
    
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
    }
    
    # Classification report (per class metrics)
    class_report = classification_report(true_labels, predictions, output_dict=True)
    metrics["classification_report"] = numpy_to_python_type(class_report)
    
    return metrics

def perform_threshold_analysis(
    true_labels: np.ndarray, 
    probabilities: np.ndarray, 
    steps: int = 10
) -> List[Dict[str, float]]:
    """
    Analyze model performance across different threshold values.
    
    Args:
        true_labels: Array of true labels
        probabilities: Array of prediction probabilities for the positive class
        steps: Number of threshold steps to evaluate
        
    Returns:
        List of dictionaries with metrics at different thresholds
    """
    thresholds = np.linspace(0.1, 0.9, steps)
    results = []
    
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(true_labels, preds)),
            "precision": float(precision_score(true_labels, preds, zero_division=0)),
            "recall": float(recall_score(true_labels, preds, zero_division=0)),
            "f1": float(f1_score(true_labels, preds, zero_division=0))
        }
        results.append(metrics)
    
    return results

def perform_error_analysis(
    texts: List[str], 
    true_labels: np.ndarray, 
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    knowledge_items: Optional[List[str]] = None, 
    max_samples: int = 100
) -> pd.DataFrame:
    """
    Perform detailed error analysis to understand when and why the model makes mistakes.
    
    Args:
        texts: List of text inputs
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities for the positive class
        knowledge_items: Optional list of knowledge items used for predictions
        max_samples: Maximum number of samples to include in the analysis
        
    Returns:
        DataFrame with error analysis
    """
    # Identify correct and incorrect predictions
    correct_mask = (predictions == true_labels)
    incorrect_mask = ~correct_mask
    
    # Define confidence levels
    confidence = probabilities.copy()
    # For predicted class 0, confidence should reflect that
    confidence[predictions == 0] = 1 - confidence[predictions == 0]
    
    high_conf = confidence >= 0.9
    med_conf = (confidence >= 0.7) & (confidence < 0.9)
    low_conf = confidence < 0.7
    
    # Create categories for analysis
    categories = []
    for i in range(len(true_labels)):
        if correct_mask[i]:
            if high_conf[i]:
                categories.append("Correct - High Confidence")
            elif med_conf[i]:
                categories.append("Correct - Medium Confidence")
            else:
                categories.append("Correct - Low Confidence")
        else:
            if high_conf[i]:
                categories.append("Incorrect - High Confidence")
            elif med_conf[i]:
                categories.append("Incorrect - Medium Confidence")
            else:
                categories.append("Incorrect - Low Confidence")
    
    # Convert predictions to category labels
    label_map = {0: "Factual", 1: "Misinformation"}
    true_labels_text = [label_map[l] for l in true_labels]
    pred_labels_text = [label_map[p] for p in predictions]
    
    # Build analysis dataframe
    data = {
        "text": texts[:len(true_labels)],  # Ensure same length
        "true_label": true_labels_text,
        "predicted_label": pred_labels_text,
        "confidence": confidence,
        "analysis_category": categories,
    }
    
    # Add knowledge items if available
    if knowledge_items is not None:
        # Ensure same length
        knowledge_items = knowledge_items[:len(true_labels)]
        data["knowledge_items"] = knowledge_items
    
    # Create dataframe
    analysis_df = pd.DataFrame(data)
    
    # Prioritize high-confidence errors and interesting cases
    high_conf_errors = analysis_df[analysis_df["analysis_category"] == "Incorrect - High Confidence"]
    other_errors = analysis_df[(analysis_df["true_label"] != analysis_df["predicted_label"]) & 
                              (analysis_df["analysis_category"] != "Incorrect - High Confidence")]
    interesting_correct = analysis_df[analysis_df["analysis_category"] == "Correct - Low Confidence"]
    
    # Combine and limit the number of samples
    result_df = pd.concat([
        high_conf_errors,
        other_errors,
        interesting_correct
    ])
    
    # Ensure we don't exceed max_samples
    if len(result_df) > max_samples:
        result_df = result_df.head(max_samples)
    
    # Add a column for text length - useful for analysis
    result_df["text_length"] = result_df["text"].apply(lambda x: len(str(x).split()))
    
    return result_df

def generate_visualizations(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str,
    curves: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate and save visualization plots for model evaluation.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities
        output_dir: Directory to save visualizations
        model_name: Name of the model for plot titles and filenames
        curves: Optional dictionary with pre-computed curve data (ROC, PR)
    """
    # Set up figure directory
    fig_dir = f"{output_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Factual', 'Misinformation'],
               yticklabels=['Factual', 'Misinformation'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{fig_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    
    if curves and 'roc_curve' in curves:
        fpr = curves['roc_curve']['fpr']
        tpr = curves['roc_curve']['tpr']
    else:
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{fig_dir}/{model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    
    if curves and 'pr_curve' in curves:
        precision = curves['pr_curve']['precision']
        recall = curves['pr_curve']['recall']
    else:
        precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
        
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{fig_dir}/{model_name}_pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Confidence Distribution
    plt.figure(figsize=(10, 6))
    correct_mask = (predictions == true_labels)
    
    # Plot distribution of probabilities for correct and incorrect predictions
    if any(correct_mask):
        sns.histplot(probabilities[:, 1][correct_mask], bins=20, alpha=0.5, label='Correct Predictions', color='green')
    if any(~correct_mask):
        sns.histplot(probabilities[:, 1][~correct_mask], bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title(f'Prediction Confidence Distribution - {model_name}')
    plt.legend()
    plt.savefig(f"{fig_dir}/{model_name}_confidence_dist.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_threshold_plot(
    threshold_results: List[Dict[str, float]],
    output_dir: str,
    model_name: str
) -> None:
    """
    Generate and save threshold analysis plot.
    
    Args:
        threshold_results: List of dictionaries with metrics at different thresholds
        output_dir: Directory to save visualization
        model_name: Name of the model for plot title and filename
    """
    # Set up figure directory
    fig_dir = f"{output_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Create dataframe from threshold results
    df = pd.DataFrame(threshold_results)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(df['threshold'], df['accuracy'], marker='o', label='Accuracy')
    plt.plot(df['threshold'], df['precision'], marker='s', label='Precision')
    plt.plot(df['threshold'], df['recall'], marker='^', label='Recall')
    plt.plot(df['threshold'], df['f1'], marker='d', label='F1 Score')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs. Threshold - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f"{fig_dir}/{model_name}_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(
    model: Any,
    test_data: Any,
    output_dir: str,
    model_name: str = "model",
    threshold: float = 0.5,
    knowledge_column: Optional[str] = None,
    text_column: Optional[str] = None,
    label_map: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Perform detailed evaluation of a model.
    
    Args:
        model: The trained model to evaluate
        test_data: Test data
        output_dir: Directory to save evaluation results
        model_name: Name of the model for saving results
        threshold: Classification threshold 
        knowledge_column: Column name for retrieved knowledge (for error analysis)
        text_column: Column name for original text (for error analysis)
        label_map: Optional mapping from numeric labels to text labels (default: {0: "Factual", 1: "Misinformation"})
    
    Returns:
        Dictionary with detailed evaluation metrics
    """
    logger.info(f"Performing detailed evaluation of {model_name}")
    
    if label_map is None:
        label_map = {0: "Factual", 1: "Misinformation"}
    
    # Get predictions
    predictions, probabilities = model.predict(test_data)
    
    # Get true labels from test data
    if hasattr(test_data, 'get') and 'test_labels' in test_data:
        true_labels = test_data['test_labels']
    elif hasattr(test_data, 'get') and 'labels' in test_data:
        true_labels = test_data['labels']
    else:
        # Try to extract from model's evaluate method
        eval_results = model.evaluate(test_data)
        true_labels = eval_results.get('true_labels')
        
        if true_labels is None:
            raise ValueError("Could not extract true labels from test data")
    
    # Apply threshold if needed
    if threshold != 0.5 and probabilities.shape[1] > 1:
        binary_preds = (probabilities[:, 1] >= threshold).astype(int)
    else:
        binary_preds = predictions
    
    # Calculate basic metrics
    results = calculate_basic_metrics(true_labels, binary_preds)
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(true_labels, binary_preds, probabilities)
    results.update(advanced_metrics)
    
    # Threshold analysis
    threshold_metrics = perform_threshold_analysis(true_labels, probabilities[:, 1])
    results["threshold_analysis"] = threshold_metrics
    
    # Generate visualizations
    generate_visualizations(
        true_labels, 
        binary_preds, 
        probabilities, 
        output_dir, 
        model_name
    )
    
    # Generate threshold plot
    generate_threshold_plot(
        threshold_metrics,
        output_dir,
        model_name
    )
    
    # Error analysis if text column is provided
    error_examples = None
    if text_column is not None:
        try:
            # Get the texts
            if hasattr(test_data, 'get') and f'test_{text_column}' in test_data:
                texts = test_data[f'test_{text_column}']
            elif hasattr(test_data, 'get') and text_column in test_data:
                texts = test_data[text_column]
            elif hasattr(test_data, 'test_df') and text_column in test_data.test_df.columns:
                texts = test_data.test_df[text_column].tolist()
            else:
                texts = []
                logger.warning(f"Could not find text column '{text_column}' for error analysis")
            
            # Get knowledge items if available
            knowledge_items = None
            if knowledge_column is not None:
                if hasattr(test_data, 'get') and f'test_{knowledge_column}' in test_data:
                    knowledge_items = test_data[f'test_{knowledge_column}']
                elif hasattr(test_data, 'get') and knowledge_column in test_data:
                    knowledge_items = test_data[knowledge_column]
                elif hasattr(test_data, 'test_df') and knowledge_column in test_data.test_df.columns:
                    knowledge_items = test_data.test_df[knowledge_column].tolist()
            
            # Perform error analysis
            if texts:
                error_df = perform_error_analysis(
                    texts,
                    true_labels,
                    binary_preds,
                    probabilities[:, 1],
                    knowledge_items=knowledge_items
                )
                
                # Save error analysis
                os.makedirs(output_dir, exist_ok=True)
                error_df.to_csv(f"{output_dir}/{model_name}_error_analysis.csv", index=False)
                
                # Add examples to results
                error_examples = error_df.head(10).to_dict('records')
                results["error_examples"] = error_examples
        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, f"{output_dir}/{model_name}_detailed_metrics.json")
    
    logger.info(f"Detailed evaluation completed for {model_name}")
    return results

def compare_models(
    models_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    metrics_to_compare: List[str] = None
) -> pd.DataFrame:
    """
    Create a comparison of multiple models' performance.
    
    Args:
        models_results: Dictionary mapping model names to their evaluation results
        output_dir: Directory to save comparison results
        metrics_to_compare: List of metrics to include in comparison (default: accuracy, precision, recall, f1, roc_auc, pr_auc)
        
    Returns:
        DataFrame with model comparison
    """
    if metrics_to_compare is None:
        metrics_to_compare = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get model names
    model_names = list(models_results.keys())
    
    # Get baseline model for relative comparison
    baseline_model = model_names[0] if model_names else None
    
    # For each metric, get values across models
    for metric in metrics_to_compare:
        metric_values = {}
        
        for model_name in model_names:
            if metric in models_results[model_name]:
                metric_values[model_name] = models_results[model_name][metric]
        
        # Calculate differences if we have a baseline
        if baseline_model and baseline_model in metric_values:
            baseline_value = metric_values[baseline_model]
            
            for model_name in model_names:
                if model_name != baseline_model and model_name in metric_values:
                    model_value = metric_values[model_name]
                    diff = model_value - baseline_value
                    rel_change = (diff / baseline_value) * 100 if baseline_value > 0 else float('inf')
                    
                    comparison_data.append({
                        "Metric": metric,
                        "Baseline": baseline_value,
                        "Model": model_name,
                        "Value": model_value,
                        "Absolute Difference": diff,
                        "Relative Change (%)": rel_change
                    })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Create visualization if we have data
    if not comparison_df.empty:
        # Pivot the dataframe for easier plotting
        pivot_metrics = comparison_df.pivot(index="Metric", columns="Model", values="Value")
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        pivot_metrics.plot(kind="bar", figsize=(12, 8))
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Models")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmap of relative improvements
        if len(model_names) > 1:
            pivot_rel_change = comparison_df.pivot(index="Metric", columns="Model", values="Relative Change (%)")
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_rel_change, annot=True, cmap="RdBu_r", center=0, fmt=".1f")
            plt.title("Relative Improvement Over Baseline (%)")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{output_dir}/relative_improvement.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return comparison_df

def analyze_performance_by_category(
    texts: List[str],
    true_labels: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    output_dir: str,
    text_categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze model performance across different categories of text.
    
    Args:
        texts: List of text inputs
        true_labels: Array of true labels
        model_predictions: Dictionary mapping model names to their predictions
        output_dir: Directory to save analysis results
        text_categories: Optional list of category labels for each text
        
    Returns:
        DataFrame with category performance analysis
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If no categories provided, try to infer some basic ones
    if text_categories is None:
        # Try to identify some categories based on keywords
        categories = []
        
        # Define some COVID-related categories
        category_keywords = {
            "vaccine": ["vaccine", "vaccination", "pfizer", "moderna", "jab", "shot", "vaxx", "mrna"],
            "treatment": ["treatment", "medicine", "drug", "cure", "hydroxychloroquine", "ivermectin", "remedy"],
            "symptoms": ["symptom", "fever", "cough", "loss of taste", "loss of smell", "respiratory"],
            "transmission": ["transmission", "spread", "contagious", "airborne", "droplets", "aerosol"],
            "masks": ["mask", "face covering", "n95", "ppe"],
            "lockdown": ["lockdown", "quarantine", "isolation", "stay at home", "social distancing"],
            "origin": ["origin", "wuhan", "lab leak", "bioweapon", "china", "laboratory"],
            "conspiracy": ["conspiracy", "hoax", "fake", "plandemic", "scamdemic", "bill gates", "5g", "microchip"]
        }
        
        # Categorize each text
        for text in texts:
            text_lower = str(text).lower()
            matched_categories = []
            
            for category, keywords in category_keywords.items():
                for kw in keywords:
                    if kw in text_lower:
                        matched_categories.append(category)
                        break
            
            if matched_categories:
                # If multiple categories match, use the first one
                categories.append(matched_categories[0])
            else:
                categories.append("other")
        
        text_categories = categories
    
    # Create a base dataframe with texts, true labels, and categories
    analysis_df = pd.DataFrame({
        "text": texts,
        "category": text_categories,
        "true_label": true_labels
    })
    
    # Add predictions for each model
    for model_name, preds in model_predictions.items():
        analysis_df[f"{model_name}_pred"] = preds
        analysis_df[f"{model_name}_correct"] = analysis_df[f"{model_name}_pred"] == analysis_df["true_label"]
    
    # Calculate metrics per category
    category_metrics = []
    
    for category in analysis_df["category"].unique():
        category_df = analysis_df[analysis_df["category"] == category]
        
        if len(category_df) < 5:  # Skip categories with too few samples
            continue
        
        category_data = {
            "category": category,
            "sample_count": len(category_df)
        }
        
        # Calculate metrics for each model
        for model_name in model_predictions.keys():
            model_preds = category_df[f"{model_name}_pred"]
            
            # Calculate metrics
            accuracy = accuracy_score(category_df["true_label"], model_preds)
            precision = precision_score(category_df["true_label"], model_preds, zero_division=0)
            recall = recall_score(category_df["true_label"], model_preds, zero_division=0)
            f1 = f1_score(category_df["true_label"], model_preds, zero_division=0)
            
            # Add to category data
            category_data[f"{model_name}_accuracy"] = float(accuracy)
            category_data[f"{model_name}_precision"] = float(precision)
            category_data[f"{model_name}_recall"] = float(recall)
            category_data[f"{model_name}_f1"] = float(f1)
        
        # Calculate cross-model metrics
        if len(model_predictions) > 1:
            model_names = list(model_predictions.keys())
            
            # For each pair of models
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    both_correct = (category_df[f"{model1}_correct"] & category_df[f"{model2}_correct"]).sum()
                    both_incorrect = (~category_df[f"{model1}_correct"] & ~category_df[f"{model2}_correct"]).sum()
                    only_model1_correct = (category_df[f"{model1}_correct"] & ~category_df[f"{model2}_correct"]).sum()
                    only_model2_correct = (~category_df[f"{model1}_correct"] & category_df[f"{model2}_correct"]).sum()
                    
                    category_data[f"{model1}_vs_{model2}_both_correct_pct"] = float(both_correct / len(category_df))
                    category_data[f"{model1}_vs_{model2}_both_incorrect_pct"] = float(both_incorrect / len(category_df))
                    category_data[f"{model1}_vs_{model2}_only_{model1}_correct_pct"] = float(only_model1_correct / len(category_df))
                    category_data[f"{model1}_vs_{model2}_only_{model2}_correct_pct"] = float(only_model2_correct / len(category_df))
        
        category_metrics.append(category_data)
    
    # Convert to dataframe
    category_metrics_df = pd.DataFrame(category_metrics)
    
    # Save category analysis
    category_metrics_df.to_csv(f"{output_dir}/category_performance.csv", index=False)
    
    # Create visualizations if we have more than one category
    if len(category_metrics_df) > 1:
        # For each model, create a bar chart of accuracy by category
        for model_name in model_predictions.keys():
            # Sort categories by accuracy for this model
            plot_df = category_metrics_df.sort_values(f"{model_name}_accuracy", ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.bar(plot_df["category"], plot_df[f"{model_name}_accuracy"], color="skyblue")
            plt.axhline(y=plot_df[f"{model_name}_accuracy"].mean(), color='r', linestyle='-', alpha=0.7, 
                      label=f'Average Accuracy: {plot_df[f"{model_name}_accuracy"].mean():.3f}')
            
            plt.xlabel('Category')
            plt.ylabel('Accuracy')
            plt.title(f'{model_name} Accuracy by Category')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            # Add value labels
            for i, v in enumerate(plot_df[f"{model_name}_accuracy"]):
                plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_category_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # If we have multiple models, create comparison plots
        if len(model_predictions) > 1:
            model_names = list(model_predictions.keys())
            
            # Sort categories by the difference in accuracy between models
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    # Calculate difference
                    category_metrics_df[f"{model1}_vs_{model2}_diff"] = (
                        category_metrics_df[f"{model1}_accuracy"] - category_metrics_df[f"{model2}_accuracy"]
                    )
                    
                    # Sort by absolute difference
                    plot_df = category_metrics_df.sort_values(f"{model1}_vs_{model2}_diff", key=abs, ascending=False)
                    
                    # Limit to top categories for readability
                    if len(plot_df) > 10:
                        plot_df = plot_df.head(10)
                    
                    # Create bar chart
                    plt.figure(figsize=(14, 8))
                    
                    categories = plot_df["category"].tolist()
                    model1_acc = plot_df[f"{model1}_accuracy"].tolist()
                    model2_acc = plot_df[f"{model2}_accuracy"].tolist()
                    
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    rects1 = ax.bar(x - width/2, model1_acc, width, label=model1)
                    rects2 = ax.bar(x + width/2, model2_acc, width, label=model2)
                    
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'Model Comparison by Category: {model1} vs {model2}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories, rotation=45, ha='right')
                    ax.legend()
                    
                    # Add value labels
                    for rect in rects1:
                        height = rect.get_height()
                        ax.annotate(f"{height:.2f}",
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    for rect in rects2:
                        height = rect.get_height()
                        ax.annotate(f"{height:.2f}",
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    plt.ylim(0, 1)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{model1}_vs_{model2}_category_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close()
    
    return category_metrics_df

def summarize_metrics(metrics_dict, model_name=None):
    """
    Create a clean summary of metrics for display
    
    Args:
        metrics_dict: Dictionary with metrics
        model_name: Optional model name to include
        
    Returns:
        Dictionary with simplified metrics
    """
    summary = {}
    
    # Add model name if provided
    if model_name:
        summary["model"] = model_name
    
    # Core metrics that should always be included
    key_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    
    # Try different possible keys for each metric
    for metric in key_metrics:
        # Check for different variations of metric names
        value = metrics_dict.get(metric)
        if value is None:
            value = metrics_dict.get(f"eval_{metric}")
        if value is None:
            for k in metrics_dict.keys():
                if k.lower().endswith(f"_{metric}"):
                    value = metrics_dict[k]
                    break
        
        # Add to summary if found
        if value is not None:
            summary[metric] = value
    
    return summary