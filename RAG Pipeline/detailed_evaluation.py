import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Any, Dict, List, Union, Tuple
import seaborn as sns
import os
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
from transformers import Trainer
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
import json

logger = logging.getLogger(__name__)


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
    
    if isinstance(obj, ( np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python_type(i) for i in obj]
    
    # Return other types as is
    return obj

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        return numpy_to_python_type(obj)

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file with proper handling of numpy types.
    
    Args:
        data: Data to save (may contain numpy types)
        filepath: Path to save JSON file
        indent: JSON indentation level
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=indent)

def detailed_model_evaluation(
    model,
    test_dataset,
    output_dir: str,
    model_name: str = "model",
    threshold: float = 0.5,
    knowledge_column: Optional[str] = None,
    text_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform detailed evaluation of a classification model.
    
    Args:
        model: The trained model to evaluate
        test_dataset: HuggingFace dataset for testing
        output_dir: Directory to save evaluation results
        model_name: Name of the model for saving results
        threshold: Classification threshold 
        knowledge_column: Column name for retrieved knowledge (for error analysis)
        text_column: Column name for original text (for error analysis)
    
    Returns:
        Dictionary with detailed evaluation metrics
    """
    logger.info(f"Performing detailed evaluation of {model_name}")
    
    # Create trainer for inference
    trainer = Trainer(model=model)
    
    # Get raw predictions
    raw_preds = trainer.predict(test_dataset)
    logits = raw_preds.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    preds = (probs[:, 1] >= threshold).astype(int)  # Binary classification
    
    # Get true labels
    true_labels = raw_preds.label_ids
    
    # Basic metrics
    results = {}
    results["accuracy"] = accuracy_score(true_labels, preds)
    results["precision"] = precision_score(true_labels, preds, zero_division=0)
    results["recall"] = recall_score(true_labels, preds, zero_division=0)
    results["f1"] = f1_score(true_labels, preds, zero_division=0)
    
    # Confidence metrics
    correct_mask = (preds == true_labels)
    if correct_mask.sum() > 0:
        results["mean_confidence_correct"] = np.mean(probs[np.arange(len(preds)), true_labels][correct_mask])
    else:
        results["mean_confidence_correct"] = 0.0
    
    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        results["mean_confidence_incorrect"] = np.mean(probs[np.arange(len(preds)), true_labels][incorrect_mask])
    else:
        results["mean_confidence_incorrect"] = 0.0
    
    # ROC and PR curves
    fpr, tpr, roc_thresholds = roc_curve(true_labels, probs[:, 1])
    results["roc_auc"] = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(true_labels, probs[:, 1])
    results["pr_auc"] = auc(recall, precision)
    
    # Calibration
    results["brier_score"] = brier_score_loss(true_labels, probs[:, 1])
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    results["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    results["tn"], results["fp"], results["fn"], results["tp"] = int(tn), int(fp), int(fn), int(tp)
    
    # Threshold analysis
    threshold_metrics = perform_threshold_analysis(true_labels, probs[:, 1])
    results["threshold_analysis"] = threshold_metrics
    
    # Classification report (per class metrics)
    class_report = classification_report(true_labels, preds, output_dict=True)
    # Convert all numpy types in the report to native Python types
    results["classification_report"] = numpy_to_python_type(class_report)
    
    # Generate visualizations
    generate_visualizations(
        true_labels, 
        preds, 
        probs, 
        output_dir, 
        model_name, 
        fpr, tpr, 
        precision, recall
    )
    
    # Error analysis if columns are provided
    error_examples = None
    if hasattr(test_dataset, "features") and text_column is not None:
        try:
            # Get the dataset as pandas
            text_dataset = test_dataset.remove_columns(['input_ids', 'attention_mask', 'labels'])
            text_dataset = text_dataset.to_pandas() if hasattr(test_dataset, "to_pandas") else pd.DataFrame()

            # Add text column if it exists in the original dataset
            if hasattr(test_dataset, "orig_dataset") and text_column in test_dataset.orig_dataset.column_names:
                text_dataset[text_column] = test_dataset.orig_dataset[text_column]
            
            # Perform error analysis if we have the text column
            if text_column in text_dataset.columns:
                error_df = perform_error_analysis(
                    text_dataset[text_column], 
                    true_labels, 
                    preds, 
                    probs[:, 1],
                    knowledge_items=text_dataset.get(knowledge_column) if knowledge_column and knowledge_column in text_dataset.columns else None
                )
                # Save error analysis
                error_df.to_csv(f"{output_dir}/{model_name}_error_analysis.csv", index=False)
                error_examples = error_df.head(10).to_dict('records')
                results["error_examples"] = error_examples
        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
            # Continue with evaluation even if error analysis fails
    
    # Save all results using our custom JSON saver
    results_file = f"{output_dir}/{model_name}_detailed_metrics.json"
    save_json(results, results_file)
    
    logger.info(f"Detailed evaluation results saved to {results_file}")
    
    return results

def perform_threshold_analysis(true_labels, probabilities, steps=10) -> List[Dict[str, float]]:
    """
    Analyze model performance across different threshold values.
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

def generate_visualizations(
    true_labels, 
    predictions, 
    probabilities, 
    output_dir, 
    model_name, 
    fpr, tpr, 
    precision, 
    recall
):
    """
    Generate and save visualization plots for model evaluation.
    """
    # Set up figure directory
    fig_dir = f"{output_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{fig_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
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

def perform_error_analysis(
    texts, 
    true_labels, 
    predictions, 
    probabilities, 
    knowledge_items=None, 
    max_samples=100
) -> pd.DataFrame:
    """
    Perform detailed error analysis, with a focus on understanding when and why the model makes mistakes.
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

def compare_models(baseline_results, rag_results, output_dir):
    """
    Create detailed comparison between baseline and RAG models.
    """
    # Create comparison dataframe
    comparison_data = []
    
    # Basic metrics comparison
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        baseline_value = baseline_results.get(metric)
        rag_value = rag_results.get(metric)
        
        if baseline_value is not None and rag_value is not None:
            diff = rag_value - baseline_value
            rel_change = (diff / baseline_value) * 100 if baseline_value > 0 else float('inf')
            
            comparison_data.append({
                "Metric": metric,
                "Baseline": baseline_value,
                "RAG": rag_value,
                "Absolute Difference": diff,
                "Relative Change (%)": rel_change
            })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    metrics = comparison_df["Metric"].tolist()
    baseline_values = comparison_df["Baseline"].tolist()
    rag_values = comparison_df["RAG"].tolist()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
    rag_bars = ax.bar(x + width/2, rag_values, width, label='RAG', color='lightgreen')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)
    
    add_labels(baseline_bars)
    add_labels(rag_bars)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return comparison dataframe
    return comparison_df

def analyze_performance_by_category(
    texts, 
    true_labels, 
    baseline_preds, 
    rag_preds, 
    output_dir,
    text_categories=None
):
    """
    Analyze model performance across different categories of text.
    """
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
    
    # Create dataframe with all data
    analysis_df = pd.DataFrame({
        "text": texts,
        "category": text_categories,
        "true_label": true_labels,
        "baseline_pred": baseline_preds,
        "rag_pred": rag_preds,
        "baseline_correct": baseline_preds == true_labels,
        "rag_correct": rag_preds == true_labels,
        "both_correct": (baseline_preds == true_labels) & (rag_preds == true_labels),
        "both_incorrect": (baseline_preds != true_labels) & (rag_preds != true_labels),
        "only_baseline_correct": (baseline_preds == true_labels) & (rag_preds != true_labels),
        "only_rag_correct": (baseline_preds != true_labels) & (rag_preds == true_labels)
    })
    
    # Calculate metrics per category
    category_metrics = []
    for category in analysis_df["category"].unique():
        category_df = analysis_df[analysis_df["category"] == category]
        
        if len(category_df) < 5:  # Skip categories with too few samples
            continue
            
        baseline_acc = (category_df["baseline_correct"].sum() / len(category_df))
        rag_acc = (category_df["rag_correct"].sum() / len(category_df))
        
        baseline_f1 = f1_score(
            category_df["true_label"], 
            category_df["baseline_pred"], 
            zero_division=0
        )
        
        rag_f1 = f1_score(
            category_df["true_label"], 
            category_df["rag_pred"], 
            zero_division=0
        )
        
        category_metrics.append({
            "category": category,
            "sample_count": len(category_df),
            "baseline_accuracy": float(baseline_acc),
            "rag_accuracy": float(rag_acc),
            "accuracy_diff": float(rag_acc - baseline_acc),
            "baseline_f1": float(baseline_f1),
            "rag_f1": float(rag_f1),
            "f1_diff": float(rag_f1 - baseline_f1),
            "both_correct_pct": float(category_df["both_correct"].sum() / len(category_df)),
            "both_incorrect_pct": float(category_df["both_incorrect"].sum() / len(category_df)),
            "only_baseline_correct_pct": float(category_df["only_baseline_correct"].sum() / len(category_df)),
            "only_rag_correct_pct": float(category_df["only_rag_correct"].sum() / len(category_df))
        })
    
    # Convert to dataframe and sort by difference in performance
    category_metrics_df = pd.DataFrame(category_metrics)
    category_metrics_df = category_metrics_df.sort_values("accuracy_diff", ascending=False)
    
    # Save category analysis
    category_metrics_df.to_csv(f"{output_dir}/category_performance.csv", index=False)
    
    # Create visualizations
    plt.figure(figsize=(14, 8))
    
    # Sort categories by absolute difference for better visualization
    viz_df = category_metrics_df.sort_values("accuracy_diff", key=abs, ascending=False).head(10)
    
    categories = viz_df["category"].tolist()
    baseline_acc = viz_df["baseline_accuracy"].tolist()
    rag_acc = viz_df["rag_accuracy"].tolist()
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    baseline_bars = ax.bar(x - width/2, baseline_acc, width, label='Baseline', color='skyblue')
    rag_bars = ax.bar(x + width/2, rag_acc, width, label='RAG', color='lightgreen')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance by Content Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(baseline_acc):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(rag_acc):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_metrics_df

# Usage instructions
if __name__ == "__main__":
    print("This module provides functions for detailed model evaluation.")
    print("Import and use the functions in your main script.")