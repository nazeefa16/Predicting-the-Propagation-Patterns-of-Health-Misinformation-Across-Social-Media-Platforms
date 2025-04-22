# evaluation/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple, Any

def create_output_dirs(output_dir: str) -> Tuple[str, str]:
    """
    Create output directories for figures
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Tuple of (output_dir, figures_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return output_dir, figures_dir

def plot_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    model_name: str,
    class_names: List[str] = None
) -> None:
    """
    Generate and save confusion matrix plot
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        output_dir: Directory to save plot
        model_name: Model name for plot title and filename
        class_names: List of class names (default: ["Factual", "Misinformation"])
    """
    _, figures_dir = create_output_dirs(output_dir)
    
    if class_names is None:
        class_names = ["Factual", "Misinformation"]
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save
    plt.savefig(
        os.path.join(figures_dir, f"{model_name}_confusion_matrix.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()

def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str
) -> float:
    """
    Generate and save ROC curve plot
    
    Args:
        true_labels: Array of true labels
        probabilities: Array of prediction probabilities (for positive class)
        output_dir: Directory to save plot
        model_name: Model name for plot title and filename
        
    Returns:
        AUC score
    """
    _, figures_dir = create_output_dirs(output_dir)
    
    # Calculate ROC curve
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        # Multi-class case, use probabilities for positive class
        probs = probabilities[:, 1]
    else:
        # Binary case, use probabilities as is
        probs = probabilities
        
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    
    # Save
    plt.savefig(
        os.path.join(figures_dir, f"{model_name}_roc_curve.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    
    return roc_auc

def plot_pr_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str
) -> float:
    """
    Generate and save precision-recall curve plot
    
    Args:
        true_labels: Array of true labels
        probabilities: Array of prediction probabilities (for positive class)
        output_dir: Directory to save plot
        model_name: Model name for plot title and filename
        
    Returns:
        AUC score for PR curve
    """
    _, figures_dir = create_output_dirs(output_dir)
    
    # Calculate PR curve
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        # Multi-class case, use probabilities for positive class
        probs = probabilities[:, 1]
    else:
        # Binary case, use probabilities as is
        probs = probabilities
        
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    pr_auc = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    
    # Save
    plt.savefig(
        os.path.join(figures_dir, f"{model_name}_pr_curve.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    
    return pr_auc

def plot_threshold_analysis(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str,
    threshold_steps: int = 10
) -> Dict[str, List[Dict[str, float]]]:
    """
    Generate and save threshold analysis plot
    
    Args:
        true_labels: Array of true labels
        probabilities: Array of prediction probabilities (for positive class)
        output_dir: Directory to save plot
        model_name: Model name for plot title and filename
        threshold_steps: Number of threshold steps to evaluate
        
    Returns:
        Dictionary with threshold analysis results
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    _, figures_dir = create_output_dirs(output_dir)
    
    # Get probabilities for positive class
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        probs = probabilities[:, 1]
    else:
        probs = probabilities
    
    # Generate thresholds
    thresholds = np.linspace(0.1, 0.9, threshold_steps)
    threshold_results = []
    
    # Calculate metrics at each threshold
    for threshold in thresholds:
        threshold_preds = (probs >= threshold).astype(int)
        threshold_metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(true_labels, threshold_preds)),
            "precision": float(precision_score(true_labels, threshold_preds, zero_division=0)),
            "recall": float(recall_score(true_labels, threshold_preds, zero_division=0)),
            "f1": float(f1_score(true_labels, threshold_preds, zero_division=0))
        }
        threshold_results.append(threshold_metrics)
    
    # Create dataframe
    threshold_df = pd.DataFrame(threshold_results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['accuracy'], marker='o', label='Accuracy')
    plt.plot(threshold_df['threshold'], threshold_df['precision'], marker='s', label='Precision')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], marker='^', label='Recall')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], marker='d', label='F1 Score')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs. Threshold - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save
    plt.savefig(
        os.path.join(figures_dir, f"{model_name}_threshold_analysis.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    
    return {"threshold_analysis": threshold_results}

def plot_confidence_distribution(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str
) -> None:
    """
    Generate and save confidence distribution plot
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities
        output_dir: Directory to save plot
        model_name: Model name for plot title and filename
    """
    _, figures_dir = create_output_dirs(output_dir)
    
    # Get confidence for predicted class
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        # Multi-class case, get confidence for predicted class
        confidence = np.max(probabilities, axis=1)
    else:
        # Binary case, use probabilities as is
        confidence = probabilities
    
    # Separate correct and incorrect predictions
    correct_mask = (predictions == true_labels)
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)
    
    if len(correct_conf) > 0:
        plt.hist(correct_conf, bins=bins, alpha=0.5, label='Correct Predictions', color='green')
    
    if len(incorrect_conf) > 0:
        plt.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title(f'Prediction Confidence Distribution - {model_name}')
    plt.legend()
    
    # Save
    plt.savefig(
        os.path.join(figures_dir, f"{model_name}_confidence_distribution.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = None
) -> None:
    """
    Generate and save model comparison plot
    
    Args:
        comparison_df: DataFrame with model comparison data
        output_dir: Directory to save plot
        metrics: List of metrics to include in comparison
    """
    output_dir, _ = create_output_dirs(output_dir)
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    # Melt dataframe for plotting
    plot_df = pd.melt(
        comparison_df, 
        id_vars=['model'], 
        value_vars=available_metrics,
        var_name='metric',
        value_name='value'
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=plot_df, x='model', y='value', hue='metric')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Save
    plt.savefig(
        os.path.join(output_dir, "model_comparison.png"), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()

def generate_all_plots(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Generate all standard plots
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities
        output_dir: Directory to save plots
        model_name: Model name for plot titles and filenames
        
    Returns:
        Dictionary with plot metrics
    """
    metrics = {}
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, output_dir, model_name)
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(true_labels, probabilities, output_dir, model_name)
    metrics["roc_auc"] = roc_auc
    
    # Plot PR curve
    pr_auc = plot_pr_curve(true_labels, probabilities, output_dir, model_name)
    metrics["pr_auc"] = pr_auc
    
    # Plot threshold analysis
    threshold_results = plot_threshold_analysis(true_labels, probabilities, output_dir, model_name)
    metrics.update(threshold_results)
    
    # Plot confidence distribution
    plot_confidence_distribution(true_labels, predictions, probabilities, output_dir, model_name)
    
    return metrics