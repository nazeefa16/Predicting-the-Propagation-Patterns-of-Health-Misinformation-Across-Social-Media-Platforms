# evaluation/metrics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


# evaluation/metrics.py

# Fix the ROC curve calculation to handle both 1D and 2D probability arrays:
def evaluate_model(predictions, true_labels, probabilities, output_dir, model_name):
    """
    Generate evaluation metrics and plots
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        probabilities: Prediction probabilities
        output_dir: Output directory
        model_name: Name of model
        
    Returns:
        Dictionary with metrics
    """
    # Make sure true_labels are numeric for metrics calculations
    if isinstance(true_labels[0], str):
        # Convert string labels to numeric if needed (0 and 1)
        label_map = {"Reliable": 0, "Misinformation": 1}
        true_labels = [label_map.get(label, label) for label in true_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', pos_label=1, zero_division=0
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Reliable', 'Misinformation'],
                yticklabels=['Reliable', 'Misinformation'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve if probabilities are available
    roc_auc = None
    if probabilities is not None and len(probabilities) > 0:
        # Handle both 1D and 2D probability arrays
        if len(probabilities.shape) == 2:
            # 2D array with probabilities for both classes
            # Use the probability of the positive class (index 1)
            prob_positive = probabilities[:, 1]
        else:
            # 1D array already containing probabilities of positive class
            prob_positive = probabilities
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, prob_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
    
    # Plot precision-recall curve if probabilities are available
    pr_auc = None
    if probabilities is not None and len(probabilities) > 0:
        # Handle both 1D and 2D probability arrays (same as above)
        if len(probabilities.shape) == 2:
            prob_positive = probabilities[:, 1]
        else:
            prob_positive = probabilities
            
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            true_labels, prob_positive
        )
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='green', lw=2,
                 label=f'PR curve (area = {pr_auc:.3f})')
        plt.axhline(y=sum(true_labels) / len(true_labels), color='navy', 
                   linestyle='--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
        plt.close()
    
    # Create classification report
    report = classification_report(
        true_labels, predictions, target_names=['Reliable', 'Misinformation'],
        output_dict=True
    )
    
    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report
    }
    
    if roc_auc:
        metrics['roc_auc'] = roc_auc
    if pr_auc:
        metrics['pr_auc'] = pr_auc
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def compare_models(
    model_metrics: Dict[str, Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare multiple models and generate comparison plots
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        output_dir: Output directory for comparison results
        
    Returns:
        Comparison data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for model_name, metrics in model_metrics.items():
        model_data = {
            "model": model_name,
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "pr_auc": metrics.get("pr_auc", 0)
        }
        comparison_data.append(model_data)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Melt dataframe for plotting
    plot_df = pd.melt(
        comparison_df, 
        id_vars=['model'], 
        value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        var_name='metric',
        value_name='value'
    )
    
    # Create plot
    sns.barplot(data=plot_df, x='model', y='value', hue='metric')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Find best model for each metric
    best_models = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmax()
            best_model = comparison_df.loc[best_idx, 'model']
            best_value = comparison_df.loc[best_idx, metric]
            best_models[metric] = (best_model, best_value)
    
    
    # Save best models summary 
    with open(os.path.join(output_dir, "best_models.json"), "w") as f:
        json.dump(best_models, f, indent=2, cls=NumpyEncoder)
    
    return {
        "comparison_df": comparison_df,
        "best_models": best_models
    }