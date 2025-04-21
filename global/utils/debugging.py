"""
Debugging utilities for health misinformation detection system.
"""

import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

logger = logging.getLogger(__name__)

def inspect_model_outputs(output_dir):
    """
    Inspect and print details about model outputs in a directory
    
    Args:
        output_dir: Directory to inspect
    """
    logger.info(f"Inspecting model outputs in {output_dir}")
    
    # Find all JSON files
    json_files = glob(os.path.join(output_dir, "**/*.json"), recursive=True)
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Summarize each file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            file_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            # Print summary based on file type
            if "detailed_metrics" in file_name or "metrics" in file_name:
                metrics = {k: v for k, v in data.items() 
                          if isinstance(v, (int, float)) and k not in ["confusion_matrix"]}
                logger.info(f"Metrics in {dir_name}/{file_name}: {metrics}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

def debug_comparison_outputs(base_dir, output_dir=None):
    """
    Debug and fix model comparison
    
    Args:
        base_dir: Directory containing model outputs
        output_dir: Directory to save fixed comparison
    """
    if output_dir is None:
        output_dir = os.path.join(base_dir, "comparison_fixed")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Debugging comparison in {base_dir}, output to {output_dir}")
    
    # Find all metrics files
    metrics_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if "metrics.json" in file or "detailed_metrics.json" in file:
                metrics_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(metrics_files)} metrics files")
    
    # Extract metrics
    model_metrics = []
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract model name from directory
            model_dir = os.path.basename(os.path.dirname(file_path))
            
            # Get key metrics
            model_data = {"model": model_dir}
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                value = metrics.get(metric, metrics.get(f"eval_{metric}"))
                if value is not None:
                    model_data[metric] = value
            
            model_metrics.append(model_data)
            logger.info(f"Extracted metrics for {model_dir}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Create comparison dataframe
    if model_metrics:
        comparison_df = pd.DataFrame(model_metrics)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison to {csv_path}")
        
        # Create visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Melt the dataframe for easier plotting
            plot_df = comparison_df.melt(
                id_vars=["model"],
                value_vars=[m for m in comparison_df.columns if m != "model"],
                var_name="metric",
                value_name="value"
            )
            
            # Create bar chart
            ax = sns.barplot(x="model", y="value", hue="metric", data=plot_df)
            
            plt.title("Model Performance Comparison")
            plt.ylabel("Score")
            plt.xlabel("Model")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.grid(axis="y", alpha=0.3)
            plt.legend(title="Metric")
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(output_dir, "model_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
    else:
        logger.warning("No metrics found for comparison")

def debug_dataframe(df, name="DataFrame"):
    """
    Print debug information about a DataFrame
    
    Args:
        df: DataFrame to debug
        name: Name for identification
    """
    logger = logging.getLogger()
    logger.debug(f"--- Debug info for {name} ---")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
    logger.debug(f"Data types: {df.dtypes}")
    logger.debug(f"First 2 rows: \n{df.head(2)}")
    if df.columns.size > 0:
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.debug(f"Column {col} has {null_count} null values")