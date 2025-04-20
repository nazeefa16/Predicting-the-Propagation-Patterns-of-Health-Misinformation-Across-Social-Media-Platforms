
#!/usr/bin/env python3
"""
Script to compare all model results from different directories.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import logging
import sys

# Add the project root to the path to import metrics module
import sys
sys.path.append("/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms")

from evaluation.metrics import compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare all model results")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Base directory containing model output folders")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save comparison results (default: input_dir)")
    return parser.parse_args()

def find_model_results(input_dir):
    """Find all model evaluation result files in the input directory"""
    model_results = {}
    
    # List all subdirectories (one for each model)
    for model_dir in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_dir)
        
        if os.path.isdir(model_path):
            # Look for detailed metrics file
            detailed_metrics_file = None
            
            # Try specific naming pattern
            for pattern in [
                "*detailed_metrics.json",
                "*metrics.json",
                "*evaluation.json",
                "*.json"  # Last resort - any JSON file
            ]:
                matches = glob(os.path.join(model_path, pattern))
                if matches:
                    detailed_metrics_file = matches[0]
                    break
            
            if detailed_metrics_file:
                logger.info(f"Found metrics file for {model_dir}: {detailed_metrics_file}")
                
                try:
                    with open(detailed_metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    model_results[model_dir] = metrics
                except Exception as e:
                    logger.error(f"Error loading metrics for {model_dir}: {e}")
            else:
                logger.warning(f"No metrics file found for {model_dir}")
    
    return model_results

def main():
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Find all model results
    logger.info(f"Looking for model results in {args.input_dir}")
    model_results = find_model_results(args.input_dir)
    
    if not model_results:
        logger.error(f"No model results found in {args.input_dir}")
        return
    
    logger.info(f"Found results for {len(model_results)} models: {list(model_results.keys())}")
    
    # Create comparison directory
    comparison_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare models using your existing function
    comparison_df = compare_models(
        models_results=model_results,
        output_dir=comparison_dir
    )
    
    # Create a summary table with key metrics
    summary_data = []
    
    metrics_to_include = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    for model_name, metrics in model_results.items():
        model_data = {"model": model_name}
        
        for metric in metrics_to_include:
            # Handle different possible metric names
            if metric in metrics:
                model_data[metric] = metrics[metric]
            elif f"eval_{metric}" in metrics:
                model_data[metric] = metrics[f"eval_{metric}"]
            else:
                model_data[metric] = None
                
        summary_data.append(model_data)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(comparison_dir, "model_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe for easier plotting
    plot_df = summary_df.melt(
        id_vars=["model"],
        value_vars=[m for m in metrics_to_include if m in summary_df.columns],
        var_name="metric",
        value_name="value"
    )
    
    # Create grouped bar chart
    sns.barplot(x="model", y="value", hue="metric", data=plot_df)
    
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Metric")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(comparison_dir, "model_summary.png"), dpi=300, bbox_inches='tight')
    
    logger.info(f"Comparison results saved to {comparison_dir}")
    logger.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()