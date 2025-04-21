#!/usr/bin/env python3
"""
Simple script to compare models by reading their evaluation metrics.
"""
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)

def find_metrics_files(base_dir):
    """Find all metrics files in subdirectories"""
    metrics_files = {}
    
    # Look for model directories
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if os.path.isdir(model_path):
            # Look for metrics files
            json_files = glob.glob(os.path.join(model_path, "*.json"))
            for json_file in json_files:
                if os.path.basename(json_file) != "model_config.json":
                    metrics_files[model_dir] = json_file
                    break
    
    return metrics_files

def extract_metrics(metrics_files):
    """Extract key metrics from files"""
    model_metrics = {}
    
    for model_name, file_path in metrics_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics (handle different formats)
            metrics = {}
            
            # Check for direct metrics
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                if key in data:
                    metrics[key] = data[key]
                elif f'eval_{key}' in data:
                    metrics[key] = data[f'eval_{key}']
            
            # If metrics are nested in 'val_metrics'
            if 'val_metrics' in data and isinstance(data['val_metrics'], dict):
                for key in ['accuracy', 'precision', 'recall', 'f1']:
                    if key in data['val_metrics']:
                        metrics[key] = data['val_metrics'][key]
            
            if metrics:
                model_metrics[model_name] = metrics
                
        except Exception as e:
            print(f"Error loading metrics for {model_name}: {e}")
    
    return model_metrics

def create_comparison_chart(model_metrics, output_dir):
    """Create comparison chart and CSV"""
    if not model_metrics:
        print("No metrics found")
        return
    
    # Create DataFrame
    rows = []
    for model, metrics in model_metrics.items():
        row = {'model': model}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Comparison saved to {csv_path}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Reshape data for plotting
    plot_df = pd.melt(
        df, 
        id_vars=['model'], 
        value_vars=[c for c in df.columns if c != 'model'],
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
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    
    # Print summary
    print("\nModel Comparison Summary:")
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        if metric in df.columns:
            best_model = df.loc[df[metric].idxmax()]
            print(f"Best {metric}: {best_model['model']} ({best_model[metric]:.4f})")

def find_model_results(input_dir):
    """Find all model evaluation result files in the input directory"""
    model_results = {}
    
    # Check if input_dir exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return model_results
    
    # List all subdirectories (one for each model)
    for model_dir in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_dir)
        
        # Skip non-directories and comparison directories
        if not os.path.isdir(model_path) or model_dir == "comparison":
            continue
            
        logger.info(f"Checking model directory: {model_path}")
        
        # Look for metrics files in this directory and all subdirectories
        detailed_metrics_file = None
        for root, _, files in os.walk(model_path):
            for file in files:
                if "detailed_metrics.json" in file or "metrics.json" in file:
                    detailed_metrics_file = os.path.join(root, file)
                    break
            if detailed_metrics_file:
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model metrics")
    parser.add_argument("--input_dir", required=True, help="Directory containing model outputs")
    parser.add_argument("--output_dir", default=None, help="Directory to save comparison (default: input_dir)")
    
    args = parser.parse_args()
    output_dir = "/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms/comparison"
    
    # Find metrics files
    metrics_files = find_metrics_files(args.input_dir)
    print(f"Found metrics for {len(metrics_files)} models: {list(metrics_files.keys())}")
    
    # Extract metrics
    model_metrics = extract_metrics(metrics_files)
    
    # Create comparison
    create_comparison_chart(model_metrics, output_dir)

if __name__ == "__main__":
    main()