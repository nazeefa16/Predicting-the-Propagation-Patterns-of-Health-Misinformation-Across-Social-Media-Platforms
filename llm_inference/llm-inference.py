from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm  # For better progress tracking

def compute_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_model():
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def process_tweet(tweet, model, tokenizer):
    with torch.inference_mode():
        prompt = f"""Classify the following tweets as either containing Health misinformation (1) or not (0):

Example 1: "Masks don't work and actually make you sicker by reducing oxygen levels."
Classification: 1 (This is misinformation)

Example 2: "The CDC recommends wearing masks in crowded indoor settings to reduce disease transmission."
Classification: 0 (This is factual information)

Example 3: "COVID vaccines contain microchips to track people."
Classification: 1 (This is misinformation)

Example 4: "Studies show that vaccines are effective at preventing severe illness and hospitalization."
Classification: 0 (This is factual information)

Now classify this tweet: "{tweet}"
Classification (0 or 1):"""

        messages = [
            {"role": "system", "content": "You are an expert in Health information fact-checking."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        # Extract just the number from the response
        try:
            # Try to find 0 or 1 in the response
            match = re.search(r'[01]', response)
            if match:
                return int(match.group(0))
            else:
                # Default to 0 if no valid output found
                return -1
        except:
            # Default to 0 if any error occurs
            return -1

def main():
    # Load dataset
    df = pd.read_csv('/home/aledel/repos/Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms/merged_dataset.csv')
    print(f"Loaded dataset with {len(df)} entries")
    print(df.head())
    
    # Convert labels to binary format
    df['label_encoded'] = df['label'].apply(lambda x: 1 if x.lower() == 'misinformation' else 0)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Use a smaller sample size for testing if needed
    test_sample = test_df.head(100)  # Just use 100 examples for testing
    print(f"Using a sample of {len(test_sample)} tweets for evaluation")
    
    # Collect predictions
    predictions = []
    true_labels = []
    
    print("Starting inference...")
    # Use tqdm for progress tracking instead of manual printing
    for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
        tweet = row['content']
        true_label = row['label_encoded']
        
        # Get model prediction
        prediction = process_tweet(tweet, model, tokenizer)
        
        predictions.append(prediction)
        true_labels.append(true_label)
    
    # Calculate metrics
    metrics = compute_metrics(true_labels, predictions)
    
    print("\nModel Evaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Create a confusion matrix
    df_results = pd.DataFrame({
        'True': true_labels,
        'Predicted': predictions
    })
    
    # Create a confusion matrix table
    confusion = pd.crosstab(df_results['True'], df_results['Predicted'], 
                           rownames=['True'], colnames=['Predicted'])
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Save detailed results
    results_df = test_sample.copy()
    results_df['prediction'] = predictions
    results_df['correct'] = results_df['label_encoded'] == results_df['prediction']
    
    llm_results = detailed_model_evaluation(
        model,                 
        rag_test,                  
        output_dir="./output/evaluation",
        model_name="rag_model",
        threshold=0.5,            
        knowledge_column="knowledge_items",   
        text_column="text"        
    )
    # Calculate class-specific metrics
    for label in [0, 1]:
        label_indices = [i for i, x in enumerate(true_labels) if x == label]
        if label_indices:
            class_preds = [predictions[i] for i in label_indices]
            class_true = [true_labels[i] for i in label_indices]
            class_acc = sum(1 for x, y in zip(class_true, class_preds) if x == y) / len(class_true)
            print(f"Class {label} accuracy: {class_acc:.4f} ({sum(1 for x, y in zip(class_true, class_preds) if x == y)}/{len(class_true)})")
    
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nSaved detailed results to model_evaluation_results.csv")

if __name__ == "__main__":
    main()