import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(42)

# Function to run TF-IDF baseline
def run_tfidf_baseline(df):
    print("===> Running TF-IDF + Logistic Regression", flush=True)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['content'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    preds = model_lr.predict(X_test)
    print("TF-IDF Model Accuracy:", round(accuracy_score(y_test, preds) * 100, 3), "%")
    return vectorizer, model_lr

# Dataset class
class PostDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            **{key: torch.tensor(val[idx]) for key, val in self.encodings.items()},
            'labels': torch.tensor(self.labels[idx])
        }

# Training function with optimizations
def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, 
                gradient_accumulation_steps=2, use_amp=True):
    num_training_steps = num_epochs * len(train_loader) // gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, 
                               num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    print("===> Starting training", flush=True)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Mixed precision training
            if use_amp:
                with torch.cuda.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                logits = outputs.logits
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_true.extend(batch['labels'].cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}", flush=True)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model saved (loss: {val_loss:.4f})")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, val_preds, val_true

# Evaluate and create plots
def evaluate_and_visualize(true, preds, save_dir="./outputs"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Classification report
    print(classification_report(true, preds))
    
    # Confusion Matrix
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Misinfo'], yticklabels=['Real', 'Misinfo'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()
    
    # Classification report plot
    report = classification_report(true, preds, 
                                  target_names=["Real", "Misinfo"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    metrics = report_df.loc[["Real", "Misinfo"], ["precision", "recall", "f1-score"]]
    
    metrics.plot(kind='bar', figsize=(8, 6), colormap="Set2")
    plt.title("Precision, Recall & F1-Score per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.savefig(f"{save_dir}/metrics_plot.png")
    plt.close()

# Main function
def main():
    # Set device - use second GPU from previous discussion
    if torch.cuda.is_available():
        device = torch.device("cuda:1")  # Use the second GPU
        print(f"Using GPU: {torch.cuda.get_device_name(1)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    # Load dataset
    print("===> Loading dataset", flush=True)
    df = pd.read_csv("merged_dataset.csv")
    
    # Run TF-IDF baseline
    vectorizer, model_lr = run_tfidf_baseline(df)
    
    # Prepare for transformer model
    print("===> Preparing for transformer fine-tuning", flush=True)
    label_map = {'Reliable': 0, 'Misinformation': 1}
    df['label'] = df['label'].map(label_map).astype(int)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
    
    # Use DistilBERT instead of BERT for efficiency
    model_name = "distilbert-base-uncased"  # Smaller, faster model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # More efficient tokenization
    max_length = 256  # Reduce from 512 if your texts aren't too long
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding='longest',  # More efficient than padding=True
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding='longest',
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Create datasets
    train_dataset = PostDataset(train_encodings, train_labels)
    val_dataset = PostDataset(val_encodings, val_labels)
    
    # Optimized DataLoaders
    batch_size = 16  # Smaller batch size with gradient accumulation
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-5,  # Slightly lower learning rate
        weight_decay=0.01  # Add weight decay for regularization
    )
    
    # Train with optimizations
    model, val_preds, val_true = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        num_epochs=5,  # Increase epochs with early stopping
        device=device,
        gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        use_amp=True  # Use mixed precision
    )
    
    # Evaluate and visualize
    evaluate_and_visualize(val_true, val_preds)
    
    # Save model
    save_directory = "./optimized_misinfo_model"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    main()