# data/dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

class TextClassificationDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of text inputs
            labels: List of labels
            tokenizer: Optional tokenizer for transformer models
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Precompute encodings if tokenizer is provided
        self.encodings = None
        if tokenizer:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.encodings:
            # Return tokenized inputs for transformers
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        else:
            # Return raw text and label
            return self.texts[idx], self.labels[idx]

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    text_column: str = "processed_text",
    label_column: str = "label_encoded",
    tokenizer = None,
    max_length: int = 512,
    batch_size: int = 128
) -> Dict[str, Tuple[Dataset, DataLoader]]:
    """
    Create datasets and dataloaders for all splits
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe (optional)
        text_column: Column with text data
        label_column: Column with labels
        tokenizer: Optional tokenizer for transformer models
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders
        
    Returns:
        Dictionary with datasets and dataloaders for each split
    """
    # Extract texts and labels
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].tolist()
    
    val_texts = val_df[text_column].tolist()
    val_labels = val_df[label_column].tolist()
    
    test_texts = None
    test_labels = None
    if test_df is not None:
        test_texts = test_df[text_column].tolist()
        test_labels = test_df[label_column].tolist()
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = None
    if test_texts and test_labels:
        test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = None
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Return all data
    result = {
        "train": (train_dataset, train_dataloader),
        "val": (val_dataset, val_dataloader)
    }
    
    if test_dataset:
        result["test"] = (test_dataset, test_dataloader)
    
    return result