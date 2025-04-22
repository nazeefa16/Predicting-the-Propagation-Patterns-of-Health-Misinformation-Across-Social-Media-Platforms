# data/loader.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Optional

class DataLoader:
    """Unified data loader for all model types"""
    
    def __init__(self, file_path: str, text_column: str = "content", label_column: str = "label"):
        """
        Initialize data loader
        
        Args:
            file_path: Path to dataset file
            text_column: Column name containing text data
            label_column: Column name containing labels
        """
        self.file_path = file_path
        self.text_column = text_column
        self.label_column = label_column
        self.label_map = {"Reliable": 0, "Misinformation": 1}
        self.inv_label_map = {0: "Reliable", 1: "Misinformation"}
        
    def load(self) -> pd.DataFrame:
        """Load dataset from file"""
        # Determine file type from extension
        ext = os.path.splitext(self.file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.file_path)
        elif ext == '.tsv':
            df = pd.read_csv(self.file_path, sep='\t')
        elif ext == '.json':
            df = pd.read_json(self.file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Standardize label column
        if self.label_column in df.columns:
            if df[self.label_column].dtype == object:
                df["label_encoded"] = df[self.label_column].map(self.label_map)
            else:
                df["label_encoded"] = df[self.label_column]
        else:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset")
            
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataset with basic text cleaning"""
        # Handle missing values
        df = df.dropna(subset=[self.text_column, self.label_column])
        
        # Basic text preprocessing
        df["processed_text"] = df[self.text_column].str.lower()
        # Remove URLs
        df["processed_text"] = df["processed_text"].str.replace(r'http\S+', '', regex=True)
        # Remove special characters
        df["processed_text"] = df["processed_text"].str.replace(r'[^\w\s]', '', regex=True)
        # Remove extra whitespace
        df["processed_text"] = df["processed_text"].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        return df
    
    def split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        val_size: Optional[float] = None, 
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """Split dataset into train, val, and test sets"""
        if val_size is None:
            val_size = test_size / 2
            test_size = test_size / 2
            
        # Use label_encoded for stratification
        stratify = df["label_encoded"] if "label_encoded" in df.columns else None
        
        # First split: train and temp (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: val and test
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=random_state,
            stratify=temp_df["label_encoded"] if stratify is not None else None
        )
        
        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }
    
    def prepare_data(
        self, 
        test_size: float = 0.2, 
        val_size: Optional[float] = None, 
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """Main method to load, preprocess, and split data"""
        df = self.load()
        df = self.preprocess(df)
        splits = self.split(df, test_size, val_size, random_state)
        return splits