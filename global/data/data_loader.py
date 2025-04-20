# data/data_loader.py
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_dataset(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load dataset from CSV file
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with dataset
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        
        # Determine file type from extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = 'content',
    label_column: str = 'label',
    **kwargs
) -> pd.DataFrame:
    """
    Preprocess dataset
    
    Args:
        df: DataFrame with dataset
        text_column: Column name containing text data
        label_column: Column name containing labels
        **kwargs: Additional parameters
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        logger.info("Preprocessing dataset")
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = processed_df.dropna(subset=[text_column, label_column])
        
        # Convert label to numeric if it's categorical
        if processed_df[label_column].dtype == 'object':
            label_map = {'Reliable': 0, 'Misinformation': 1}
            processed_df['label_encoded'] = processed_df[label_column].map(label_map)
            logger.info(f"Mapped labels: {label_map}")
        else:
            processed_df['label_encoded'] = processed_df[label_column]
        
        # Basic text preprocessing
        if kwargs.get('clean_text', True):
            processed_df['processed_text'] = processed_df[text_column].str.lower()
            # Remove URLs
            processed_df['processed_text'] = processed_df['processed_text'].str.replace(r'http\S+', '', regex=True)
            # Remove special characters
            processed_df['processed_text'] = processed_df['processed_text'].str.replace(r'[^\w\s]', '', regex=True)
            # Remove extra whitespace
            processed_df['processed_text'] = processed_df['processed_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        else:
            processed_df['processed_text'] = processed_df[text_column]
        
        logger.info(f"Preprocessing complete. Dataset shape: {processed_df.shape}")
        return processed_df
    
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify_column: str = 'label_encoded',
    random_state: int = 42,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: DataFrame with dataset
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (if None, test_size/2)
        stratify_column: Column to use for stratified splitting
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with train, val, and test DataFrames
    """
    try:
        logger.info("Splitting dataset")
        
        if val_size is None:
            # Use half of test_size for validation
            val_size = test_size / 2
            effective_test_size = test_size / 2
        else:
            effective_test_size = test_size
        
        # Check if stratify column exists
        if stratify_column in df.columns:
            stratify = df[stratify_column]
        else:
            stratify = None
            logger.warning(f"Stratify column '{stratify_column}' not found. Using random split.")
        
        # First split: train and temp (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Update stratify for second split
        if stratify is not None:
            temp_stratify = temp_df[stratify_column]
        else:
            temp_stratify = None
        
        # Second split: val and test
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=random_state,
            stratify=temp_stratify
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise