# models/base.py
from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Tuple

class BaseModel(ABC):
    """Base class for all misinformation detection models"""
    
    def __init__(self, model_name: str, num_labels: int = 2):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
            num_labels: Number of output classes (default: 2)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1") 
        self.label_map = {0: "Reliable", 1: "Misinformation"}
        self.inv_label_map = {"Reliable": 0, "Misinformation": 1}
        
    @abstractmethod
    def prepare_data(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for the model
        
        Args:
            data_splits: Dictionary of data splits (train, val, test)
            
        Returns:
            Dictionary with prepared data
        """
        pass
    
    @abstractmethod
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any], 
             num_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Data to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, output_dir: str) -> None:
        """
        Save model to disk
        
        Args:
            output_dir: Directory to save model
        """
        pass
    
    @abstractmethod
    def load(self, input_dir: str) -> None:
        """
        Load model from disk
        
        Args:
            input_dir: Directory to load model from
        """
        pass