# models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all misinformation detection models.
    All models (traditional ML, transformers, LLMs) should implement this interface.
    """
    
    def __init__(self, model_name: str, num_labels: int = 2):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
            num_labels: Number of output classes 
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def prepare_data(self, data: Any, **kwargs) -> Any:
        """
        Prepare data for the model
        
        Args:
            data: Input data (DataFrame or Dataset)
            **kwargs: Additional keyword arguments
            
        Returns:
            Prepared data ready for training/inference
        """
        pass
    
    @abstractmethod
    def train(self, train_data: Any, val_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Training data
            val_data: Validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Data to predict on
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            test_data: Test data
            **kwargs: Additional evaluation parameters
            
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
    
    def __str__(self):
        """String representation of the model"""
        return f"{self.__class__.__name__}(model_name={self.model_name}, num_labels={self.num_labels})"