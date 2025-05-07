# models/vllm_llm.py
import os
import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# Import vLLM components
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")

from .base import BaseModel

logger = logging.getLogger(__name__)

class VLLMModel(BaseModel):
    """LLM model using vLLM for accelerated inference"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M", 
        num_labels: int = 2,
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 50,
        system_prompt: Optional[str] = None,
        max_model_len: int = 2048, 
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 64,
        **kwargs
    ):
        """
        Initialize vLLM model
        
        Args:
            model_name: Name of pretrained LLM
            num_labels: Number of output classes
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_new_tokens: Maximum new tokens to generate
            system_prompt: System prompt for the model
            batch_size: Batch size for inference
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, num_labels)
        
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required but not installed. Install with: pip install vllm")
        
        self.tensor_parallel_size = tensor_parallel_size
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.system_prompt = system_prompt or "You are an expert in Health information fact-checking."
        self.label_map = {0: "Reliable", 1: "Misinformation"}
        self.inv_label_map = {"Reliable": 0, "Misinformation": 1}
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize vLLM
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize vLLM model"""
        try:
            logger.info(f"Loading vLLM model {self.model_name}")
            
            # Get GPU-specific settings
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            actual_tp_size = min(self.tensor_parallel_size, gpu_count)
            if actual_tp_size < self.tensor_parallel_size:
                logger.warning(f"Requested tensor_parallel_size={self.tensor_parallel_size} but only {gpu_count} GPUs available")
                self.tensor_parallel_size = actual_tp_size
            
            # Configure vLLM parameters
            vllm_kwargs = {
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "trust_remote_code": True,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_num_seqs": self.batch_size,  # Maximum number of sequences in batch
            }
            
            # Add GPU memory optimization if specified
            if kwargs.get("gpu_memory_utilization", None):
                vllm_kwargs["gpu_memory_utilization"] = kwargs.get("gpu_memory_utilization")
            
            # Add quantization if specified
            
                vllm_kwargs["quantization"] = "int8"
            
            # Create vLLM instance
            self.model = LLM(**vllm_kwargs)
            
            # Create sampling parameters for generation
            self.sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0)
            )
            
            logger.info(f"vLLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing vLLM model: {e}")
            raise
    
    def _create_prompt(self, text: str, **kwargs) -> str:
        """
        Create prompt for LLM
        
        Args:
            text: Input text
            **kwargs: Additional prompt parameters
            
        Returns:
            Formatted prompt
        """
        # Default prompt
        prompt = f"""Classify the following tweets as either containing Health misinformation (1) or not (0):

Example 1: "Masks don't work and actually make you sicker by reducing oxygen levels."
Classification: 1 (This is misinformation)

Example 2: "The CDC recommends wearing masks in crowded indoor settings to reduce disease transmission."
Classification: 0 (This is factual information)

Example 3: "COVID vaccines contain microchips to track people."
Classification: 1 (This is misinformation)

Example 4: "Studies show that vaccines are effective at preventing severe illness and hospitalization."
Classification: 0 (This is factual information)

Now classify this tweet: "{text}"
Classification (0 or 1):"""

        # Use custom prompt if provided
        custom_prompt = kwargs.get("prompt", None)
        if custom_prompt:
            prompt = custom_prompt.format(text=text)
        
        # Format for chat models
        if "chat" in self.model_name.lower() or "instruct" in self.model_name.lower():
            # Format as chat prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Check if model uses specific chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Basic chat format fallback
                formatted_prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            # Basic text prompt format for non-chat models
            formatted_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        return formatted_prompt
    
    def _extract_label(self, text: str) -> int:
        """
        Extract label from generated text with improved parsing
        
        Args:
            text: Generated text from LLM
            
        Returns:
            Extracted label (0 or 1, or -1 if unable to extract)
        """
        # Look for classification at the end of analysis
        if "classification: 0" in text.lower() or "classification (0)" in text.lower():
            return 0
        elif "classification: 1" in text.lower() or "classification (1)" in text.lower():
            return 1
        
        # Also look for clear statements about misinformation
        if "this is factual" in text.lower() or "not misinformation" in text.lower():
            return 0
        elif "this is misinformation" in text.lower() or "contains misinformation" in text.lower():
            return 1
        
        # Default fallback extraction
        try:
            # Try to find just "0" or "1" at the end
            lines = text.strip().split('\n')
            for line in reversed(lines):  # Check from end
                line = line.strip()
                if line == "0" or line == "1":
                    return int(line)
                # Check for 0 or 1 in parentheses at end
                if line.endswith("(0)") or line.endswith("(1)"):
                    return int(line[-2])
        except:
            logger.warning("Error extracting label")
        return 0
            
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, Dict], 
        text_column: str = "content",
        label_column: str = "label", 
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare data for inference
        
        Args:
            data: DataFrame with text and labels or already split data dict
            text_column: Column name containing text data
            label_column: Column name containing labels
            test_size: Proportion of data to use for testing (if splitting)
            **kwargs: Additional data preparation parameters
            
        Returns:
            Dictionary with prepared datasets
        """
        logger.info("Preparing data for vLLM inference")
        
        # Check if data is already split
        if isinstance(data, dict) and all(k in data for k in ["train", "val", "test"]):
            logger.info("Data is already split, assigning values")
            train_df = data["train"]
            val_df = data["val"]
            test_df = data.get("test")
        else:
            # Convert labels if they are strings
            if isinstance(data[label_column].iloc[0], str):
                data = data.copy()
                data[label_column] = data[label_column].map(self.inv_label_map).astype(int)
            
            # Split data
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(
                data, 
                test_size=test_size * 2,  # Split into val and test
                random_state=42,
                stratify=data[label_column] if kwargs.get("stratify", True) else None
            )
            
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=42,
                stratify=temp_df[label_column] if kwargs.get("stratify", True) else None
            )
        
        # Subsample if needed (LLMs can be slow/expensive to run on large datasets)
        max_samples = kwargs.get("max_samples", None)
        if max_samples:
            if len(train_df) > max_samples:
                train_df = train_df.sample(max_samples, random_state=42)
            
            val_max = max(100, max_samples // 10)  # at least 100 samples
            if len(val_df) > val_max:
                val_df = val_df.sample(val_max, random_state=42)
            
            test_max = max(100, max_samples // 10)  # at least 100 samples
            if test_df is not None and len(test_df) > test_max:
                test_df = test_df.sample(test_max, random_state=42)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, "
                   f"Test: {len(test_df) if test_df is not None else 'N/A'}")
        
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "text_column": text_column,
            "label_column": label_column
        }
    
    def train(
        self, 
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LLMs are not trained (zero-shot inference only).
        This method is implemented for API compatibility.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            **kwargs: Additional training parameters
            
        Returns:
            Empty dictionary (LLM is not trained)
        """
        logger.info(f"vLLM {self.model_name} does not need training (using zero-shot inference)")
        return {}
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with vLLM
        
        Args:
            data: Input data (dataframe or list of texts)
            batch_size: Batch size for inference (overrides self.batch_size)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions with vLLM")
        
        # Setup batch size
        batch_size = batch_size or self.batch_size
        
        # Setup data for prediction
        if isinstance(data, dict):
            # Using data dictionary with dataframe
            df = None
            if "test_df" in data:
                df = data["test_df"]
            elif "val_df" in data:
                df = data["val_df"]
            
            if df is None:
                raise ValueError("No dataframe found in data dictionary")
            
            text_column = data.get("text_column", "content")
            texts = df[text_column].tolist()
        elif isinstance(data, pd.DataFrame):
            texts = data[kwargs.get("text_column", "content")].tolist()
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Make predictions
        all_preds = []
        all_confidences = []
        
        logger.info(f"Processing {len(texts)} texts")
        
        # Process all texts in batches
        prompts = [self._create_prompt(text) for text in texts]
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="vLLM inference"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate outputs with vLLM
            outputs = self.model.generate(batch_prompts, self.sampling_params)
            
            # Process outputs
            for output in outputs:
                generated_text = output.outputs[0].text
                prediction = self._extract_label(generated_text)
                
                # Use a fixed confidence value for simplicity
                confidence = 0.9 if prediction in [0, 1] else 0.5
                
                all_preds.append(prediction)
                all_confidences.append(confidence)
        
        # Format outputs
        predictions = np.array(all_preds)
        
        # Create probabilities
        probabilities = np.zeros((len(predictions), self.num_labels))
        for i, (pred, conf) in enumerate(zip(predictions, all_confidences)):
            if pred >= 0:  # Valid prediction
                probabilities[i, pred] = conf
                if self.num_labels > 1:  # Handle binary case
                    other_class = 1 - pred if pred in [0, 1] else 0
                    probabilities[i, other_class] = 1.0 - conf
            else:  # Invalid prediction
                # Assign equal probabilities if we couldn't extract a valid label
                probabilities[i, :] = 1.0 / self.num_labels
        
        return predictions, probabilities
    
    def evaluate(
        self, 
        test_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            test_data: Test data dictionary
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating vLLM model")
        
        # Get test dataframe
        test_df = test_data["test_df"]
        text_column = test_data.get("text_column", "content")
        label_column = test_data.get("label_column", "label")
        
        # Get predictions
        preds, probs = self.predict(test_data, **kwargs)
        
        # Get true labels
        y_test = test_df[label_column].values

        if y_test.size > 0 and isinstance(y_test[0], str):
            y_test = np.array([self.inv_label_map.get(label, -1) for label in y_test])
    
        # Remove invalid predictions (-1) for evaluation
        valid_indices = [i for i, p in enumerate(preds) if p in [0, 1]]
        if len(valid_indices) < len(preds):
            logger.warning(f"Removed {len(preds) - len(valid_indices)} invalid predictions")
            
            if not valid_indices:
                logger.error("No valid predictions found")
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "predictions": preds,
                    "probabilities": probs,
                    "true_labels": y_test,
                    "invalid_count": len(preds) - len(valid_indices)
                }
            
            valid_preds = preds[valid_indices]
            valid_y_test = y_test[valid_indices]
        else:
            valid_preds = preds
            valid_y_test = y_test
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(valid_y_test, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_y_test, valid_preds, average='binary', zero_division=0
        )
        
        logger.info(f"Test metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": preds,
            "probabilities": probs,
            "true_labels": y_test,
            "texts": test_df[text_column].tolist(),
            "invalid_count": len(preds) - len(valid_indices)
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save model configuration to disk (model itself is not saved)
        
        Args:
            output_dir: Directory to save configuration
        """
        logger.info(f"Saving vLLM configuration to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config only (not the model itself since it would be too large)
        model_config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": self.system_prompt,
            "batch_size": self.batch_size
        }
        
        with open(os.path.join(output_dir, "vllm_config.json"), "w") as f:
            import json
            json.dump(model_config, f, indent=2)
    
    def load(self, input_dir: str) -> None:
        """
        Load model configuration from disk and reinitialize vLLM
        
        Args:
            input_dir: Directory to load configuration from
        """
        logger.info(f"Loading vLLM configuration from {input_dir}")
        
        # Load config
        try:
            import json
            with open(os.path.join(input_dir, "vllm_config.json"), "r") as f:
                model_config = json.load(f)
            
            # Update configuration
            self.model_name = model_config.get("model_name", self.model_name)
            self.num_labels = model_config.get("num_labels", self.num_labels)
            self.tensor_parallel_size = model_config.get("tensor_parallel_size", self.tensor_parallel_size)
            self.max_new_tokens = model_config.get("max_new_tokens", self.max_new_tokens)
            self.system_prompt = model_config.get("system_prompt", self.system_prompt)
            self.batch_size = model_config.get("batch_size", self.batch_size)
            
            # Reinitialize model with updated config
            self._initialize_model()
            
        except FileNotFoundError:
            logger.warning("vllm_config.json not found, using default configuration")
            # Reinitialize with current config
            self._initialize_model()