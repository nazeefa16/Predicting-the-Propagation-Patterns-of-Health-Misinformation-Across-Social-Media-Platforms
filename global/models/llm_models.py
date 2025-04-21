import os
import torch
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class LLMModel(BaseModel):
    """Base class for LLM inference models"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M", 
        num_labels: int = 2,
        quantization: Optional[str] = "int4",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        batch_size: int = 64,
        **kwargs
    ):
        """
        Initialize LLM model
        
        Args:
            model_name: Name of pretrained LLM
            num_labels: Number of output classes
            quantization: Type of quantization (None, "int4", "int8")
            system_prompt: System prompt for the LLM
            max_new_tokens: Maximum number of tokens to generate
            batch_size: Batch size for inference
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, num_labels)
        self.quantization = quantization
        self.system_prompt = system_prompt or "You are an expert in Health information fact-checking."
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.label_map = {0: "Reliable", 1: "Misinformation"}
        self.inv_label_map = {"Reliable": 0, "Misinformation": 1}
        self.generation_config = kwargs.get("generation_config", None)
        
        # Initialize model and tokenizer
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model {self.model_name}")
            
            # Configure quantization if specified
            quantization_config = None
            if self.quantization == "int4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.quantization == "int8":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load model
            model_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Set appropriate dtype
            if torch.cuda.is_available():
                if kwargs.get("torch_dtype", None) == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif kwargs.get("torch_dtype", None) == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.bfloat16
            
            # Add flash attention if available
            if kwargs.get("use_flash_attention", True) and torch.cuda.is_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set generation config
            if self.generation_config is None:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=kwargs.get("do_sample", False),
                    temperature=kwargs.get("temperature", 0),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 50),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.0)
                )
            
            logger.info(f"Model loaded successfully on {self.model.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
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
        
        # Get prompt message format based on model type
        if "chat" in self.model_name.lower() or "instruct" in self.model_name.lower():
            # Format as chat/instruct prompt
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
    
    def _extract_label(self, response: str) -> int:
        """
        Extract label from model response
        
        Args:
            response: Model generated response
            
        Returns:
            Extracted label (0 or 1)
        """
        try:
            # Try to find 0 or 1 in the response
            match = re.search(r'[01]', response)
            if match:
                return int(match.group(0))
            
            # If no direct number found, check for words
            response_lower = response.lower()
            if "misinformation" in response_lower or "false" in response_lower:
                return 1
            elif "reliable" in response_lower or "factual" in response_lower or "true" in response_lower:
                return 0
            
            # Default if nothing found
            return -1
        except Exception as e:
            logger.error(f"Error extracting label: {e}")
            return -1
    
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
        logger.info("Preparing data for LLM inference")
        
        # Check if data is already split
        if isinstance(data, dict) and all(k in data for k in ["train", "val", "test"]):
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
        logger.info(f"LLM {self.model_name} does not need training (using zero-shot inference)")
        return {}
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with LLM
        
        Args:
            data: Input data (dataframe or list of texts)
            batch_size: Batch size for inference (overrides self.batch_size)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions with LLM")
        
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
        
        # Make predictions in batches
        all_preds = []
        all_confidences = []
        
        logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="LLM inference"):
            batch_texts = texts[i:i+batch_size]
            batch_preds, batch_confidences = self._predict_batch(batch_texts, **kwargs)
            
            all_preds.extend(batch_preds)
            all_confidences.extend(batch_confidences)
        
        # Format outputs
        predictions = np.array(all_preds)
        
        # Create dummy probabilities (LLMs don't typically output calibrated probabilities)
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
    
    def _predict_batch(
        self, 
        texts: List[str],
        **kwargs
    ) -> Tuple[List[int], List[float]]:
        """
        Predict a batch of texts
        
        Args:
            texts: List of texts to classify
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (predictions, confidences)
        """
        predictions = []
        confidences = []
        
        # Process each text
        for text in texts:
            # Create prompt
            prompt = self._create_prompt(text, **kwargs)
            
            # Tokenize input
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            
            # Generate response with torch.no_grad() for efficiency
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    generation_config=self.generation_config
                )
                
                # Extract generated text (excluding prompt)
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract prediction from response
            prediction = self._extract_label(response)
            
            # Use a fixed confidence value (LLMs don't provide calibrated probabilities)
            # A more sophisticated approach would parse the confidence from the response
            confidence = 0.9 if prediction in [0, 1] else 0.5
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
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
        logger.info("Evaluating LLM")
        
        # Get test dataframe
        test_df = test_data["test_df"]
        text_column = test_data.get("text_column", "content")
        label_column = test_data.get("label_column", "label")
        
        # Get predictions
        preds, probs = self.predict(test_data, **kwargs)
        
        # Get true labels
        y_test = test_df[label_column].values
        
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
            "invalid_count": len(preds) - len(valid_indices)
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save model configuration to disk (LLM itself is not saved)
        
        Args:
            output_dir: Directory to save configuration
        """
        logger.info(f"Saving LLM configuration to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config only (not the model itself since it would be too large)
        model_config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "quantization": self.quantization,
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "generation_config": self.generation_config.to_dict() if self.generation_config else None
        }
        
        with open(os.path.join(output_dir, "llm_config.json"), "w") as f:
            import json
            json.dump(model_config, f, indent=2)
    
    def load(self, input_dir: str) -> None:
        """
        Load model configuration from disk and reinitialize LLM
        
        Args:
            input_dir: Directory to load configuration from
        """
        logger.info(f"Loading LLM configuration from {input_dir}")
        
        # Load config
        try:
            import json
            with open(os.path.join(input_dir, "llm_config.json"), "r") as f:
                model_config = json.load(f)
            
            # Update configuration
            self.model_name = model_config.get("model_name", self.model_name)
            self.num_labels = model_config.get("num_labels", self.num_labels)
            self.quantization = model_config.get("quantization", self.quantization)
            self.system_prompt = model_config.get("system_prompt", self.system_prompt)
            self.max_new_tokens = model_config.get("max_new_tokens", self.max_new_tokens)
            self.batch_size = model_config.get("batch_size", self.batch_size)
            
            # Reinitialize model with updated config
            self._initialize_model()
            
            # Set generation config
            if model_config.get("generation_config"):
                self.generation_config = GenerationConfig.from_dict(model_config["generation_config"])
            
        except FileNotFoundError:
            logger.warning("llm_config.json not found, using default configuration")
            # Reinitialize with current config
            self._initialize_model()


class QwenModel(LLMModel):
    """Qwen model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M", 
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            **kwargs
        )


class MixtralModel(LLMModel):
    """Mixtral model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "mistralai/Ministral-8B-Instruct-2410", 
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            **kwargs
        )


class LlamaModel(LLMModel):
    """Llama model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            **kwargs
        )