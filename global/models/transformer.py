# models/transformer.py
import os
import gc
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Union, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler

from .base import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for transformer model"""
    model_name: str = "distilroberta-base"
    max_length: int = 512
    num_labels: int = 2
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    early_stopping_patience: int = 3
    output_dir: str = "./outputs"
    streaming: bool = False
    streaming_batch_size: int = 1000
    cache_dir: Optional[str] = None
    use_auth_token: bool = False

def ensure_type(value, target_type):
    """
    Ensure a value is of the target type
    
    Args:
        value: Value to convert
        target_type: Target type (int, float, str, etc.)
        
    Returns:
        Value converted to target type
    """
    if isinstance(value, target_type):
        return value
    
    try:
        return target_type(value)
    except (ValueError, TypeError):
        # Return default values based on type
        if target_type == int:
            return 0
        elif target_type == float:
            return 0.0
        elif target_type == str:
            return ""
        elif target_type == bool:
            return False
        elif target_type == list:
            return []
        elif target_type == dict:
            return {}
        else:
            return None

def compute_metrics(pred):
    """Compute metrics for HuggingFace Trainer"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class TextDataset(Dataset):
    """Dataset for text classification with transformers"""
    
    def __init__(self, encodings: Dict, labels: List[int]):
        """
        Initialize dataset
        
        Args:
            encodings: Token encodings from tokenizer
            labels: List of class labels
        """
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class StreamingTextDataset(IterableDataset):
    """Streaming dataset for text classification with transformers"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int, batch_size: int = 1000):
        """
        Initialize streaming dataset
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size for streaming
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.labels)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield batches of encoded examples"""
        for i in range(0, len(self.texts), self.batch_size):
            end_idx = min(i + self.batch_size, len(self.texts))
            batch_texts = self.texts[i:end_idx]
            batch_labels = self.labels[i:end_idx]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts, 
                truncation=True, 
                padding=True,
                max_length=self.max_length
            )
            
            # Convert to tensors and yield
            for j in range(len(batch_labels)):
                item = {key: torch.tensor(val[j]) for key, val in encodings.items()}
                item['labels'] = torch.tensor(batch_labels[j])
                yield item

class TransformerModel(BaseModel):
    """Optimized transformer model for text classification"""
    
    def __init__(self, config: Optional[Union[ModelConfig, Dict[str, Any]]] = None):
        """
        Initialize transformer model
        
        Args:
            config: Configuration for the model (ModelConfig or dict)
        """
        # Initialize configuration
        if config is None:
            self.config = ModelConfig()
        elif isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config
            
        # Initialize base model
        super().__init__(self.config.model_name, self.config.num_labels)
        
        # Initialize tokenizer and model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                use_auth_token=self.config.use_auth_token
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, 
                num_labels=self.config.num_labels,
                cache_dir=self.config.cache_dir,
                use_auth_token=self.config.use_auth_token
            )
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise ValueError(f"Failed to load model '{self.config.model_name}': {str(e)}")
        
        # Initialize trainer and collator
        self.trainer = None
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.config.fp16 and torch.cuda.is_available() else None
        
    def prepare_data(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for transformer model with optimized memory usage
        
        Args:
            data_splits: Dictionary with data splits (train, val, test)
        
        Returns:
            Dictionary with processed features
        """
        result = {}
        
        # Process each split
        for split_name, df in data_splits.items():
            # Extract texts and labels
            texts = df["processed_text"].tolist()
            labels = df["label_encoded"].tolist()
            
            if self.config.streaming:
                # Create streaming dataset
                dataset = StreamingTextDataset(
                    texts, 
                    labels, 
                    self.tokenizer, 
                    self.config.max_length,
                    self.config.streaming_batch_size
                )
            else:
                # Tokenize texts with batching for large datasets
                if len(texts) > 10000:  # For large datasets, use batching
                    encodings = {"input_ids": [], "attention_mask": []}
                    batch_size = 1000
                    
                    for i in range(0, len(texts), batch_size):
                        end_idx = min(i + batch_size, len(texts))
                        batch_texts = texts[i:end_idx]
                        
                        batch_encodings = self.tokenizer(
                            batch_texts, 
                            truncation=True, 
                            padding=True,
                            max_length=self.config.max_length
                        )
                        
                        for key in encodings:
                            encodings[key].extend(batch_encodings[key])
                        
                        # Free memory
                        del batch_encodings
                        if i % 5000 == 0 and i > 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                else:
                    # For smaller datasets, tokenize all at once
                    encodings = self.tokenizer(
                        texts, 
                        truncation=True, 
                        padding=True,
                        max_length=self.config.max_length
                    )
                
                # Create dataset
                dataset = TextDataset(encodings, labels)
            
            # Determine batch size based on available GPU memory
            batch_size = self.config.batch_size
            if torch.cuda.is_available():
                # Scale batch size based on GPU memory (simplified heuristic)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if self.config.model_name.startswith("bert-base"):
                    mem_per_example = 0.0002  # Approximate memory per example in GB
                elif self.config.model_name.startswith("bert-large"):
                    mem_per_example = 0.0004
                elif "roberta" in self.config.model_name:
                    mem_per_example = 0.0003
                else:
                    mem_per_example = 0.0003  # Default estimate
                
                # Leave 20% GPU memory for overhead
                avail_mem = gpu_mem * 0.8
                max_batch_size = int(avail_mem / mem_per_example / self.config.max_length)
                
                # Cap batch size
                batch_size = min(max(16, batch_size), max_batch_size)
                logger.info(f"Dynamic batch size for {split_name}: {batch_size}")
            
            # Create dataloader with optimal batch size
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size if split_name == "train" else self.config.eval_batch_size,
                shuffle=(split_name == "train" and not self.config.streaming),
                num_workers=4 if not self.config.streaming else 0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.data_collator
            )
            
            # Store in result
            result[f"{split_name}_dataset"] = dataset
            result[f"{split_name}_dataloader"] = dataloader
            result[f"{split_name}_texts"] = texts
            result[f"{split_name}_labels"] = labels
            result[f"{split_name}_df"] = df
            
        return result
    
    def train(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any],
        num_epochs: Optional[int] = None, 
        learning_rate: Optional[Union[float, str]] = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Train transformer model with optimized training
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs (overrides config)
            learning_rate: Learning rate (overrides config)
            output_dir: Output directory (overrides config)
            resume_from_checkpoint: Whether to resume from checkpoint if available
            
        Returns:
            Dictionary with training metrics
        """
        # Get datasets
        train_dataset = train_data["train_dataset"]
        val_dataset = val_data["val_dataset"]
        
        # Override config values if provided
        num_epochs = num_epochs or self.config.num_epochs
        
        # Convert learning_rate to float if it's a string
        if learning_rate is not None:
            if isinstance(learning_rate, str):
                learning_rate = float(learning_rate)
        else:
            learning_rate = self.config.learning_rate
            
        # Setup output dir
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,  # Keep only the 3 best checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            fp16_opt_level="O1",  # Use O1 for mixed precision
            dataloader_num_workers=4,
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to="none",
            ddp_find_unused_parameters=False  # Optimize distributed training
        )
        
        # Initialize callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        
        # Check for existing checkpoint if resume_from_checkpoint is True
        last_checkpoint = None
        if resume_from_checkpoint:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is not None:
                logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train model
        train_result = self.trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Save model and tokenizer
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        # Evaluate on validation set
        eval_metrics = self.trainer.evaluate(val_dataset)
        
        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "validation_metrics": eval_metrics
        }
    
    def _normalize_input_data(self, data: Any) -> Tuple[Dataset, List[int]]:
        """
        Normalize different input data formats
        
        Args:
            data: Input data (dataset, texts, or dict)
            
        Returns:
            Tuple of (dataset, labels)
        """
        if isinstance(data, dict) and "test_dataset" in data:
            return data["test_dataset"], data.get("test_labels", [0] * len(data["test_dataset"]))
        
        # Convert to list of texts
        if isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
            labels = data.get("labels", [0] * len(texts))
        elif isinstance(data, list):
            texts = data
            labels = [0] * len(texts)  # Dummy labels
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # Tokenize in batches for large datasets
        if len(texts) > 10000:
            encodings = {"input_ids": [], "attention_mask": []}
            batch_size = 1000
            
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                
                batch_encodings = self.tokenizer(
                    batch_texts, 
                    truncation=True, 
                    padding=True,
                    max_length=self.config.max_length
                )
                
                for key in encodings:
                    encodings[key].extend(batch_encodings[key])
                
                # Free memory
                del batch_encodings
                if i % 5000 == 0 and i > 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # For smaller datasets, tokenize all at once
            encodings = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True,
                max_length=self.config.max_length
            )
        
        # Create dataset
        dataset = TextDataset(encodings, labels)
        return dataset, labels
    
    def predict(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with optimized memory usage
        
        Args:
            data: Input data (dataset, texts, or dict)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Normalize input data
            dataset, _ = self._normalize_input_data(data)
            
            # Use trainer for prediction if available
            if hasattr(self, "trainer") and self.trainer is not None:
                output = self.trainer.predict(dataset)
                raw_preds = output.predictions
            else:
                # Manual prediction with batching and mixed precision
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.config.eval_batch_size, 
                    num_workers=4,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=self.data_collator
                )
                all_outputs = []
                
                self.model.eval()
                with torch.no_grad():
                    for batch in dataloader:
                        batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                        
                        # Use mixed precision if enabled
                        if self.scaler is not None:
                            with autocast():
                                outputs = self.model(**batch)
                        else:
                            outputs = self.model(**batch)
                            
                        all_outputs.append(outputs.logits.cpu().numpy())
                        
                        # Free memory
                        del batch, outputs
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                raw_preds = np.vstack(all_outputs)
            
            # Get predictions and probabilities
            preds = np.argmax(raw_preds, axis=1)
            probs = nn.functional.softmax(torch.tensor(raw_preds), dim=1).numpy()
            
            # Free memory
            del raw_preds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return preds, probs
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model with optimized implementation
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get test dataset and labels
            if "test_dataset" in test_data and "test_labels" in test_data:
                test_dataset = test_data["test_dataset"]
                test_labels = test_data["test_labels"]
            else:
                # Handle case where dataset needs to be created
                test_dataset, test_labels = self._normalize_input_data(test_data)
            
            # Make predictions
            if hasattr(self, "trainer") and self.trainer is not None:
                # Use trainer for evaluation
                metrics = self.trainer.evaluate(test_dataset)
                pred_output = self.trainer.predict(test_dataset)
                preds = np.argmax(pred_output.predictions, axis=1)
                probs = nn.functional.softmax(
                    torch.tensor(pred_output.predictions), 
                    dim=1
                ).numpy()
            else:
                # Manual evaluation
                preds, probs = self.predict({"test_dataset": test_dataset})
                
                # Calculate metrics
                accuracy = accuracy_score(test_labels, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_labels, preds, average='binary', zero_division=0
                )
                
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            
            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                **metrics,
                "predictions": preds,
                "probabilities": probs,
                "true_labels": test_labels
            }
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def save(self, output_dir: str) -> None:
        """
        Save model to disk with additional configuration
        
        Args:
            output_dir: Directory to save model
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save config
            model_config = {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "eval_batch_size": self.config.eval_batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "warmup_ratio": self.config.warmup_ratio,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "fp16": self.config.fp16,
                "streaming": self.config.streaming,
                "streaming_batch_size": self.config.streaming_batch_size
            }
            
            with open(os.path.join(output_dir, "model_config.json"), "w") as f:
                json.dump(model_config, f, indent=2)
                
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self, input_dir: str) -> None:
        """
        Load model from disk with error handling
        
        Args:
            input_dir: Directory to load model from
        """
        try:
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
            
            # Move model to device
            self.model.to(self.device)
            
            # Load config
            try:
                with open(os.path.join(input_dir, "model_config.json"), "r") as f:
                    model_config = json.load(f)
                    
                # Update model attributes
                self.num_labels = ensure_type(model_config.get("num_labels", self.num_labels), int)
                
                # Update config attributes
                self.config.max_length = ensure_type(model_config.get("max_length", self.config.max_length), int)
                self.config.batch_size = ensure_type(model_config.get("batch_size", self.config.batch_size), int)
                self.config.eval_batch_size = ensure_type(model_config.get("eval_batch_size", self.config.eval_batch_size), int)
                self.config.learning_rate = ensure_type(model_config.get("learning_rate", self.config.learning_rate), float)
                self.config.weight_decay = ensure_type(model_config.get("weight_decay", self.config.weight_decay), float)
                self.config.warmup_ratio = ensure_type(model_config.get("warmup_ratio", self.config.warmup_ratio), float)
                self.config.gradient_accumulation_steps = ensure_type(
                    model_config.get("gradient_accumulation_steps", self.config.gradient_accumulation_steps), int
                )
                self.config.fp16 = ensure_type(model_config.get("fp16", self.config.fp16), bool)
                self.config.streaming = ensure_type(model_config.get("streaming", self.config.streaming), bool)
                self.config.streaming_batch_size = ensure_type(
                    model_config.get("streaming_batch_size", self.config.streaming_batch_size), int
                )
                
            except FileNotFoundError:
                logger.warning(f"Model config file not found in {input_dir}, using default values")
            
            # Initialize data collator
            self.data_collator = DataCollatorWithPadding(self.tokenizer)
            
            # Initialize mixed precision scaler
            self.scaler = GradScaler() if self.config.fp16 and torch.cuda.is_available() else None
            
            logger.info(f"Model loaded from {input_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {input_dir}: {str(e)}")
    
    def train_with_distributed(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any],
        num_epochs: Optional[int] = None,
        learning_rate: Optional[Union[float, str]] = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: bool = True,
        num_gpus: int = -1  # -1 means use all available GPUs
    ) -> Dict[str, Any]:
        """
        Train with distributed data parallel
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Output directory
            resume_from_checkpoint: Whether to resume from checkpoint
            num_gpus: Number of GPUs to use (-1 for all)
            
        Returns:
            Dictionary with training metrics
        """
        # Check if multiple GPUs are available
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            logger.warning("Multiple GPUs not available, falling back to single GPU/CPU training")
            return self.train(
                train_data, 
                val_data, 
                num_epochs, 
                learning_rate, 
                output_dir, 
                resume_from_checkpoint
            )
        
        # Determine number of GPUs to use
        if num_gpus < 0:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = min(num_gpus, torch.cuda.device_count())
            
        logger.info(f"Training with {num_gpus} GPUs")
        
        # Override config values if provided
        num_epochs = num_epochs or self.config.num_epochs
        
        # Convert learning_rate to float if it's a string
        if learning_rate is not None:
            if isinstance(learning_rate, str):
                learning_rate = float(learning_rate)
        else:
            learning_rate = self.config.learning_rate
            
        # Setup output dir
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define distributed training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=self.config.fp16,
            fp16_opt_level="O1",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="none",
            # Distributed training parameters
            local_rank=-1,
            sharded_ddp=True,
            ddp_find_unused_parameters=False
        )
        
        # Initialize callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        
        # Check for existing checkpoint if resume_from_checkpoint is True
        last_checkpoint = None
        if resume_from_checkpoint:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is not None:
                logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        
        # Initialize trainer with distributed settings
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data["train_dataset"],
            eval_dataset=val_data["val_dataset"],
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train model
        train_result = self.trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Save model and tokenizer
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        self.trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        # Evaluate on validation set
        eval_metrics = self.trainer.evaluate(val_data["val_dataset"])
        
        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "validation_metrics": eval_metrics
        }