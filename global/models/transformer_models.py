import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_scheduler,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset class for text classification with transformers"""
    
    def __init__(self, encodings: Dict, labels: List[int]):
        """
        Args:
            encodings: Dictionary of token encodings from the tokenizer
            labels: List of labels
        """
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

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

class TransformerModel(BaseModel):
    """Base class for transformer-based models (BERT, RoBERTa, etc.)"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        num_labels: int = 2,
        max_length: int = 512,
        tokenizer_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize transformer model
        
        Args:
            model_name: Name or path of pretrained model
            num_labels: Number of output classes
            max_length: Maximum sequence length
            tokenizer_name: Tokenizer name (if different from model_name)
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, num_labels)
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name or model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_map = {0: "Reliable", 1: "Misinformation"}
        self.inv_label_map = {"Reliable": 0, "Misinformation": 1}
        
        # Initialize tokenizer and model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize tokenizer and model"""
        try:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            )
            
            # Move model to device
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, Dict], 
        text_column: str = "content",
        label_column: str = "label", 
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare data for training or prediction
        
        Args:
            data: DataFrame with text and labels or already split data dict
            text_column: Column name containing text data
            label_column: Column name containing labels
            test_size: Proportion of data to use for testing (if splitting)
            **kwargs: Additional data preparation parameters
            
        Returns:
            Dictionary with prepared datasets and dataloaders
        """
        logger.info("Preparing data for transformer model")
        
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
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, "
                   f"Test: {len(test_df) if test_df is not None else 'N/A'}")
        
        # Prepare datasets
        return self._create_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            text_column=text_column,
            label_column=label_column,
            **kwargs
        )
    
    def _create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        text_column: str = "content",
        label_column: str = "label",
        batch_size: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create datasets and dataloaders
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data (optional)
            text_column: Column name containing text data
            label_column: Column name containing labels
            batch_size: Batch size for dataloaders
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with datasets and dataloaders
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
        
        # Tokenize texts
        train_encodings = self.tokenizer(
            train_texts, 
            truncation=True, 
            padding=True,
            max_length=self.max_length
        )
        
        val_encodings = self.tokenizer(
            val_texts, 
            truncation=True, 
            padding=True,
            max_length=self.max_length
        )
        
        test_encodings = None
        if test_texts:
            test_encodings = self.tokenizer(
                test_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        # Create datasets
        train_dataset = TextDataset(train_encodings, train_labels)
        val_dataset = TextDataset(val_encodings, val_labels)
        test_dataset = None
        if test_encodings:
            test_dataset = TextDataset(test_encodings, test_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size
            )
        
        # Store original data
        if test_dataset:
            test_dataset.orig_df = test_df
        
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df
        }
    
    def train(
        self, 
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs",
        use_trainer: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Training data dictionary with loader/dataset
            val_data: Validation data dictionary with loader/dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Portion of steps for warmup
            output_dir: Directory to save model checkpoints
            use_trainer: Whether to use HuggingFace Trainer (otherwise use manual loop)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_name} for {num_epochs} epochs")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if use_trainer:
            return self._train_with_trainer(
                train_data=train_data,
                val_data=val_data,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                output_dir=output_dir,
                **kwargs
            )
        else:
            return self._train_manual_loop(
                train_data=train_data,
                val_data=val_data,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                output_dir=output_dir,
                **kwargs
            )
    
    def _train_with_trainer(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs",
        **kwargs
    ) -> Dict[str, Any]:
        """Train with HuggingFace Trainer"""
        train_dataset = train_data.get("train_dataset")
        val_dataset = val_data.get("val_dataset") if val_data else None
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=kwargs.get("batch_size", 16),
            per_device_eval_batch_size=kwargs.get("batch_size", 16),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_dir=f"{output_dir}/logs",
            logging_steps=kwargs.get("logging_steps", 50),
            eval_strategy=kwargs.get("evaluation_strategy", "steps"),
            save_strategy=kwargs.get("save_strategy", "steps"),
            save_steps=kwargs.get("save_steps", 100),
            eval_steps=kwargs.get("eval_steps", 100),
            load_best_model_at_end=kwargs.get("load_best_model_at_end", True),
            metric_for_best_model=kwargs.get("metric_for_best_model", "f1"),
            greater_is_better=kwargs.get("greater_is_better", True),
            save_total_limit=kwargs.get("save_total_limit", 2),
            fp16=torch.cuda.is_available() and kwargs.get("fp16", True),
            report_to=kwargs.get("report_to", "none"),
            dataloader_drop_last=kwargs.get("dataloader_drop_last", False),
            remove_unused_columns=kwargs.get("remove_unused_columns", False)
        )
        
        # Initialize callbacks
        callbacks = []
        if kwargs.get("early_stopping", True):
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=kwargs.get("early_stopping_patience", 3)
            )
            callbacks.append(early_stopping)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # Train model
        logger.info("Starting training with Trainer")
        train_result = self.trainer.train()
        
        # Save best model
        self.trainer.save_model(f"{output_dir}/best_model")
        
        # Evaluate on validation set
        if val_dataset:
            logger.info("Evaluating on validation set")
            eval_metrics = self.trainer.evaluate(val_dataset)
            logger.info(f"Validation metrics: {eval_metrics}")
        
        return {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "validation_metrics": eval_metrics if val_dataset else None
        }
    
    def _train_manual_loop(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs",
        **kwargs
    ) -> Dict[str, Any]:
        """Train with manual loop (for more control)"""
        train_loader = train_data.get("train_loader")
        val_loader = val_data.get("val_loader") if val_data else None
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize trackers
        best_val_f1 = 0.0
        best_model_state = None
        epochs_without_improvement = 0
        max_epochs_without_improvement = kwargs.get("early_stopping_patience", 3)
        
        # Training loop
        logger.info("Starting manual training loop")
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg. Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_metrics = self._validate_manual(val_loader)
                val_f1 = val_metrics.get("f1", 0.0)
                
                logger.info(f"Validation - F1: {val_f1:.4f}, Accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    epochs_without_improvement = 0
                    
                    # Save checkpoint
                    torch.save(self.model.state_dict(), f"{output_dir}/best_model.pt")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if kwargs.get("early_stopping", True) and epochs_without_improvement >= max_epochs_without_improvement:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Save final model
        torch.save(self.model.state_dict(), f"{output_dir}/final_model.pt")
        
        return {
            "num_epochs_trained": epoch + 1,
            "best_val_f1": best_val_f1
        }
    
    def _validate_manual(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with manual loop"""
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        return {
            "loss": val_loss / len(val_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        batch_size: int = 16,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input data (dataloader, dataframe, or list of texts)
            batch_size: Batch size for inference
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions")
        
        # Setup data for prediction
        if isinstance(data, dict):
            # Using data dictionary with dataloader
            test_loader = data.get("test_loader")
            if test_loader is None:
                raise ValueError("No test_loader found in data dictionary")
        else:
            # Create dataloader from raw data
            if isinstance(data, pd.DataFrame):
                texts = data[kwargs.get("text_column", "content")].tolist()
            elif isinstance(data, list):
                texts = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Create dummy labels (not used for prediction)
            dummy_labels = [0] * len(texts)
            
            # Create dataset and dataloader
            dataset = TextDataset(encodings, dummy_labels)
            test_loader = DataLoader(dataset, batch_size=batch_size)
        
        # Make predictions
        self.model.eval()
        all_logits = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                all_logits.append(logits.cpu().numpy())
        
        # Concatenate batch results
        logits = np.vstack(all_logits)
        
        # Convert to probabilities and predictions
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        
        return preds, probs
    
    def evaluate(
        self, 
        test_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            test_data: Test data dictionary with loader/dataset
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")
        
        if self.trainer:
            # Use trainer for evaluation
            test_dataset = test_data.get("test_dataset")
            if test_dataset is None:
                raise ValueError("No test_dataset found in test_data")
                
            # Evaluate
            metrics = self.trainer.evaluate(test_dataset)
            return metrics
        else:
            # Use manual evaluation
            test_loader = test_data.get("test_loader")
            if test_loader is None:
                raise ValueError("No test_loader found in test_data")
                
            # Get predictions
            preds, probs = self.predict({"test_loader": test_loader})
            
            # Get true labels
            true_labels = []
            for batch in test_loader:
                true_labels.extend(batch["labels"].cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, preds, average='binary', zero_division=0
            )
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predictions": preds,
                "probabilities": probs,
                "true_labels": true_labels
            }
    
    def save(self, output_dir: str) -> None:
        """
        Save model to disk
        
        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        if self.trainer:
            # Save with trainer
            self.trainer.save_model(output_dir)
        else:
            # Save manually
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save config
            model_config = {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "max_length": self.max_length,
                "tokenizer_name": self.tokenizer_name
            }
            
            with open(os.path.join(output_dir, "model_config.json"), "w") as f:
                import json
                json.dump(model_config, f)
    
    def load(self, input_dir: str) -> None:
        """
        Load model from disk
        
        Args:
            input_dir: Directory to load model from
        """
        logger.info(f"Loading model from {input_dir}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        
        # Move model to device
        self.model.to(self.device)
        
        # Update model config
        try:
            import json
            with open(os.path.join(input_dir, "model_config.json"), "r") as f:
                model_config = json.load(f)
                
            self.num_labels = model_config.get("num_labels", self.num_labels)
            self.max_length = model_config.get("max_length", self.max_length)
        except FileNotFoundError:
            logger.warning("model_config.json not found, using default configuration")


class BERTModel(TransformerModel):
    """BERT model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        num_labels: int = 2,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            **kwargs
        )


class RoBERTaModel(TransformerModel):
    """RoBERTa model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "roberta-base", 
        num_labels: int = 2,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            **kwargs
        )


class DistilBERTModel(TransformerModel):
    """DistilBERT model for text classification"""
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased", 
        num_labels: int = 2,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            **kwargs
        )


# Custom model architecture class for RAG
class CustomTransformerModel(TransformerModel):
    """Custom transformer model architecture"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        num_labels: int = 2,
        max_length: int = 512,
        **kwargs
    ):
        # Initialize base class but don't load the model yet
        super(TransformerModel, self).__init__(model_name, num_labels)
        self.max_length = max_length
        self.tokenizer_name = kwargs.get("tokenizer_name", model_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Create custom model
        self._create_custom_model()
    
    def _create_custom_model(self):
        """Create custom model architecture"""
        # Here you would implement your custom model architecture
        # For example, the SimpleRAGModel from your code
        class SimpleRAGModel(nn.Module):
            def __init__(self, base_model_name, num_labels=2):
                super().__init__()
                # Load pre-trained model
                from transformers import AutoModel
                self.base_model = AutoModel.from_pretrained(base_model_name)
                hidden_size = self.base_model.config.hidden_size
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, num_labels)
                )
                
                # Set config for compatibility with HF Trainer
                self.config = self.base_model.config
                self.config.num_labels = num_labels
            
            def forward(self, input_ids, attention_mask, labels=None):
                # Get base model outputs
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get CLS token representation
                pooled_output = outputs.last_hidden_state[:, 0]
                
                # Classification
                logits = self.classifier(pooled_output)
                
                # Calculate loss if labels provided
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                    
                return (loss, logits) if loss is not None else logits
        
        # Create the model
        self.model = SimpleRAGModel(self.model_name, self.num_labels)
        self.model.to(self.device)