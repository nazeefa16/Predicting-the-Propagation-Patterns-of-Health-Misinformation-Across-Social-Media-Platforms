import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
from .base import BaseModel

logger = logging.getLogger(__name__)

class TraditionalModel(BaseModel):
    """Base class for traditional ML models (e.g., LogisticRegression)"""
    
    def __init__(
        self, 
        model_name: str = "logistic_regression", 
        num_labels: int = 2,
        vectorizer_type: str = "tfidf",
        max_features: int = 5000,
        **kwargs
    ):
        """
        Initialize traditional model
        
        Args:
            model_name: Name of the model type
            num_labels: Number of output classes
            vectorizer_type: Type of vectorizer ("tfidf" or "count")
            max_features: Maximum number of features for vectorizer
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, num_labels)
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        self.label_map = {0: "Reliable", 1: "Misinformation"}
        self.inv_label_map = {"Reliable": 0, "Misinformation": 1}
        
        # Initialize vectorizer and model
        self._initialize_components(**kwargs)
    
    def _initialize_components(self, **kwargs):
        """Initialize vectorizer and model components"""
        # Initialize vectorizer
        if self.vectorizer_type.lower() == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=kwargs.get("ngram_range", (1, 1)),
                min_df=kwargs.get("min_df", 5),
                max_df=kwargs.get("max_df", 0.9),
                strip_accents=kwargs.get("strip_accents", 'unicode'),
                use_idf=kwargs.get("use_idf", True),
                analyzer=kwargs.get("analyzer", "word"),
                lowercase=kwargs.get("lowercase", True)
            )
        elif self.vectorizer_type.lower() == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=kwargs.get("ngram_range", (1, 1)),
                min_df=kwargs.get("min_df", 5),
                max_df=kwargs.get("max_df", 0.9),
                analyzer=kwargs.get("analyzer", "word"),
                lowercase=kwargs.get("lowercase", True)
            )
        else:
            raise ValueError(f"Invalid vectorizer_type: {self.vectorizer_type}")
        
        # Initialize model based on model_name
        if self.model_name.lower() == "logistic_regression":
            self.model = LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 1000),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=kwargs.get("random_state", 42),
                solver=kwargs.get("solver", "liblinear"),
                n_jobs=kwargs.get("n_jobs", -1)
            )
        elif self.model_name.lower() == "naive_bayes":
            self.model = MultinomialNB(
                alpha=kwargs.get("alpha", 1.0),
                fit_prior=kwargs.get("fit_prior", True)
            )
        elif self.model_name.lower() == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1)
            )
        elif self.model_name.lower() == "svm":
            self.model = SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "linear"),
                probability=kwargs.get("probability", True),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Invalid model_name: {self.model_name}")
    
    def prepare_data(self, data_splits):
        """Prepare data for traditional model"""
        logger.info("Preparing data for traditional model")
        
        # Log split sizes
        logger.info(f"Data split - Train: {len(data_splits['train'])}, "
                    f"Val: {len(data_splits['val'])}, "
                    f"Test: {len(data_splits['test'])}")
        
        # Process each split
        result = {}
        
        for split_name, df in data_splits.items():
            # Get texts and labels
            texts = df["processed_text"].tolist()
            
            # Important: Store both original labels and encoded labels
            labels =  df["label"].tolist()
            labels_encoded = df["label_encoded"].tolist()  # Use label_encoded column
            
            # Vectorize text with TF-IDF
            if split_name == "train":
                # Fit vectorizer on training data
                logger.info("Fitting tfidf vectorizer")
                self.vectorizer.fit(texts)
            
            # Transform texts to feature vectors
            features = self.vectorizer.transform(texts)
            
            # Store in result dictionary
            result[f"{split_name}_features"] = features
            result[f"{split_name}_labels"] = labels  # Original labels
            result[f"{split_name}_labels_encoded"] = labels_encoded  # Encoded labels
            result[f"{split_name}_texts"] = texts
            result[f"{split_name}_df"] = df
        
        return result
    
    def train(self, train_data, val_data, num_epochs=3, learning_rate=0.001):
        """Train traditional model"""
        logger.info(f"Training {self.model_name}")
        
        # Get data - IMPORTANT: Use label_encoded instead of original labels
        X_train = train_data["train_features"]
        y_train = train_data["train_labels_encoded"]  # Use encoded labels here
        X_val = val_data["val_features"]
        y_val = val_data["val_labels_encoded"]  # Use encoded labels here
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        # Get probabilities if available
        if hasattr(self.model, "predict_proba"):
            train_probs = self.model.predict_proba(X_train)
            val_probs = self.model.predict_proba(X_val)
            # For binary classification, get probabilities of positive class
            if train_probs.shape[1] == 2:
                train_probs = train_probs[:, 1]
                val_probs = val_probs[:, 1]
        else:
            # Fallback for models without predict_proba
            train_probs = None
            val_probs = None
        
        # Calculate metrics - now using numeric labels
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train, train_preds, average='binary', pos_label=1, zero_division=0
        )
        
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='binary', pos_label=1, zero_division=0
        )
        
        return {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "train_time": train_time
        }

    
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input data (features, dataframe, or list of texts)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions")
        
        # Setup data for prediction
        if isinstance(data, dict):
            # Using data dictionary with features
            features = None
            if "test_features" in data:
                features = data["test_features"]
            elif "val_features" in data:
                features = data["val_features"]
            
            if features is None:
                raise ValueError("No features found in data dictionary")
        else:
            # Create features from raw data
            if isinstance(data, pd.DataFrame):
                texts = data[kwargs.get("text_column", "content")].tolist()
            elif isinstance(data, list):
                texts = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Vectorize texts
            features = self.vectorizer.transform(texts)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba(features)
        except AttributeError:
            # Some models don't have predict_proba, create dummy probabilities
            probabilities = np.zeros((len(predictions), self.num_labels))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0
        
        return predictions, probabilities
    
    def evaluate(self, test_data):
        """Evaluate model"""
        # Get data - IMPORTANT: Use label_encoded instead of original labels
        X_test = test_data["test_features"]
        y_test = test_data["test_labels_encoded"]  # Use encoded labels here
        
        # Make predictions
        preds = self.model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_test)
            # For binary classification, get probabilities of positive class
            if probs.shape[1] == 2:
                probs = probs[:, 1]
        else:
            # Fallback for models without predict_proba
            probs = np.zeros(len(preds))
        
        # Calculate metrics - now using numeric labels
        accuracy = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary', pos_label=1, zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": preds,
            "true_labels": y_test,
            "probabilities": probs
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save model to disk
        
        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save config
        model_config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "vectorizer_type": self.vectorizer_type,
            "max_features": self.max_features
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
        
        # Load model
        with open(os.path.join(input_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(os.path.join(input_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        
        # Load config
        try:
            import json
            with open(os.path.join(input_dir, "model_config.json"), "r") as f:
                model_config = json.load(f)
                
            self.num_labels = model_config.get("num_labels", self.num_labels)
            self.vectorizer_type = model_config.get("vectorizer_type", self.vectorizer_type)
            self.max_features = model_config.get("max_features", self.max_features)
        except FileNotFoundError:
            logger.warning("model_config.json not found, using default configuration")


class LogisticRegressionModel(TraditionalModel):
    """Logistic Regression model for text classification"""
    
    def __init__(
        self, 
        vectorizer_type: str = "tfidf",
        max_features: int = 5000,
        **kwargs
    ):
        super().__init__(
            model_name="logistic_regression",
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            **kwargs
        )


class NaiveBayesModel(TraditionalModel):
    """Naive Bayes model for text classification"""
    
    def __init__(
        self, 
        vectorizer_type: str = "tfidf",
        max_features: int = 5000,
        **kwargs
    ):
        super().__init__(
            model_name="naive_bayes",
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            **kwargs
        )


class RandomForestModel(TraditionalModel):
    """Random Forest model for text classification"""
    
    def __init__(
        self, 
        vectorizer_type: str = "tfidf",
        max_features: int = 5000,
        **kwargs
    ):
        super().__init__(
            model_name="random_forest",
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            **kwargs
        )


class SVMModel(TraditionalModel):
    """SVM model for text classification"""
    
    def __init__(
        self, 
        vectorizer_type: str = "tfidf",
        max_features: int = 5000,
        **kwargs
    ):
        super().__init__(
            model_name="svm",
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            **kwargs
        )