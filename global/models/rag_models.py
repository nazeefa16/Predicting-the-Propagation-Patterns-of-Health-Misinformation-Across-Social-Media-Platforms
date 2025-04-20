import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer, util
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class BaseRetriever:
    """Base class for knowledge retrievers"""
    
    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    ):
        """
        Initialize base retriever
        
        Args:
            embedding_model: Name of embedding model
            **kwargs: Additional retriever parameters
        """
        self.embedding_model = embedding_model
        self.encoder = None
        self.knowledge_base = None
        self.knowledge_texts = None
        self.knowledge_embeddings = None
        self.knowledge_keys = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize encoder
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize encoder"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self.encoder = SentenceTransformer(self.embedding_model)
            self.encoder.to(self.device)
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise
    
    def index_knowledge_base(self, knowledge_base: Dict[str, str]):
        """
        Index knowledge base
        
        Args:
            knowledge_base: Dictionary mapping keys to knowledge texts
        """
        try:
            logger.info(f"Indexing knowledge base with {len(knowledge_base)} items")
            self.knowledge_base = knowledge_base
            self.knowledge_texts = list(knowledge_base.values())
            self.knowledge_keys = list(knowledge_base.keys())
            
            # Compute embeddings
            self.knowledge_embeddings = self.encoder.encode(
                self.knowledge_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            ).to(self.device)
            
            logger.info(f"Knowledge base indexed successfully")
        except Exception as e:
            logger.error(f"Error indexing knowledge base: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant knowledge for a query
        
        Args:
            query: Input query
            top_k: Number of items to retrieve
            
        Returns:
            List of (knowledge_text, similarity_score, knowledge_key) tuples
        """
        try:
            # Encode query
            query_embedding = self.encoder.encode(
                query,
                convert_to_tensor=True
            ).to(self.device)
            
            # Calculate similarities
            similarities = util.pytorch_cos_sim(query_embedding, self.knowledge_embeddings)[0]
            
            # Get top-k
            top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
            
            # Get items and scores
            items = [self.knowledge_texts[idx] for idx in top_indices]
            scores = [similarities[idx].item() for idx in top_indices]
            keys = [self.knowledge_keys[idx] for idx in top_indices]
            
            return list(zip(items, scores, keys))
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            # Return fallback
            return [("Health information should be verified with trusted sources.", 0.5, "general")]


class BatchRetriever(BaseRetriever):
    """Retriever with batch processing support"""
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = 2, 
        batch_size: int = 64
    ) -> List[List[Tuple[str, float, str]]]:
        """
        Retrieve knowledge for multiple queries in batches
        
        Args:
            queries: List of input queries
            top_k: Number of items to retrieve per query
            batch_size: Batch size for processing
            
        Returns:
            List of retrieval results for each query
        """
        try:
            all_results = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(queries), batch_size):
                # Get batch of queries
                batch_queries = queries[i:i+batch_size]
                
                # Encode batch
                batch_embeddings = self.encoder.encode(
                    batch_queries,
                    convert_to_tensor=True,
                    show_progress_bar=False
                ).to(self.device)
                
                # Calculate similarities for batch
                batch_similarities = util.pytorch_cos_sim(batch_embeddings, self.knowledge_embeddings)
                
                # Process each query in batch
                batch_results = []
                for j, similarities in enumerate(batch_similarities):
                    # Get top-k indices
                    top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
                    
                    # Get items and scores
                    items = [self.knowledge_texts[idx] for idx in top_indices]
                    scores = [similarities[idx].item() for idx in top_indices]
                    keys = [self.knowledge_keys[idx] for idx in top_indices]
                    
                    # Store results for this query
                    query_results = list(zip(items, scores, keys))
                    batch_results.append(query_results)
                
                # Add batch results
                all_results.extend(batch_results)
            
            return all_results
        except Exception as e:
            logger.error(f"Error in batch_retrieve: {e}")
            # Return fallback results
            return [[("Health information should be verified with trusted sources.", 0.5, "general")] 
                   for _ in range(len(queries))]


class RAGModelWrapper(BaseModel):
    """
    Wrapper for RAG-enhanced models
    
    This class wraps any base model and enhances it with
    retrieval-augmented generation capabilities.
    """
    
    def __init__(
        self, 
        base_model: BaseModel,
        retriever: Optional[BaseRetriever] = None,
        knowledge_base: Optional[Dict[str, str]] = None,
        top_k: int = 2,
        batch_size: int = 64,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    ):
        """
        Initialize RAG wrapper
        
        Args:
            base_model: Base model to wrap
            retriever: Knowledge retriever (if None, a new one will be created)
            knowledge_base: Knowledge base dictionary (required if retriever is None)
            top_k: Number of knowledge items to retrieve
            batch_size: Batch size for retrieval
            embedding_model: Name of embedding model (if retriever is None)
            **kwargs: Additional parameters
        """
        super().__init__(f"rag_{base_model.model_name}", base_model.num_labels)
        self.base_model = base_model
        self.top_k = top_k
        self.batch_size = batch_size
        
        # Initialize retriever
        if retriever is None:
            if knowledge_base is None:
                raise ValueError("Either retriever or knowledge_base must be provided")
            
            logger.info("Creating new retriever")
            self.retriever = BatchRetriever(embedding_model=embedding_model)
            self.retriever.index_knowledge_base(knowledge_base)
        else:
            self.retriever = retriever
            
            # Index knowledge base if provided
            if knowledge_base is not None:
                self.retriever.index_knowledge_base(knowledge_base)
    
    def combine_text_with_knowledge(
        self, 
        text: str, 
        knowledge_items: List[str],
        **kwargs
    ) -> str:
        """
        Combine text with retrieved knowledge
        
        Args:
            text: Original text
            knowledge_items: Retrieved knowledge items
            **kwargs: Additional parameters
            
        Returns:
            Combined text
        """
        if not knowledge_items:
            return text
        
        # Create combined text
        combined = f"Text: {text} [SEP] "
        
        # Add knowledge items
        for i, knowledge in enumerate(knowledge_items):
            combined += f"Knowledge {i+1}: {knowledge} [SEP] "
        
        return combined.strip()
    
    def retrieve_and_combine(
        self, 
        texts: List[str],
        **kwargs
    ) -> Tuple[List[str], List[List[Tuple[str, float, str]]]]:
        """
        Retrieve knowledge and combine with texts
        
        Args:
            texts: List of input texts
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (combined_texts, retrieval_results)
        """
        # Retrieve knowledge
        logger.info(f"Retrieving knowledge for {len(texts)} texts")
        retrieval_results = self.retriever.batch_retrieve(
            texts, 
            top_k=self.top_k, 
            batch_size=self.batch_size
        )
        
        # Combine texts with knowledge
        combined_texts = []
        for i, text in enumerate(texts):
            # Extract knowledge items
            knowledge_items = [item for item, _, _ in retrieval_results[i]]
            
            # Combine
            combined_text = self.combine_text_with_knowledge(text, knowledge_items, **kwargs)
            combined_texts.append(combined_text)
        
        return combined_texts, retrieval_results
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, Dict], 
        text_column: str = "content",
        label_column: str = "label", 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare data with RAG enhancement
        
        Args:
            data: DataFrame with text and labels or already split data dict
            text_column: Column name containing text data
            label_column: Column name containing labels
            **kwargs: Additional data preparation parameters
            
        Returns:
            Dictionary with prepared datasets
        """
        logger.info("Preparing data with RAG enhancement")
        
        # Let base model prepare the data first
        base_data = self.base_model.prepare_data(
            data, 
            text_column=text_column,
            label_column=label_column,
            **kwargs
        )
        
        # Extract dataframes
        train_df = base_data.get("train_df")
        val_df = base_data.get("val_df")
        test_df = base_data.get("test_df")
        
        # Process each split with RAG
        processed_data = {}
        
        if train_df is not None:
            train_processed = self._process_split(
                train_df, text_column, label_column, "train", **kwargs
            )
            processed_data.update(train_processed)
        
        if val_df is not None:
            val_processed = self._process_split(
                val_df, text_column, label_column, "val", **kwargs
            )
            processed_data.update(val_processed)
        
        if test_df is not None:
            test_processed = self._process_split(
                test_df, text_column, label_column, "test", **kwargs
            )
            processed_data.update(test_processed)
        
        # Merge with base_data (keeping RAG specific data)
        processed_data.update({
            k: v for k, v in base_data.items() 
            if k not in processed_data and not k.startswith(("train_", "val_", "test_"))
        })
        
        return processed_data
    
    def _process_split(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        label_column: str,
        split_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a data split with RAG
        
        Args:
            df: DataFrame to process
            text_column: Column name containing text data
            label_column: Column name containing labels
            split_name: Name of the split (train, val, test)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processed data
        """
        logger.info(f"Processing {split_name} split with RAG")
        
        # Extract texts and labels
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Retrieve and combine
        combined_texts, retrieval_results = self.retrieve_and_combine(texts, **kwargs)
        
        # Extract knowledge items and scores
        knowledge_items = []
        knowledge_scores = []
        
        for results in retrieval_results:
            items = [item for item, _, _ in results]
            scores = [score for _, score, _ in results]
            knowledge_items.append(items)
            knowledge_scores.append(scores)
        
        # Create processed dataframe
        processed_df = df.copy()
        processed_df["combined_text"] = combined_texts
        processed_df["knowledge_items"] = knowledge_items
        processed_df["knowledge_scores"] = knowledge_scores
        
        # Re-prepare with base model using combined text
        base_data = self.base_model.prepare_data(
            {f"{split_name}": processed_df},
            text_column="combined_text",
            label_column=label_column,
            **kwargs
        )
        
        # Extract and rename the elements
        df_key = f"{split_name}_df"
        features_key = f"{split_name}_features"
        dataset_key = f"{split_name}_dataset"
        loader_key = f"{split_name}_loader"
        texts_key = f"{split_name}_texts"
        labels_key = f"{split_name}_labels"
        
        return {
            df_key: processed_df,
            features_key: base_data.get(features_key),
            dataset_key: base_data.get(dataset_key),
            loader_key: base_data.get(loader_key),
            texts_key: combined_texts,
            labels_key: labels,
            f"{split_name}_knowledge_items": knowledge_items,
            f"{split_name}_knowledge_scores": knowledge_scores,
            f"{split_name}_original_texts": texts
        }
    
    def train(
        self, 
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model with RAG-enhanced data
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training RAG-enhanced model")
        
        # Train base model using rag-enhanced data
        return self.base_model.train(train_data, val_data, **kwargs)
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with RAG enhancement
        
        Args:
            data: Input data (Dictionary, DataFrame, or list of texts)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions with RAG enhancement")
        
        # Handle different input types
        if isinstance(data, dict):
            # Using already processed data
            return self.base_model.predict(data, **kwargs)
        else:
            # Need to process with RAG first
            if isinstance(data, pd.DataFrame):
                texts = data[kwargs.get("text_column", "content")].tolist()
            elif isinstance(data, list):
                texts = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Retrieve and combine
            combined_texts, _ = self.retrieve_and_combine(texts, **kwargs)
            
            # Use base model for prediction
            return self.base_model.predict(combined_texts, **kwargs)
    
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
        logger.info("Evaluating RAG-enhanced model")
        
        # Use base model for evaluation
        results = self.base_model.evaluate(test_data, **kwargs)
        
        # Add RAG-specific information
        results["model_type"] = "rag"
        results["base_model_name"] = self.base_model.model_name
        results["top_k"] = self.top_k
        
        return results
    
    def save(self, output_dir: str) -> None:
        """
        Save model to disk
        
        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving RAG-enhanced model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base model
        base_model_dir = os.path.join(output_dir, "base_model")
        self.base_model.save(base_model_dir)
        
        # Save RAG configuration
        rag_config = {
            "model_name": self.model_name,
            "base_model_name": self.base_model.model_name,
            "base_model_type": self.base_model.__class__.__name__,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "embedding_model": self.retriever.embedding_model if hasattr(self.retriever, "embedding_model") else None,
        }
        
        with open(os.path.join(output_dir, "rag_config.json"), "w") as f:
            import json
            json.dump(rag_config, f, indent=2)
    
    def load(self, input_dir: str) -> None:
        """
        Load model from disk
        
        Args:
            input_dir: Directory to load model from
        """
        logger.info(f"Loading RAG-enhanced model from {input_dir}")
        
        # Load RAG configuration
        try:
            import json
            with open(os.path.join(input_dir, "rag_config.json"), "r") as f:
                rag_config = json.load(f)
                
            self.top_k = rag_config.get("top_k", self.top_k)
            self.batch_size = rag_config.get("batch_size", self.batch_size)
            
            # Note: we don't load the retriever here since the knowledge base
            # and embeddings can be large. They should be reinitialized separately.
            
        except FileNotFoundError:
            logger.warning("rag_config.json not found, using default configuration")
        
        # Load base model
        base_model_dir = os.path.join(input_dir, "base_model")
        self.base_model.load(base_model_dir)


class RAGTransformerModel(RAGModelWrapper):
    """
    RAG-enhanced transformer model
    
    This is a convenience class for transformer models enhanced with RAG.
    """
    
    def __init__(
        self, 
        base_model_name: str = "bert-base-uncased",
        transformer_type: str = "bert",
        retriever: Optional[BaseRetriever] = None,
        knowledge_base: Optional[Dict[str, str]] = None,
        top_k: int = 2,
        **kwargs
    ):
        """
        Initialize RAG transformer model
        
        Args:
            base_model_name: Name of base transformer model
            transformer_type: Type of transformer ("bert", "roberta", "distilbert")
            retriever: Knowledge retriever (if None, a new one will be created)
            knowledge_base: Knowledge base dictionary (required if retriever is None)
            top_k: Number of knowledge items to retrieve
            **kwargs: Additional parameters
        """
        # Import here to avoid circular imports
        from .transformer_models import BERTModel, RoBERTaModel, DistilBERTModel
        
        # Create base model based on type
        if transformer_type.lower() == "bert":
            base_model = BERTModel(base_model_name, **kwargs)
        elif transformer_type.lower() == "roberta":
            base_model = RoBERTaModel(base_model_name, **kwargs)
        elif transformer_type.lower() == "distilbert":
            base_model = DistilBERTModel(base_model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported transformer_type: {transformer_type}")
        
        # Initialize RAG wrapper
        super().__init__(
            base_model=base_model,
            retriever=retriever,
            knowledge_base=knowledge_base,
            top_k=top_k,
            **kwargs
        )


class RAGLLMModel(RAGModelWrapper):
    """
    RAG-enhanced LLM model
    
    This is a convenience class for LLM models enhanced with RAG.
    """
    
    def __init__(
        self, 
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        llm_type: str = "qwen",
        retriever: Optional[BaseRetriever] = None,
        knowledge_base: Optional[Dict[str, str]] = None,
        top_k: int = 2,
        **kwargs
    ):
        """
        Initialize RAG LLM model
        
        Args:
            base_model_name: Name of base LLM model
            llm_type: Type of LLM ("qwen", "mixtral", "llama")
            retriever: Knowledge retriever (if None, a new one will be created)
            knowledge_base: Knowledge base dictionary (required if retriever is None)
            top_k: Number of knowledge items to retrieve
            **kwargs: Additional parameters
        """
        # Import here to avoid circular imports
        from .llm_models import QwenModel, MixtralModel, LlamaModel
        
        # Create base model based on type
        if llm_type.lower() == "qwen":
            base_model = QwenModel(base_model_name, **kwargs)
        elif llm_type.lower() == "mixtral":
            base_model = MixtralModel(base_model_name, **kwargs)
        elif llm_type.lower() == "llama":
            base_model = LlamaModel(base_model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported llm_type: {llm_type}")
        
        # Initialize RAG wrapper
        super().__init__(
            base_model=base_model,
            retriever=retriever,
            knowledge_base=knowledge_base,
            top_k=top_k,
            **kwargs
        )
    
    def combine_text_with_knowledge(
        self, 
        text: str, 
        knowledge_items: List[str],
        **kwargs
    ) -> str:
        """
        Customize prompt for LLM with knowledge items
        
        Args:
            text: Original text
            knowledge_items: Retrieved knowledge items
            **kwargs: Additional parameters
            
        Returns:
            Combined text
        """
        if not knowledge_items:
            return text
        
        # Format knowledge as a separate context section
        knowledge_context = "\n\n".join([f"FACT: {item}" for item in knowledge_items])
        
        # Create RAG-enhanced prompt
        combined = f"""RELEVANT HEALTH FACTS:
{knowledge_context}

TEXT TO CLASSIFY:
{text}

Based on the health facts above, classify if the text contains health misinformation."""
        
        return combined