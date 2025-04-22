# models/rag.py
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, List
from sentence_transformers import SentenceTransformer, util

from .base import BaseModel
from .transformer import TransformerModel
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
from .llm import VLLMModel

logger = logging.getLogger(__name__)

class Retriever:
    """Simple knowledge retriever for RAG models"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retriever
        
        Args:
            embedding_model: Name of sentence transformer model for embeddings
        """
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1") 
        self.encoder.to(self.device)
        
        # Knowledge base
        self.knowledge_base = None
        self.knowledge_texts = None
        self.knowledge_keys = None
        self.knowledge_embeddings = None
        
    def index_knowledge_base(self, knowledge_base: Dict[str, str]):
        """
        Index knowledge base for retrieval
        
        Args:
            knowledge_base: Dictionary mapping keys to knowledge texts
        """
        self.knowledge_base = knowledge_base
        self.knowledge_texts = list(knowledge_base.values())
        self.knowledge_keys = list(knowledge_base.keys())
        
        # Compute embeddings
        self.knowledge_embeddings = self.encoder.encode(
            self.knowledge_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        ).to(self.device)
        
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant knowledge for a query
        
        Args:
            query: Input query
            top_k: Number of items to retrieve
            
        Returns:
            List of (knowledge_text, similarity_score, knowledge_key) tuples
        """
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
    
    def batch_retrieve(self, queries: List[str], top_k: int = 2, batch_size: int = 64) -> List[List[Tuple[str, float, str]]]:
        """
        Retrieve knowledge for multiple queries in batches
        
        Args:
            queries: List of queries
            top_k: Number of items to retrieve per query
            batch_size: Batch size for processing
            
        Returns:
            List of retrieval results for each query
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            # Get batch
            batch_queries = queries[i:i+batch_size]
            
            # Encode batch
            batch_embeddings = self.encoder.encode(
                batch_queries,
                convert_to_tensor=True,
                show_progress_bar=False
            ).to(self.device)
            
            # Calculate similarities
            batch_similarities = util.pytorch_cos_sim(batch_embeddings, self.knowledge_embeddings)
            
            # Process each query in batch
            batch_results = []
            for j, similarities in enumerate(batch_similarities):
                # Get top-k
                top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.tolist()
                
                # Get items and scores
                items = [self.knowledge_texts[idx] for idx in top_indices]
                scores = [similarities[idx].item() for idx in top_indices]
                keys = [self.knowledge_keys[idx] for idx in top_indices]
                
                # Store results
                query_results = list(zip(items, scores, keys))
                batch_results.append(query_results)
            
            # Add batch results
            all_results.extend(batch_results)
        
        return all_results

class RAGModel(BaseModel):
    """
    RAG-enhanced model for health misinformation detection
    
    This model combines a transformer model with a retrieval component
    to improve prediction by providing relevant knowledge.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        top_k: int = 2,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        knowledge_base: Dict[str, str] = None
    ):
        """
        Initialize RAG model
        
        Args:
            model_name: Name of base transformer model
            top_k: Number of knowledge items to retrieve
            embedding_model: Embedding model for retriever
            knowledge_base: Knowledge base dictionary
        """
        super().__init__(f"rag_{model_name}")
        self.base_model = TransformerModel(model_name)
        self.top_k = top_k
        
        # Initialize retriever
        self.retriever = Retriever(embedding_model)
        
        # Index knowledge base if provided
        if knowledge_base:
            self.retriever.index_knowledge_base(knowledge_base)
    
    def combine_text_with_knowledge(self, text: str, knowledge_items: List[str]) -> str:
        """
        Combine text with retrieved knowledge
        
        Args:
            text: Original text
            knowledge_items: Retrieved knowledge items
            
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
    
    def prepare_data(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data with RAG enhancement
        
        Args:
            data_splits: Dictionary with data splits (train, val, test)
        
        Returns:
            Dictionary with processed features
        """
        result = {}
        
        # Process each split
        for split_name, df in data_splits.items():
            # Extract texts
            texts = df["processed_text"].tolist()
            labels = df["label_encoded"].tolist()
            
            # Retrieve knowledge for each text
            knowledge_results = self.retriever.batch_retrieve(texts, self.top_k)
            
            # Combine texts with knowledge
            combined_texts = []
            for i, text in enumerate(texts):
                # Extract knowledge items
                knowledge_items = [item for item, _, _ in knowledge_results[i]]
                
                # Combine text with knowledge
                combined_text = self.combine_text_with_knowledge(text, knowledge_items)
                combined_texts.append(combined_text)
            
            # Create augmented dataframe
            df_aug = df.copy()
            df_aug["processed_text"] = combined_texts
            
            # Store in result
            result[f"{split_name}_df"] = df_aug
            result[f"{split_name}_texts"] = combined_texts
            result[f"{split_name}_labels"] = labels
            result[f"{split_name}_knowledge"] = knowledge_results
        
        # Process with base model
        base_data = self.base_model.prepare_data(result)
        
        # Merge results
        result.update(base_data)
        
        return result
    
    def train(
        self, 
        train_data: Dict[str, Any], 
        val_data: Dict[str, Any],
        num_epochs: int = 3, 
        learning_rate: float = 5e-5
    ) -> Dict[str, Any]:
        """
        Train RAG model (delegates to base model)
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training metrics
        """
        return self.base_model.train(train_data, val_data, num_epochs, learning_rate)
    
    def predict(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input data (dataset or texts)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Process data with RAG if it's a list of texts
        if isinstance(data, list):
            # Retrieve knowledge for each text
            knowledge_results = self.retriever.batch_retrieve(data, self.top_k)
            
            # Combine texts with knowledge
            combined_texts = []
            for i, text in enumerate(data):
                # Extract knowledge items
                knowledge_items = [item for item, _, _ in knowledge_results[i]]
                
                # Combine text with knowledge
                combined_text = self.combine_text_with_knowledge(text, knowledge_items)
                combined_texts.append(combined_text)
            
            # Use base model for prediction
            return self.base_model.predict(combined_texts)
        else:
            # Use base model directly
            return self.base_model.predict(data)
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.base_model.evaluate(test_data)
    
    def save(self, output_dir: str) -> None:
        """
        Save model to disk
        
        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base model
        base_model_dir = os.path.join(output_dir, "base_model")
        self.base_model.save(base_model_dir)
        
        # Save RAG config
        rag_config = {
            "model_name": self.model_name,
            "base_model_name": self.base_model.model_name,
            "top_k": self.top_k,
            "embedding_model": self.retriever.embedding_model
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
        # Load base model
        base_model_dir = os.path.join(input_dir, "base_model")
        self.base_model.load(base_model_dir)
        
        # Load RAG config
        try:
            import json
            with open(os.path.join(input_dir, "rag_config.json"), "r") as f:
                rag_config = json.load(f)
                
            self.top_k = rag_config.get("top_k", self.top_k)
            # Note: we don't reload the retriever as we would need to reindex the knowledge base
        except FileNotFoundError:
            pass  # Use default values

class LLMRAGModel(VLLMModel):
    """
    RAG-enhanced LLM model for health misinformation detection using vLLM
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M", 
        top_k: int = 2,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        knowledge_base: Dict[str, str] = None,
        num_labels: int = 2,
        system_prompt: Optional[str] = None,
        max_model_len: int = 2048, 
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 64,
        **kwargs
    ):
        """
        Initialize RAG-enhanced LLM model
        
        Args:
            model_name: Name of base LLM model
            top_k: Number of knowledge items to retrieve
            embedding_model: Embedding model for retriever
            knowledge_base: Knowledge base dictionary
            num_labels: Number of output classes
            system_prompt: System prompt for the LLM
            **kwargs: Additional model parameters
        """
        # Initialize with a default RAG-specific system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert in Health information fact-checking. "
                "You will be provided with both a text to classify and relevant knowledge from trusted sources. "
                "Use this knowledge to help determine if the text contains misinformation."
            )
            
        # Initialize base LLM
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            system_prompt=system_prompt,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
            **kwargs
        )
        
        # Set model name to reflect RAG enhancement
        self.model_name = f"rag_{model_name}"
        self.top_k = top_k
        
        # Initialize retriever
        self.retriever = Retriever(embedding_model)
        
        # Index knowledge base if provided
        if knowledge_base:
            self.retriever.index_knowledge_base(knowledge_base)
            logger.info(f"Indexed knowledge base with {len(knowledge_base)} entries")
    
    def combine_text_with_knowledge(self, text: str, knowledge_items: List[Tuple[str, float, str]]) -> str:
        """
        Combine text with retrieved knowledge
        
        Args:
            text: Original text
            knowledge_items: Retrieved knowledge items (text, score, key)
            
        Returns:
            Combined text with knowledge context
        """
        if not knowledge_items:
            return text
        
        # Format retrieved knowledge items
        knowledge_section = "\n\nRelevant Knowledge:\n"
        for i, (knowledge_text, score, key) in enumerate(knowledge_items):
            knowledge_section += f"[{i+1}] {knowledge_text} (Source: {key}, Relevance: {score:.2f})\n"
        
        return text + knowledge_section
    
    def _create_prompt(self, text: str, **kwargs) -> str:
        """
        Create prompt for LLM with RAG enhancement
        
        Args:
            text: Input text
            **kwargs: Additional prompt parameters
            
        Returns:
            Formatted prompt
        """
        # Retrieve relevant knowledge
        knowledge_items = kwargs.get("knowledge_items", None)
        
        # If knowledge items not provided, retrieve them
        if knowledge_items is None and hasattr(self, "retriever") and self.retriever.knowledge_embeddings is not None:
            knowledge_items = self.retriever.retrieve(text, self.top_k)
        
        # Enhance text with knowledge if available
        if knowledge_items:
            enhanced_text = self.combine_text_with_knowledge(text, knowledge_items)
        else:
            enhanced_text = text
        
        # Create base prompt using enhanced text
        base_prompt = f"""Classify the following information as either containing Health misinformation (1) or reliable health information (0):

Example 1: "Masks don't work and actually make you sicker by reducing oxygen levels."
Classification: 1 (This is misinformation)

Example 2: "The CDC recommends wearing masks in crowded indoor settings to reduce disease transmission."
Classification: 0 (This is factual information)

Text to classify: "{enhanced_text}"

Based on the text and any relevant knowledge provided, classify if this contains health misinformation.
Classification (0 or 1):"""

        # Use custom prompt if provided
        custom_prompt = kwargs.get("prompt", None)
        if custom_prompt:
            base_prompt = custom_prompt.format(text=enhanced_text)
        
        # Format as chat/instruct prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": base_prompt}
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
            formatted_prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{base_prompt}\n<|assistant|>\n"
        
        return formatted_prompt
    
    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame, List[str]], 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with RAG-enhanced LLM
        
        Args:
            data: Input data (dataframe or list of texts)
            batch_size: Batch size for inference (overrides self.batch_size)
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info("Making predictions with RAG-enhanced LLM")
        
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
        
        # Retrieve knowledge for all texts if we have a retriever
        if hasattr(self, "retriever") and self.retriever.knowledge_embeddings is not None:
            logger.info(f"Retrieving knowledge for {len(texts)} texts")
            batch_knowledge = self.retriever.batch_retrieve(texts, self.top_k)
        else:
            batch_knowledge = [[] for _ in texts]
        
        # Create prompts with retrieved knowledge
        prompts = []
        for i, text in enumerate(texts):
            kwargs["knowledge_items"] = batch_knowledge[i]
            prompt = self._create_prompt(text, **kwargs)
            prompts.append(prompt)
        
        # Process all texts in batches using vLLM
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
    
    # No need to override _extract_label as it's already fixed in the parent class
    
    def save(self, output_dir: str) -> None:
        """
        Save model configuration to disk
        
        Args:
            output_dir: Directory to save configuration
        """
        logger.info(f"Saving RAG-enhanced vLLM configuration to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config (including base vLLM config + RAG configuration)
        model_config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": self.system_prompt,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "embedding_model": self.retriever.embedding_model
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
        logger.info(f"Loading RAG-enhanced vLLM configuration from {input_dir}")
        
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
            
            # Update RAG parameters if available
            if "top_k" in model_config:
                self.top_k = model_config["top_k"]
            
            # Reinitialize model with updated config
            self._initialize_model()
            
        except FileNotFoundError:
            logger.warning("vllm_config.json not found, using default configuration")
            # Reinitialize with current config
            self._initialize_model()