# retrieval/batch_retriever.py
import torch
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from .retriever import BaseRetriever

logger = logging.getLogger(__name__)

class BatchRetriever(BaseRetriever):
    """Retriever with batch processing support"""
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = 2, 
        batch_size: int = 128
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

    def process_dataset(
        self, 
        texts: List[str], 
        top_k: int = 2, 
        batch_size: int = 128,
        combine: bool = True
    ) -> Tuple[List[str], List[List[Tuple[str, float, str]]]]:
        """
        Process a dataset of texts with knowledge retrieval
        
        Args:
            texts: List of texts to process
            top_k: Number of knowledge items to retrieve
            batch_size: Batch size for processing
            combine: Whether to combine texts with knowledge
            
        Returns:
            Tuple of (combined_texts or original_texts, retrieval_results)
        """
        # Retrieve knowledge
        logger.info(f"Processing dataset with {len(texts)} texts")
        retrieval_results = self.batch_retrieve(
            texts, 
            top_k=top_k, 
            batch_size=batch_size
        )
        
        if not combine:
            return texts, retrieval_results
        
        # Combine texts with knowledge
        combined_texts = []
        for i, text in enumerate(texts):
            # Extract knowledge items
            knowledge_items = [item for item, _, _ in retrieval_results[i]]
            
            # Combine
            combined_text = self.combine_text_with_knowledge(text, knowledge_items)
            combined_texts.append(combined_text)
        
        return combined_texts, retrieval_results
    
    def combine_text_with_knowledge(
        self, 
        text: str, 
        knowledge_items: List[str]
    ) -> str:
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