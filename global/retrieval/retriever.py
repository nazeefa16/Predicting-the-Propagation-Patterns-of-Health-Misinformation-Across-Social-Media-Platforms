# retrieval/retriever.py
import torch
import logging
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util

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