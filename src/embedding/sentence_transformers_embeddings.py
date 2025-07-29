"""
Sentence Transformers-based embedding service for semantic spreadsheet search.
Fast, scalable, and no API rate limits.
"""

import logging
import numpy as np
import os
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from core.config import Config
except ImportError:
    # Fallback if config import fails
    from dotenv import load_dotenv
    load_dotenv()
    
    class Config:
        EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

logger = logging.getLogger(__name__)

@dataclass
class EmbeddedChunk:
    """A semantic chunk with its embedding."""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str
    embedding: List[float]

class SentenceTransformersEmbeddingService:
    """Embedding service using Sentence Transformers - fast and scalable."""
    
    def __init__(self, model_name: str = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package not available. Install with: pip install sentence-transformers")
        
        if not model_name:
            model_name = Config.EMBEDDING_MODEL
        
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Load the model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded embedding model: {model_name}")
        
    def embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_tensor=False, 
                normalize_embeddings=True,
                show_progress_bar=show_progress  # Control progress bar visibility
            )
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def embed_semantic_chunks(self, chunks: List[Any], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for semantic chunks efficiently in batches."""
        if not chunks:
            return []
        
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks in {total_batches} batches of {batch_size}...")
        
        embedded_chunks = []
        
        # Process chunks in batches for better performance and accuracy
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            batch_num = i // batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
            
            # Generate embeddings for this batch (show progress only for first batch to avoid spam)
            show_progress = (batch_num == 1)
            batch_embeddings = self.embed_batch(batch_texts, show_progress=show_progress)
            
            # Create embedded chunks in the format expected by ChromaManager
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                embedded_chunk = {
                    "id": f"chunk_{i + j}_{hash(chunk.content) % 100000}",  # Generate unique ID
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_type": chunk.chunk_type,
                    "embedding": embedding
                }
                embedded_chunks.append(embedded_chunk)
        
        self.logger.info(f"✅ Successfully generated embeddings for {len(embedded_chunks)} chunks in {total_batches} batches")
        return embedded_chunks

# For backward compatibility, create an alias
EmbeddingService = SentenceTransformersEmbeddingService
