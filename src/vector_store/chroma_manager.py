import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import logging
import os
from core.config import Config

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages interactions with ChromaDB vector database."""
    
    def __init__(self, collection_name: str = "semantic_spreadsheet"):
        """Initialize ChromaDB client and collection."""
        self.collection_name = collection_name
        self.persist_directory = Config.get_chroma_path()
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        logger.info(f"Connected to ChromaDB collection: {collection_name}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Semantic spreadsheet embeddings"}
            )
    
    def add_embeddings(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Add embedded chunks to the vector database."""
        if not embedded_chunks:
            logger.warning("No embeddings to add")
            return
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in embedded_chunks]
            embeddings = [chunk["embedding"] for chunk in embedded_chunks]
            documents = [chunk["content"] for chunk in embedded_chunks]
            metadatas = [{
                "chunk_type": chunk["chunk_type"],
                **chunk["metadata"]
            } for chunk in embedded_chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(embedded_chunks)} embeddings to collection")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def query_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar embeddings from the collection."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Collection reference invalid for querying, refreshing: {e}")
            try:
                # Try to refresh the collection reference
                self.collection = self._get_or_create_collection()
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Format results
                formatted_results = []
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    }
                    formatted_results.append(result)
                
                return formatted_results
                
            except Exception as refresh_error:
                logger.error(f"Failed to query embeddings after refresh: {refresh_error}")
                raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_embeddings": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.warning(f"Collection reference invalid, refreshing: {e}")
            try:
                # Try to refresh the collection reference
                self.collection = self._get_or_create_collection()
                count = self.collection.count()
                return {
                    "collection_name": self.collection_name,
                    "total_embeddings": count,
                    "persist_directory": self.persist_directory
                }
            except Exception as refresh_error:
                logger.error(f"Failed to get collection stats after refresh: {refresh_error}")
                return {
                    "collection_name": self.collection_name,
                    "total_embeddings": 0,
                    "persist_directory": self.persist_directory
                }
    
    def clear_collection(self) -> None:
        """Clear all embeddings from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def list_all_chunks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all chunks in the collection."""
        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"])):
                result = {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Collection reference invalid for listing, refreshing: {e}")
            try:
                # Try to refresh the collection reference
                self.collection = self._get_or_create_collection()
                results = self.collection.get(
                    limit=limit,
                    include=["documents", "metadatas"]
                )
                
                formatted_results = []
                for i in range(len(results["ids"])):
                    result = {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    }
                    formatted_results.append(result)
                
                return formatted_results
                
            except Exception as refresh_error:
                logger.error(f"Failed to list chunks after refresh: {refresh_error}")
                return []