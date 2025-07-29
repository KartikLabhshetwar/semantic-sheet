import logging
import json
from typing import List, Dict, Any, Optional
from core.config import Config
from embedding.sentence_transformers_embeddings import SentenceTransformersEmbeddingService
from vector_store.chroma_manager import ChromaManager

# Handle Google GenAI import compatibility
try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_GOOGLE_GENAI = False
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")

logger = logging.getLogger(__name__)

class RAGProcessor:
    """Processes user queries using Retrieval-Augmented Generation."""
    
    def __init__(self):
        """Initialize the RAG processor with all necessary components."""
        Config.validate()  # Ensure API key is available
        
        # Initialize Gemini based on available library
        if HAS_GOOGLE_GENAI:
            # New google-genai library
            self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
            self.model = "gemini-2.5-flash"
            self._use_new_api = True
        else:
            # Old google-generativeai library
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self._use_new_api = False
        
        # Initialize embedding and vector store services
        self.embedding_service = SentenceTransformersEmbeddingService()
        self.vector_store = ChromaManager()
        
        logger.info("RAG Processor initialized successfully")
    
    def process_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Process a natural language query and return enriched results."""
        if not top_k:
            top_k = Config.TOP_K_RESULTS
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Generate embedding for the query
            query_embedding = self.embedding_service.embed_single_text(query)
            
            # Step 2: Retrieve relevant chunks from vector store
            relevant_chunks = self.vector_store.query_similar(
                query_embedding, 
                top_k=top_k
            )
            
            if not relevant_chunks:
                return {
                    "query": query,
                    "response": "I couldn't find any relevant information in the spreadsheet to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Step 3: Generate response using retrieved context
            response_data = self._generate_response(query, relevant_chunks)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the retrieved context and Gemini."""
        
        # Prepare context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"Context {i+1}: {chunk['content']}")
            sources.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "relevance_score": 1 - chunk.get("distance", 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt for Gemini
        prompt = self._create_prompt(query, context)
        
        try:
            # Generate response using Gemini based on available API
            if self._use_new_api:
                # New google-genai library
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                response_text = response.text
            else:
                # Old google-generativeai library
                response = self.model.generate_content(prompt)
                response_text = response.text
            
            # Calculate confidence based on relevance scores
            confidence = sum(source["relevance_score"] for source in sources) / len(sources) if sources else 0
            
            return {
                "query": query,
                "response": response_text,
                "sources": sources,
                "confidence": confidence,
                "context_used": len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "query": query,
                "response": "I encountered an error while generating the response. Please try again.",
                "sources": sources,
                "confidence": 0.0
            }
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a comprehensive prompt for the LLM."""
        
        return f"""You are an expert data analyst helping users understand spreadsheet data through natural language queries.

Based on the following spreadsheet data, provide a clear, accurate, and helpful answer to the user's question.

Spreadsheet Data Context:
{context}

User Question: {query}

Instructions:
1. Provide a direct answer to the question based on the spreadsheet data
2. Include specific values, cell references, or formulas when relevant
3. Explain any calculations or relationships found in the data
4. If the data is insufficient to answer the question, clearly state this
5. Format the response in a clear, human-readable way
6. Use bullet points or numbered lists when appropriate

Answer:"""
    
    def get_available_data_summary(self) -> Dict[str, Any]:
        """Get a summary of available data in the vector store."""
        stats = self.vector_store.get_collection_stats()
        
        # Handle case where stats is empty or missing total_embeddings
        total_embeddings = stats.get("total_embeddings", 0)
        
        if total_embeddings == 0:
            return {
                "status": "empty",
                "message": "No spreadsheet data has been loaded yet. Please upload a spreadsheet first."
            }
        
        # Get a sample of chunks to understand the data structure
        sample_chunks = self.vector_store.list_all_chunks(limit=10)
        
        # Analyze chunk types
        chunk_types = {}
        sheets = set()
        
        for chunk in sample_chunks:
            chunk_type = chunk["metadata"].get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            if "sheet_name" in chunk["metadata"]:
                sheets.add(chunk["metadata"]["sheet_name"])
        
        return {
            "status": "loaded",
            "total_chunks": total_embeddings,
            "chunk_types": chunk_types,
            "available_sheets": list(sheets),
            "message": f"Loaded {total_embeddings} chunks from {len(sheets)} sheets"
        }