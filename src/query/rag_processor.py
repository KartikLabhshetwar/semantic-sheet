import logging
import json
from typing import List, Dict, Any, Optional
from core.config import Config
from embedding.sentence_transformers_embeddings import SentenceTransformersEmbeddingService
from vector_store.chroma_manager import ChromaManager
from google import genai
HAS_GOOGLE_GENAI = True

logger = logging.getLogger(__name__)

class RAGProcessor:
    """Processes user queries using Retrieval-Augmented Generation."""
    
    def __init__(self):
        """Initialize the RAG processor with all necessary components."""
        Config.validate()  # Ensure API key is available

        # New google-genai library
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = "gemini-2.5-flash"
        self._use_new_api = True

        # Initialize embedding and vector store services
        self.embedding_service = SentenceTransformersEmbeddingService()
        self.vector_store = ChromaManager()
        
        logger.info("RAG Processor initialized successfully")
    
    def process_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Process a natural language query and return enriched results with comprehensive data coverage."""
        if not top_k:
            top_k = Config.TOP_K_RESULTS
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Generate embedding for the query
            query_embedding = self.embedding_service.embed_single_text(query)
            
            # Step 2: Use multi-strategy retrieval for better coverage
            relevant_chunks = self._comprehensive_retrieval(query, query_embedding, top_k)
            
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
    
    def _comprehensive_retrieval(self, query: str, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Implement comprehensive retrieval strategy to ensure complete data coverage.
        This addresses the issue of missing data by using multiple retrieval approaches.
        """
        all_chunks = []
        
        # Strategy 1: Standard semantic similarity search
        semantic_chunks = self.vector_store.query_similar(query_embedding, top_k=top_k)
        all_chunks.extend(semantic_chunks)
        
        # Strategy 2: For queries about complete datasets, get comprehensive column data
        comprehensive_keywords = ['highest', 'lowest', 'maximum', 'minimum', 'all months', 'complete', 'total', 'entire', 'exceeded', 'targets', 'variance', 'profitability', 'revenue', 'performance', 'sales', 'person', 'team', 'individual', 'performed', 'poorly', 'well', 'best', 'worst', 'comparison', 'ranking', 'results', 'achievement', 'customer', 'product', 'pipeline']
        if any(keyword in query.lower() for keyword in comprehensive_keywords):
            logger.info("Query requires comprehensive data - expanding retrieval")
            
            # Get all chunks from relevant sheets that contain complete column data
            comprehensive_chunks = self._get_comprehensive_column_data(query, top_k * 2)
            all_chunks.extend(comprehensive_chunks)
        
        # Strategy 3: For time-series queries, ensure we get all time periods
        temporal_keywords = ['month', 'quarter', 'year', 'period', 'time', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'exceeded', 'targets', 'growth', 'trends']
        if any(keyword in query.lower() for keyword in temporal_keywords):
            logger.info("Time-series query detected - ensuring complete temporal coverage")
            
            # Get all row chunks that contain time-series data
            temporal_chunks = self._get_temporal_data_chunks(query, top_k)
            all_chunks.extend(temporal_chunks)
        
        # Remove duplicates while preserving order and relevance
        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_hash = hash(chunk['content'])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        # Sort by relevance (distance) and limit to reasonable size for context window
        unique_chunks.sort(key=lambda x: x.get('distance', 1.0))
        
        # For comprehensive queries, ensure we always get enough chunks
        is_comprehensive_query = any(kw in query.lower() for kw in comprehensive_keywords + temporal_keywords)
        max_chunks = min(top_k * 2, 25) if is_comprehensive_query else max(top_k, 12)  # Minimum 12 chunks for all queries
        
        logger.info(f"Retrieved {len(unique_chunks[:max_chunks])} chunks using comprehensive strategy")
        return unique_chunks[:max_chunks]
    
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
            # New google-genai library
            response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
            )
            response_text = response.text
            
            # Calculate confidence using a more meaningful approach
            confidence = self._calculate_meaningful_confidence(sources, query, len(relevant_chunks))
            
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
    
    def _get_comprehensive_column_data(self, query: str, max_chunks: int) -> List[Dict[str, Any]]:
        """
        Get comprehensive column data for queries that need complete datasets.
        This ensures we don't miss data due to chunking limitations.
        """
        try:
            # Get all chunks of type 'enhanced_column' and 'business_relationship'
            all_chunks = self.vector_store.list_all_chunks(limit=max_chunks)
            
            relevant_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                content = chunk.get('content', '').lower()
                
                # Include chunks that contain comprehensive column data
                if (chunk_type in ['enhanced_column', 'business_relationship', 'column'] or
                    any(keyword in content for keyword in ['revenue', 'actual', 'target', 'variance', 'month', 'sales', 'team', 'person', 'performance', 'customer', 'product', 'pipeline', 'rep', 'manager', 'deal', 'quota', 'achievement'])):
                    relevant_chunks.append(chunk)
            
            logger.info(f"Found {len(relevant_chunks)} comprehensive column chunks")
            return relevant_chunks[:max_chunks]
            
        except Exception as e:
            logger.warning(f"Error getting comprehensive column data: {e}")
            return []
    
    def _get_temporal_data_chunks(self, query: str, max_chunks: int) -> List[Dict[str, Any]]:
        """
        Get temporal data chunks to ensure complete time-series coverage.
        """
        try:
            # Get all chunks that contain temporal/row data
            all_chunks = self.vector_store.list_all_chunks(limit=max_chunks * 2)
            
            temporal_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                content = chunk.get('content', '').lower()
                
                # Include chunks that contain temporal data
                if (chunk_type in ['row', 'business_relationship'] or
                    any(month in content for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                      'july', 'august', 'september', 'october', 'november', 'december',
                                                      'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) or
                    any(keyword in content for keyword in ['2024', '2023', 'month', 'quarter', 'period'])):
                    temporal_chunks.append(chunk)
            
            logger.info(f"Found {len(temporal_chunks)} temporal data chunks")
            return temporal_chunks[:max_chunks]
            
        except Exception as e:
            logger.warning(f"Error getting temporal data chunks: {e}")
            return []
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a comprehensive prompt for the LLM."""
        
        return f"""You are an expert spreadsheet data analyst specializing in financial and business data analysis. You help users understand complex spreadsheet relationships, calculations, and metrics.

SPREADSHEET DATA CONTEXT:
{context}

USER QUESTION: {query}

ANALYSIS GUIDELINES:

For VARIANCE ANALYSIS queries:
- Look for columns with headers like "Variance", "Difference", "Gap", or formulas calculating differences
- Identify Target vs Actual comparisons (e.g., Target Revenue vs Actual Revenue)
- Explain positive/negative variances and their business meaning
- Include specific values and calculations

For TARGET PERFORMANCE queries:
- Identify months/periods where Actual > Target (positive variance)
- Look for percentage achievement columns (% to Target, Achievement Rate, etc.)
- List specific periods that exceeded targets with their values
- Calculate achievement rates when possible

For PROFITABILITY METRICS queries:
- Look for columns showing percentages, ratios, or achievement rates
- Identify efficiency metrics like "% to Target", "Achievement %", "Performance Ratio"
- Explain margin calculations, profit ratios, and efficiency indicators
- Consider Revenue/Target ratios as profitability indicators

For FORMULA analysis:
- Explain the business logic behind calculations
- Show how different columns relate to each other
- Identify key performance indicators and their formulas

DATA INTERPRETATION RULES:
1. Always examine column headers carefully to understand data relationships
2. Look for patterns in row and column data to identify business metrics
3. When you see Target and Actual columns, always look for Variance calculations
4. Percentage columns often represent profitability or efficiency metrics
5. Positive variances indicate exceeding targets; negative indicate shortfalls
6. Values over 100% in ratio columns indicate target exceeded

RESPONSE FORMAT:
1. Start with a direct answer to the question
2. Provide specific data points with cell references when available
3. Explain calculations and business meanings
4. List relevant items with their values
5. Include context about data relationships
6. Use clear formatting with bullets or numbers for lists

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
        sample_chunks = self.vector_store.list_all_chunks(limit=8)
        
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
    
    def _calculate_meaningful_confidence(self, sources: List[Dict[str, Any]], query: str, chunk_count: int) -> float:
        """
        Calculate a more meaningful confidence score based on multiple factors.
        """
        if not sources:
            return 0.0
        
        # Factor 1: Average relevance score (from vector similarity)
        avg_relevance = sum(source["relevance_score"] for source in sources) / len(sources)
        
        # Factor 2: Data completeness (more chunks = higher confidence for comprehensive queries)
        comprehensive_keywords = ['highest', 'lowest', 'all', 'complete', 'exceeded', 'targets', 'variance', 'months']
        is_comprehensive_query = any(kw in query.lower() for kw in comprehensive_keywords)
        
        if is_comprehensive_query:
            # For comprehensive queries, reward having more chunks
            completeness_score = min(chunk_count / 15.0, 1.0)  # Normalize to 1.0 at 15+ chunks
        else:
            # For specific queries, fewer but highly relevant chunks are better
            completeness_score = min(chunk_count / 10.0, 1.0)   # Normalize to 1.0 at 10+ chunks

        # Factor 3: Content type diversity (having multiple chunk types increases confidence)
        chunk_types = set()
        for source in sources:
            chunk_type = source.get("metadata", {}).get("chunk_type", "unknown")
            chunk_types.add(chunk_type)
        
        # Reward having complete data chunks for comprehensive queries
        has_complete_data = any("complete" in ct for ct in chunk_types)
        has_business_relationships = any("business" in ct for ct in chunk_types)
        
        type_diversity_score = len(chunk_types) / 6.0  # Assuming max 6 different chunk types
        if has_complete_data and is_comprehensive_query:
            type_diversity_score += 0.2  # Bonus for complete data chunks
        if has_business_relationships:
            type_diversity_score += 0.1  # Bonus for business relationship chunks
        
        type_diversity_score = min(type_diversity_score, 1.0)
        
        # Factor 4: Query-specific adjustments
        query_match_bonus = 0.0
        if is_comprehensive_query and chunk_count >= 15:
            query_match_bonus = 0.15  # Bonus for comprehensive queries with lots of data
        elif not is_comprehensive_query and avg_relevance > 0.8:
            query_match_bonus = 0.1   # Bonus for specific queries with high relevance
        
        # Combine factors with weights
        confidence = (
            avg_relevance * 0.4 +           # 40% from similarity scores
            completeness_score * 0.3 +      # 30% from data completeness  
            type_diversity_score * 0.2 +    # 20% from chunk type diversity
            query_match_bonus               # 10-15% bonus for query type match
        )
        
        return min(confidence, 1.0)  # Cap at 100%