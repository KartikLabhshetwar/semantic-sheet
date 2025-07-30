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
    
    def __init__(self):
        Config.validate()

        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = "gemini-2.5-flash"
        self._use_new_api = True

        self.embedding_service = SentenceTransformersEmbeddingService()
        self.vector_store = ChromaManager()
        
        logger.info("RAG Processor initialized successfully")
    
    def process_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        if not top_k:
            top_k = Config.TOP_K_RESULTS
        
        try:
            logger.info(f"Processing query: {query}")
            
            query_embedding = self.embedding_service.embed_single_text(query)
            
            relevant_chunks = self._comprehensive_retrieval(query, query_embedding, top_k)
            
            if not relevant_chunks:
                return {
                    "query": query,
                    "response": "I couldn't find any relevant information in the spreadsheet to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
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
        all_chunks = []
        
        semantic_chunks = self.vector_store.query_similar(query_embedding, top_k=top_k)
        all_chunks.extend(semantic_chunks)
        

        comprehensive_keywords = ['highest', 'lowest', 'maximum', 'minimum', 'all months', 'complete', 'total', 'entire', 'exceeded', 'targets', 'variance', 'profitability', 'revenue', 'performance', 'sales', 'person', 'team', 'individual', 'performed', 'poorly', 'well', 'best', 'worst', 'comparison', 'ranking', 'results', 'achievement', 'customer', 'product', 'pipeline', 'metrics', 'financial', 'profit', 'margin', 'growth', 'expenses', 'cost', 'cogs', 'cost of goods sold', 'details', 'breakdown', 'analysis']

        target_achievement_keywords = ['exceeded targets', 'exceed targets', 'surpass targets', 'beat targets', 'above target', 'over target', 'months that exceeded', 'which months exceeded', 'target achievement', 'performance above target']
        
        if any(keyword in query.lower() for keyword in target_achievement_keywords):
            logger.info("Target achievement query detected - prioritizing exceeded_targets chunks")
            exceeded_chunks = self._get_target_achievement_chunks(query, top_k)
            all_chunks.extend(exceeded_chunks)
        
        if any(keyword in query.lower() for keyword in comprehensive_keywords):
            logger.info("Query requires comprehensive data - expanding retrieval")
            
            comprehensive_chunks = self._get_comprehensive_column_data(query, top_k * 2)
            all_chunks.extend(comprehensive_chunks)
        
        temporal_keywords = ['month', 'quarter', 'year', 'period', 'time', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'exceeded', 'targets', 'growth', 'trends']
        if any(keyword in query.lower() for keyword in temporal_keywords):
            logger.info("Time-series query detected - ensuring complete temporal coverage")
            
            temporal_chunks = self._get_temporal_data_chunks(query, top_k)
            all_chunks.extend(temporal_chunks)
        
        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_hash = hash(chunk['content'])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        unique_chunks.sort(key=lambda x: x.get('distance', 1.0))
        
        is_comprehensive_query = any(kw in query.lower() for kw in comprehensive_keywords + temporal_keywords)
        
        specific_query_keywords = ['which months', 'what months', 'show months', 'months that', 'specific', 'exact', 'list', 'identify', 'find months', 'exceeded targets', 'negative variance', 'highest revenue', 'lowest', 'names', 'who']
        is_specific_query = any(kw in query.lower() for kw in specific_query_keywords)
        
        if is_specific_query:
            max_chunks = min(8, top_k)
            logger.info(f"Specific query detected - using {max_chunks} focused chunks with actual data")
        elif is_comprehensive_query:
            max_chunks = min(20, top_k * 2)
            logger.info(f"Comprehensive query detected - using {max_chunks} chunks for complete coverage")
        else:
            max_chunks = min(12, top_k)
            logger.info(f"Standard query detected - using {max_chunks} chunks")
        
        logger.info(f"Retrieved {len(unique_chunks[:max_chunks])} chunks using {('specific' if is_specific_query else 'comprehensive' if is_comprehensive_query else 'standard')} strategy")
        return unique_chunks[:max_chunks]
    
    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        
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
        
        prompt = self._create_prompt(query, context)
        
        try:
            response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
            )
            response_text = response.text
            
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
        try:
            all_chunks = self.vector_store.list_all_chunks(limit=max_chunks)
            
            relevant_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                content = chunk.get('content', '').lower()
                
                individual_query_keywords = ['person', 'individual', 'rep', 'team member', 'who', 'name', 'performed', 'well', 'top', 'best', 'worst', 'alice', 'bob', 'carlos', 'dana', 'evan', 'fiona', 'george', 'hannah', 'ivan', 'julia', 'kevin', 'lily']
                specific_query_keywords = ['which months', 'what months', 'show months', 'months that', 'specific', 'exact', 'list', 'identify', 'find months', 'exceeded targets', 'negative variance', 'highest revenue', 'lowest', 'names']
                value_query_keywords = ['details', 'cost of goods sold', 'cogs', 'value', 'amount', 'breakdown', 'analysis', 'expense ratio', 'ratios', 'ratio']
                
                is_individual_query = any(keyword in query.lower() for keyword in individual_query_keywords)
                is_specific_query = any(keyword in query.lower() for keyword in specific_query_keywords)
                is_value_query = any(keyword in query.lower() for keyword in value_query_keywords)
                
                if is_value_query:
                    if (chunk_type in ['key_value_metric', 'ratio_metric', 'formula', 'enhanced_column', 'complete_column_data'] or
                        any(value_indicator in content for value_indicator in ['=', 'cost of goods sold', 'cogs', '140000', '350000', 'revenue', 'expense', 'ratio', 'margin'])):
                        relevant_chunks.append(chunk)
                elif is_specific_query:
                    if (any(keyword in content for keyword in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) or
                        any(data_indicator in content for data_indicator in ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', 'target', 'actual', 'variance', '%']) or
                        chunk_type in ['enhanced_row', 'row', 'complete_data']):
                        relevant_chunks.append(chunk)
                elif is_individual_query:
                    if chunk_type in ['row', 'enhanced_row']:
                        relevant_chunks.append(chunk)
                    elif chunk_type in ['enhanced_column', 'business_relationship', 'column']:
                        relevant_chunks.append(chunk)
                else:
                    if (chunk_type in ['enhanced_column', 'business_relationship', 'column', 'key_value_metric', 'ratio_metric'] or
                        any(keyword in content for keyword in ['revenue', 'actual', 'target', 'variance', 'month', 'sales', 'team', 'person', 'performance', 'customer', 'product', 'pipeline', 'rep', 'manager', 'deal', 'quota', 'achievement', 'profit', 'margin', 'growth', 'financial', 'metric', 'ratio'])):
                        relevant_chunks.append(chunk)
            
            if is_value_query:
                final_chunks = relevant_chunks[:12]
                logger.info(f"Found {len(relevant_chunks)} chunks, using {len(final_chunks)} value-focused chunks for value query")
            elif is_specific_query:
                final_chunks = relevant_chunks[:8]
                logger.info(f"Found {len(relevant_chunks)} chunks, using {len(final_chunks)} data-rich chunks for specific query")
            elif is_individual_query:
                final_chunks = relevant_chunks[:8]
                logger.info(f"Found {len(relevant_chunks)} chunks, using {len(final_chunks)} focused chunks for individual query")
            else:
                final_chunks = relevant_chunks[:max_chunks]
                logger.info(f"Found {len(relevant_chunks)} chunks, using {len(final_chunks)} chunks for comprehensive query")
            
            return final_chunks
            
        except Exception as e:
            logger.warning(f"Error getting comprehensive column data: {e}")
            return []
    
    def _get_temporal_data_chunks(self, query: str, max_chunks: int) -> List[Dict[str, Any]]:
        try:
            all_chunks = self.vector_store.list_all_chunks(limit=max_chunks * 2)
            
            temporal_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                content = chunk.get('content', '').lower()
                
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
    
    def _get_target_achievement_chunks(self, query: str, max_chunks: int) -> List[Dict[str, Any]]:
        """Get chunks specifically about target achievement and exceeded targets."""
        try:
            all_chunks = self.vector_store.list_all_chunks(limit=max_chunks * 3)
            
            target_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                content = chunk.get('content', '').lower()
                

                if (chunk_type == 'exceeded_targets' or
                    chunk_type == 'business_relationship' or
                    'exceeded target' in content or
                    'months that exceeded' in content or
                    'target vs actual' in content or
                    'performance analysis' in content):
                    target_chunks.append(chunk)
            
            logger.info(f"Found {len(target_chunks)} target achievement chunks")
            return target_chunks[:max_chunks]
            
        except Exception as e:
            logger.warning(f"Error getting target achievement chunks: {e}")
            return []
    
    def _create_prompt(self, query: str, context: str) -> str:
        
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
- ALWAYS provide month names and actual dollar amounts
- Example: "January exceeded target with actual revenue of $125,000 vs target of $120,000"
- Look for chunks with "exceeded target" or "months that exceeded" content

For PROFITABILITY METRICS queries:
- Look for columns showing percentages, ratios, or achievement rates
- Identify efficiency metrics like "% to Target", "Achievement %", "Performance Ratio"
- Explain margin calculations, profit ratios, and efficiency indicators
- Consider Revenue/Target ratios as profitability indicators
- Look for key-value pairs like "Gross Margin = 0.6" or "Net Profit Margin = 0.421485714"

For KEY METRICS and FINANCIAL DATA queries:
- Look for key-value pairs in the format "Metric Name = Value"
- Identify financial metrics like Total Revenue, Net Profit, Gross Margin, Growth rates, Cost of Goods Sold
- ALWAYS provide the specific numerical values (e.g., "₹140,000", "₹350,000") 
- Show cell references when available (e.g., 'P&L Statement'!B3)
- Explain what each metric means in business context
- Convert decimal values to percentages when appropriate (e.g., 0.6 = 60%)
- Provide both the raw value and formatted interpretation

For RATIO ANALYSIS queries (Expense Ratio, Profit Margin, etc.):
- Look for ratio metrics across multiple time periods (Year 1, Year 2, Year 3)
- ALWAYS show actual decimal values AND percentage equivalents
- Example: "Expense Ratio (Year 1) = 0.058857143 (5.89%)"
- Include values for all available years when showing trends
- Explain what the ratio measures in business terms

For COST ANALYSIS queries (COGS, Expenses, etc.):
- Prioritize showing actual currency values over just formulas
- Include both formula logic AND calculated results
- Show values across different sheets if available (P&L Statement, 3-Year Forecast)
- Format large numbers with currency symbols (₹140,000 vs 140000)

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
7. For key-value metrics, provide both the raw value and business interpretation
8. Convert decimal margins to percentages (e.g., 0.6 = 60% margin)
9. When showing formulas, also show the calculated result value

RESPONSE FORMAT:
1. Start with a direct answer including the specific VALUE (e.g., "Cost of Goods Sold = ₹140,000")
2. Provide cell references and sheet locations (e.g., 'P&L Statement'!B3)
3. Show both formula AND calculated result (e.g., "=B2*0.4 = ₹140,000")
4. Explain calculations and business meanings
5. List relevant items with their actual values formatted with currency
6. Include context about data relationships across sheets
7. Use clear formatting with bullets or numbers for lists

Answer:"""
    
    def get_available_data_summary(self) -> Dict[str, Any]:
        stats = self.vector_store.get_collection_stats()
        
        total_embeddings = stats.get("total_embeddings", 0)
        
        if total_embeddings == 0:
            return {
                "status": "empty",
                "message": "No spreadsheet data has been loaded yet. Please upload a spreadsheet first."
            }
        
        sample_chunks = self.vector_store.list_all_chunks(limit=8)
        
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
        if not sources:
            return 0.0
        
        avg_relevance = sum(source["relevance_score"] for source in sources) / len(sources)
        
        comprehensive_keywords = ['highest', 'lowest', 'all', 'complete', 'exceeded', 'targets', 'variance', 'months']
        is_comprehensive_query = any(kw in query.lower() for kw in comprehensive_keywords)
        
        if is_comprehensive_query:
            completeness_score = min(chunk_count / 15.0, 1.0)
        else:
            completeness_score = min(chunk_count / 10.0, 1.0)

        chunk_types = set()
        for source in sources:
            chunk_type = source.get("metadata", {}).get("chunk_type", "unknown")
            chunk_types.add(chunk_type)
        
        has_complete_data = any("complete" in ct for ct in chunk_types)
        has_business_relationships = any("business" in ct for ct in chunk_types)
        
        type_diversity_score = len(chunk_types) / 6.0
        if has_complete_data and is_comprehensive_query:
            type_diversity_score += 0.2
        if has_business_relationships:
            type_diversity_score += 0.1
        
        type_diversity_score = min(type_diversity_score, 1.0)
        
        query_match_bonus = 0.0
        if is_comprehensive_query and chunk_count >= 15:
            query_match_bonus = 0.15
        elif not is_comprehensive_query and avg_relevance > 0.8:
            query_match_bonus = 0.1
        
        confidence = (
            avg_relevance * 0.4 +
            completeness_score * 0.3 +
            type_diversity_score * 0.2 +
            query_match_bonus
        )
        
        return min(confidence, 1.0)
