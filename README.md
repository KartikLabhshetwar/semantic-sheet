# Semantic-sheet

A production-ready RAG system that transforms spreadsheet data into queryable knowledge using natural language. Built for business analysts who need semantic understanding of financial data without SQL or pivot tables.

## Overview

Ask questions like *"Which months exceeded targets?"* or *"Find all profitability metrics"* and get intelligent answers with business context and confidence scores.

**Key Features:**
-  **Semantic Understanding**: Recognizes business concepts (revenue, margins, variance analysis)
-  **Multi-Strategy Retrieval**: Adaptive query processing with 85-95% confidence scores
-  **Business Intelligence**: Target vs actual analysis, performance metrics, time-series data
-  **Production Ready**: Streamlit web UI, ChromaDB persistence, batch processing

## Architecture

**Multi-Strategy RAG Pipeline:**

1. **Data Ingestion**: Extracts Excel/CSV with business context recognition
2. **Semantic Chunking**: Creates complete data chunks + business relationships  
3. **Vector Storage**: ChromaDB with `all-mpnet-base-v2` embeddings (768D)
4. **Query Classification**: Three-tier system (specific/individual/comprehensive)
5. **Response Generation**: Google Gemini 2.5 Flash with confidence scoring

**Performance:** 543 chunks processed in <2min, <3s query response, 85-95% confidence

## Quick Start

```bash
# Setup
git clone https://github.com/KartikLabhshetwar/semantic-sheet

# Create conda environment
conda create --name semanticsheet python=3.10 pandas openpyxl streamlit google-genai
conda activate semanticsheet

# Install additional dependencies
pip install -r requirements.txt

# Run
streamlit run src/ui/app.py
```

## Usage Examples

```python
# Business Queries That Work
"Show months that exceeded targets"           # → 7 specific months identified
"Which person performed poorly?"              # → Names with performance context  
"Find all profitability metrics"             # → Complete financial analysis
"What is the variance analysis data?"         # → Target vs actual with calculations
```

## Tech Stack

- **Embeddings**: sentence-transformers (`all-mpnet-base-v2`)
- **Vector DB**: ChromaDB with persistence
- **LLM**: Google Gemini 2.5 Flash  
- **Frontend**: Streamlit
- **Processing**: pandas, openpyxl

## Project Structure

```text
semantic-sheet/
├── src/
│   ├── core/config.py              # Configuration management
│   ├── ingestion/spreadsheet_reader.py  # Excel/CSV processing + business chunking  
│   ├── embedding/embeddings.py     # Vector embedding generation
│   ├── vector_store/chroma_manager.py   # ChromaDB operations
│   ├── query/rag_processor.py      # Multi-strategy RAG pipeline
│   └── ui/app.py                   # Streamlit interface
├── data/                           # Upload directory
├── main.py                         # CLI entry point
└── requirements.txt
```

## Configuration

**Environment Variables:**

| Variable | Description | Default | Notes |
|----------|-------------|---------|--------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required | For response generation only |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB storage path | `./chroma_db` | Persistent embeddings |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-mpnet-base-v2` | Upgraded for better accuracy |
| `MAX_CHUNK_SIZE` | Maximum text chunk size | `1000` | Optimized for business data |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` | Ensures context continuity |
| `TOP_K_RESULTS` | Base retrieval count | `20` | Adaptive: 12-25 based on query |


**Key Settings:**
- **Confidence Range**: 85-95% for business queries
- **Processing**: 32 chunks/batch, persistent ChromaDB storage
- **Memory**: Optimized for 500+ chunk datasets

## Development

**Core Architecture Decisions:**

```python
# Multi-strategy retrieval implementation
def _comprehensive_retrieval(self, query, query_embedding, top_k):
    # 1. Semantic similarity search
    # 2. Business context expansion  
    # 3. Temporal data coverage
    # 4. Deduplication and ranking
```

```python
# Multi-factor confidence scoring
confidence = (
    avg_relevance * 0.4 +           # Semantic similarity
    completeness_score * 0.3 +      # Data completeness
    type_diversity_score * 0.2 +    # Chunk diversity
    query_match_bonus               # Query-specific bonus
)
```

**Business Context Features:**

- Enhanced prompt engineering for financial terminology
- Variance analysis pattern recognition  
- Target vs actual relationship mapping
- Time-series data continuity validation



## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.