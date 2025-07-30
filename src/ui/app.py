import streamlit as st
import os
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))

from ingestion.spreadsheet_reader import SpreadsheetReader
from embedding.sentence_transformers_embeddings import SentenceTransformersEmbeddingService
from vector_store.chroma_manager import ChromaManager
from query.rag_processor import RAGProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Semantic Spreadsheet Search",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .query-section {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .source-item {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_processor' not in st.session_state:
        try:
            st.session_state.rag_processor = RAGProcessor()
            st.session_state.initialized = True
        except ValueError as e:
            st.error(f"❌ Initialization Error: {e}")
            st.info("Please check your .env file and ensure GOOGLE_API_KEY is set correctly.")
            st.session_state.initialized = False
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def process_uploaded_file(uploaded_file):
    """Process the uploaded Excel or CSV file."""
    if uploaded_file is None:
        return
    
    with st.spinner("📊 Processing file..."):
        try:
            file_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            with st.spinner("Reading file..."):
                reader = SpreadsheetReader(file_path)
                chunks = reader.process_file()
            
            if not chunks:
                st.error("❌ No data found in the file.")
                return
            
            st.info(f"📊 Found {len(chunks)} data chunks to process")
            
            with st.spinner("Generating embeddings in batches..."):
                embedding_service = SentenceTransformersEmbeddingService()
                embedded_chunks = embedding_service.embed_semantic_chunks(chunks, batch_size=32)
            
            with st.spinner("Storing embeddings in vector database..."):
                vector_store = ChromaManager()
                vector_store.clear_collection()
                vector_store.add_embeddings(embedded_chunks)
            
            st.session_state.processing_complete = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.chunk_count = len(embedded_chunks)
            
            st.success(f"✅ Successfully processed {len(embedded_chunks)} data points from {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"File processing error: {e}")

def display_data_summary():
    """Display summary of loaded data."""
    if not st.session_state.get('initialized', False):
        return
    
    try:
        rag_processor = st.session_state.rag_processor
        summary = rag_processor.get_available_data_summary()
        
        if summary["status"] == "loaded":
            st.sidebar.success("📊 Data Loaded")
            st.sidebar.write(f"**Total Chunks:** {summary['total_chunks']}")
            st.sidebar.write(f"**Available Sheets:** {', '.join(summary['available_sheets'])}")
            
            if summary["chunk_types"]:
                st.sidebar.write("**Data Types:**")
                for chunk_type, count in summary["chunk_types"].items():
                    st.sidebar.write(f"- {chunk_type}: {count}")
        else:
            st.sidebar.info(summary["message"])
            
    except Exception as e:
        st.sidebar.error(f"Error loading data summary: {e}")

def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🔍 Semantic Spreadsheet Search</h1>', unsafe_allow_html=True)
    st.markdown("### Ask natural language questions about your Excel or CSV data")
    
    with st.sidebar:
        st.header("📁 Upload & Status")
        
        uploaded_file = st.file_uploader(
            "Choose an Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            key="file_uploader"
        )
        
        if uploaded_file is not None and uploaded_file != st.session_state.get('uploaded_file'):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.processing_complete = False
            process_uploaded_file(uploaded_file)
        
        display_data_summary()

        if st.button("🗑️ Clear All Data", use_container_width=True):
            try:
                vector_store = ChromaManager()
                vector_store.clear_collection()
                st.session_state.processing_complete = False
                st.session_state.uploaded_file = None
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing data: {e}")
    
    if not st.session_state.get('initialized', False):
        st.warning("⚠️ Application not initialized. Please check your configuration.")
        st.info("Make sure you have:")
        st.info("1. Set up your .env file with GOOGLE_API_KEY")
        st.info("2. Installed all required dependencies")
        return
    
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    st.subheader("💬 Ask a Question")
    
    example_questions = [
        "Find variance analysis data",
        "Show months that exceeded targets", 
        "Find profitability metrics",
        "What is the total revenue for Q3?",
        "Show me all formulas in the budget sheet",
        "Which products have the highest sales?",
        "What are the column headers in the data?",
        "Find any cells with negative values"
    ]
    
    selected_example = st.selectbox(
        "Try an example question:",
        [""] + example_questions,
        format_func=lambda x: "Select an example..." if x == "" else x
    )
    
    query = st.text_area(
        "Or type your own question:",
        value=selected_example if selected_example else "",
        placeholder="Ask any question about your Excel or CSV data...",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, disabled=not query)
    with col3:
        advanced_options = st.checkbox("Advanced")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if search_button and query:
        if not st.session_state.get('processing_complete', False):
            st.warning("⚠️ Please upload and process a file first.")
            return
        
        with st.spinner("🔍 Searching and generating response..."):
            try:
                top_k = 8
                if advanced_options:
                    top_k = st.slider("Number of results to consider:", 1, 10, 8)


                rag_processor = st.session_state.rag_processor
                result = rag_processor.process_query(query, top_k=top_k)
                

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("💡 Answer")
                st.write(result["response"])
                

                if result["confidence"] > 0:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)

                if result["sources"]:
                    st.subheader("📋 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i} - {source['metadata'].get('chunk_type', 'Unknown')}"):
                            st.write(f"**Content:** {source['content']}")

                            st.write("**Metadata:**")
                            for key, value in source["metadata"].items():
                                if key != "chunk_type":
                                    st.write(f"- {key}: {value}")

                            st.write(f"**Relevance:** {source['relevance_score']:.2f}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"Query processing error: {e}")
    

    st.markdown("---")
    st.markdown("Built using Streamlit, ChromaDB, and Google Gemini")

if __name__ == "__main__":
    main()