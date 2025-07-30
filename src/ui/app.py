import streamlit as st
import os
import sys
import shutil
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
        font-weight: 600;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .query-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #1f77b4;
    }
    .source-item {
        background: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Clean up spacing */
    .element-container {
        margin-bottom: 1rem;
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
            st.success("📊 Data Successfully Loaded")

            # Create a nice info box
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            """, unsafe_allow_html=True)

            st.metric("Total Data Chunks", summary['total_chunks'])
            st.write(f"**📝 Available Sheets:** {', '.join(summary['available_sheets'])}")

            if summary["chunk_types"]:
                st.write("**📊 Data Types:**")
                for chunk_type, count in summary["chunk_types"].items():
                    st.write(f"• {chunk_type.replace('_', ' ').title()}: {count}")

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("📁 " + summary["message"])
            st.markdown("""
                <div style="background: #4facfe; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <small>💡 <strong>Tip:</strong> Upload an Excel or CSV file to start asking questions!</small>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error loading data summary: {e}")

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">Semantic Sheet</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Ask natural language questions about your Excel or CSV data</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Upload & Status")

        uploaded_file = st.file_uploader(
            "Choose an Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            key=f"file_uploader_{st.session_state.get('uploader_key', 0)}",
            help="Upload your Excel or CSV file to start asking questions"
        )

        if uploaded_file is not None:
            # Check if this is a new file or if we need to process it
            current_file_info = f"{uploaded_file.name}_{uploaded_file.size}"
            stored_file_info = st.session_state.get('current_file_info', '')
            
            if current_file_info != stored_file_info and not st.session_state.get('clearing_data', False):
                st.session_state.uploaded_file = uploaded_file
                st.session_state.current_file_info = current_file_info
                st.session_state.processing_complete = False
                process_uploaded_file(uploaded_file)

        display_data_summary()

        # Clear data section with confirmation
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        if st.session_state.get('processing_complete', False):
            st.warning("⚠️ This will permanently delete all uploaded data and embeddings.")
            if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
                try:
                    # Clear vector store
                    vector_store = ChromaManager()
                    vector_store.clear_collection()
                    
                    # Reset all session state variables
                    st.session_state.processing_complete = False
                    st.session_state.uploaded_file = None
                    st.session_state.current_file_info = ''
                    if 'file_name' in st.session_state:
                        del st.session_state.file_name
                    if 'chunk_count' in st.session_state:
                        del st.session_state.chunk_count
                    
                    # Increment uploader key to force widget reset
                    st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
                    
                    # Clear any uploaded files from data directory
                    if os.path.exists("data"):
                        shutil.rmtree("data")
                    os.makedirs("data", exist_ok=True)
                    
                    st.success("✅ All data cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
                    logger.error(f"Clear data error: {e}")
        else:
            st.info("💡 No data to clear. Upload a file to get started.")

    # Main content
    if not st.session_state.get('initialized', False):
        st.error("⚠️ Application not initialized. Please check your configuration.")
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **To get started:**
            1. Create a `.env` file in the project root
            2. Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`  
            3. Install dependencies: `pip install -r requirements.txt`
            4. Restart the application
            """)
        return

    # Query Section
    st.markdown("### 💬 Ask a Question")

    # Example questions in an organized way
    col1, col2 = st.columns(2)

    with col1:
        business_questions = [
            "Find variance analysis data",
            "Show months that exceeded targets",
            "Find profitability metrics",
            "Which person performed poorly in sales?"
        ]

    with col2:
        data_questions = [
            "What is the total revenue for Q3?",
            "Show me all formulas in the budget sheet",
            "Which products have the highest sales?",
            "Find any cells with negative values"
        ]

    all_questions = business_questions + data_questions

    selected_example = st.selectbox(
        "Choose an example question to get started:",
        [""] + all_questions,
        format_func=lambda x: "Select an example..." if x == "" else x,
        index=0
    )

    query = st.text_area(
        "Or type your own question:",
        value=selected_example if selected_example else "",
        placeholder="Ask any question about your Excel or CSV data...",
        height=120,
        help="Try questions like: 'Find variance analysis data' or 'Show top performing sales reps'"
    )

    # Search button - centered and prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "🔍 Search",
            use_container_width=True,
            disabled=not query or not st.session_state.get('processing_complete', False),
            type="primary"
        )

    if search_button and query:
        if not st.session_state.get('processing_complete', False):
            st.warning("⚠️ Please upload and process a file first.")
            return

        with st.spinner("🔍 Searching and generating response..."):
            try:
                rag_processor = st.session_state.rag_processor
                result = rag_processor.process_query(query, top_k=20)  # Use optimal chunk count

                # Display Results
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 💡 Answer")
                st.write(result["response"])
                
                # Confidence metric with custom styling
                if result["confidence"] > 0:
                    confidence_pct = f"{result['confidence']:.1%}"
                    confidence_color = "#28a745" if result["confidence"] > 0.8 else "#ffc107" if result["confidence"] > 0.6 else "#dc3545"
                    st.markdown(f'''
                        <div class="metric-container" style="background: linear-gradient(135deg, {confidence_color} 0%, {confidence_color}aa 100%);">
                            <h3 style="margin: 0; color: white;">Confidence: {confidence_pct}</h3>
                            <p style="margin: 0; font-size: 0.9rem;">Based on {result.get("context_used", 0)} data sources</p>
                        </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Sources section with better organization
                if result["sources"]:
                    st.markdown("### 📋 Data Sources")
                    st.markdown(f"*Found {len(result['sources'])} relevant data sources*")
                    
                    # Group sources by type if possible
                    for i, source in enumerate(result["sources"], 1):
                        chunk_type = source['metadata'].get('chunk_type', 'Unknown')
                        sheet_name = source['metadata'].get('sheet_name', 'Unknown Sheet')
                        
                        with st.expander(f"📊 Source {i}: {chunk_type.replace('_', ' ').title()} from {sheet_name}"):
                            st.markdown("**Content:**")
                            st.code(source['content'], language="text")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Metadata:**")
                                for key, value in source["metadata"].items():
                                    if key not in ["chunk_type", "sheet_name"]:
                                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                            
                            with col2:
                                relevance = source['relevance_score']
                                st.metric("Relevance Score", f"{relevance:.2f}", f"{relevance*100:.0f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"Query processing error: {e}")
                with st.expander("🔧 Troubleshooting"):
                    st.markdown("""
                    **Common issues:**
                    - Make sure your file was uploaded and processed successfully
                    - Try a simpler question first
                    - Check that your data contains the information you're looking for  
                    """)
    
    # Footer with enhanced styling
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>Built with <strong>Streamlit</strong>, <strong>ChromaDB</strong>, and <strong>Google Gemini</strong></p>
            <p>Semantic search powered by advanced RAG architecture</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()