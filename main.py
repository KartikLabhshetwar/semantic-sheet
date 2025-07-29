#!/usr/bin/env python3
"""
Semantic Spreadsheet Search - Main Entry Point

This script provides a command-line interface for the semantic spreadsheet search application.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ingestion.spreadsheet_reader import SpreadsheetReader
from embedding.sentence_transformers_embeddings import SentenceTransformersEmbeddingService
from vector_store.chroma_manager import ChromaManager
from query.rag_processor import RAGProcessor

def process_file(file_path: str):
    """Process a spreadsheet file and store embeddings."""
    print(f"📊 Processing file: {file_path}")
    
    try:
        # Read and chunk the spreadsheet
        reader = SpreadsheetReader(file_path)
        chunks = reader.process_file()
        
        print(f"✅ Extracted {len(chunks)} semantic chunks")
        
        # Generate embeddings
        embedding_service = SentenceTransformersEmbeddingService()
        embedded_chunks = embedding_service.embed_semantic_chunks(chunks)
        
        print(f"✅ Generated embeddings")
        
        # Store in vector database
        vector_store = ChromaManager()
        vector_store.clear_collection()
        vector_store.add_embeddings(embedded_chunks)
        
        print(f"✅ Stored embeddings in vector database")
        
        # Display collection stats
        stats = vector_store.get_collection_stats()
        print(f"📈 Collection stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

def query_data(query: str):
    """Query the processed data."""
    print(f"🔍 Querying: {query}")
    
    try:
        rag_processor = RAGProcessor()
        result = rag_processor.process_query(query)
        
        print("\n" + "="*50)
        print("💡 ANSWER")
        print("="*50)
        print(result["response"])
        print("="*50)
        
        if result["sources"]:
            print(f"\n📋 SOURCES ({len(result['sources'])})")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. {source['metadata'].get('chunk_type', 'Unknown')}:")
                print(f"   Content: {source['content'][:100]}...")
                print(f"   Relevance: {source['relevance_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Error querying data: {e}")
        sys.exit(1)

def interactive_mode():
    """Run in interactive query mode."""
    print("🤖 Semantic Spreadsheet Search - Interactive Mode")
    print("Type 'exit' to quit\n")
    
    try:
        rag_processor = RAGProcessor()
        
        while True:
            query = input("💬 Ask a question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                result = rag_processor.process_query(query)
                
                print("\n" + "="*50)
                print("💡 ANSWER")
                print("="*50)
                print(result["response"])
                print("="*50)
                
                if result["confidence"] > 0:
                    print(f"Confidence: {result['confidence']:.1%}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing: {e}")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Semantic Spreadsheet Search")
    parser.add_argument("file", nargs="?", help="Excel file to process")
    parser.add_argument("-q", "--query", help="Query to run on processed data")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit web interface")
    
    args = parser.parse_args()
    
    if args.streamlit:
        print("🚀 Launching Streamlit web interface...")
        os.system("streamlit run src/ui/app.py")
        return
    
    if args.file:
        process_file(args.file)
        
        if args.query:
            query_data(args.query)
    elif args.interactive:
        interactive_mode()
    elif args.query:
        query_data(args.query)
    else:
        print("🤖 Semantic Spreadsheet Search")
        print("\nUsage:")
        print("  python main.py <file.xlsx>           # Process file")
        print('  python main.py <file.xlsx> -q "query" # Process and query')
        print('  python main.py -q "query"             # Query existing data')
        print("  python main.py -i                    # Interactive mode")
        print("  python main.py --streamlit           # Launch web interface")

if __name__ == "__main__":
    main()