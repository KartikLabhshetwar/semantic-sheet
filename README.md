# 🔍 Semantic Spreadsheet Search

A powerful Python application that enables natural language queries on Excel spreadsheets using Retrieval-Augmented Generation (RAG) and Large Language Models.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your spreadsheet data in plain English
- **Semantic Understanding**: Understands context, formulas, and relationships in your data
- **Multiple Query Modes**: CLI, interactive, and web interface
- **Rich Results**: Get human-readable answers with source references
- **Formula Analysis**: Understand and explain spreadsheet formulas
- **Multi-Sheet Support**: Process complex workbooks with multiple sheets

## 🏗️ Architecture

The application uses a Retrieval-Augmented Generation (RAG) architecture:

1. **Ingestion**: Reads Excel files and extracts semantic chunks
2. **Embedding**: Converts chunks to vector embeddings using Sentence Transformers
3. **Storage**: Stores embeddings in ChromaDB vector database  
4. **Retrieval**: Finds relevant chunks for user queries
5. **Generation**: Uses Google Gemini to synthesize human-readable answers

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or create the project
cd semantic-sheet

# Create conda environment
conda create --name semanticsheet python=3.10 pandas openpyxl streamlit google-genai
conda activate semanticsheet

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Copy `.env.example` to `.env`
2. Add your Google Gemini API key:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

### 3. Run the Application

#### Option A: Web Interface (Recommended)
```bash
python main.py --streamlit
# or
streamlit run src/ui/app.py
```

## 📊 Usage Examples

### Web Interface
1. Upload your Excel file (.xlsx)
2. Wait for processing to complete
3. Ask questions like:
   - "What is the total revenue for Q3?"
   - "Show me all formulas in the budget sheet"
   - "Which products have negative profit?"
   - "What are the column headers?"

### Command Line Examples
```bash
# Process financial data
python main.py examples/financial_data.xlsx

# Query specific information
python main.py -q "What is the net profit for 2023?"

# Interactive session
python main.py -i
💬 Ask a question: Show me all expense categories
```

## 🗂️ Project Structure

```
semantic-sheet/
├── src/
│   ├── core/
│   │   └── config.py          # Configuration management
│   ├── ingestion/
│   │   └── spreadsheet_reader.py # Excel file processing
│   ├── embedding/
│   │   └── embeddings.py       # Vector embedding generation
│   ├── vector_store/
│   │   └── chroma_manager.py   # ChromaDB operations
│   ├── query/
│   │   └── rag_processor.py    # RAG query processing
│   └── ui/
│       └── app.py             # Streamlit web interface
├── data/                      # Uploaded Excel files
├── examples/                  # Sample Excel files
├── tests/                     # Unit tests
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key (for generation only) | Required |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB storage path | `./chroma_db` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `MAX_CHUNK_SIZE` | Maximum text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Default results per query | `5` |

**Note**: The application uses local Sentence Transformers for embeddings (no API limits) and Google Gemini only for final answer generation.

### Advanced Configuration

For custom embedding models or different vector databases, modify the respective modules in `src/embedding/` and `src/vector_store/`.

## 🧪 Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding Custom Chunk Types

To add new semantic chunk types:

1. Modify `src/ingestion/spreadsheet_reader.py`
2. Add new chunk creation methods
3. Update the embedding pipeline in `src/embedding/embeddings.py`

### Custom Embedding Models

To use a different embedding model:

1. Update `EMBEDDING_MODEL` in `.env`
2. Ensure the model is compatible with Sentence Transformers
3. Popular alternatives:
   - `all-mpnet-base-v2` (higher quality, slower)
   - `all-distilroberta-v1` (balanced)

## 📈 Performance Tips

1. **Large Files**: For very large spreadsheets, consider:
   - Increasing `MAX_CHUNK_SIZE`
   - Using a more powerful embedding model
   - Processing in batches

2. **Query Optimization**:
   - Use specific questions for better results
   - Adjust `TOP_K_RESULTS` based on your needs
   - Clear old data before processing new files

3. **Memory Management**:
   - ChromaDB automatically manages memory
   - Clear collection when switching datasets

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   ValueError: GOOGLE_API_KEY is required
   ```
   Solution: Set `GOOGLE_API_KEY` in your `.env` file

2. **Import Error**:
   ```
   ModuleNotFoundError: No module named 'src...'
   ```
   Solution: Run from project root directory

3. **Excel File Error**:
   ```
   ValueError: Failed to load workbook
   ```
   Solution: Ensure file is a valid Excel format and not corrupted

4. **Memory Issues**:
   Solution: Reduce `MAX_CHUNK_SIZE` or use a smaller embedding model

5. **Rate Limit Error (429)**:
   ```
   429 RESOURCE_EXHAUSTED
   ```
   Solution: The application automatically handles rate limits with delays and caching. For large files:
   - Wait for processing to complete (it will resume)
   - Consider processing smaller files first
   - Check your Gemini API quota at [Google AI Studio](https://aistudio.google.com)

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for LLM capabilities
- Sentence Transformers for embedding models
- ChromaDB for vector storage
- Streamlit for the web interface
- OpenPyXL for Excel file processing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in debug mode
3. Open an issue on GitHub

---

**Happy querying! 🎉**