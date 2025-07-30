import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    """Configuration management for the Semantic Spreadsheet application."""
    
    # Google API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")  # More powerful model
    
    # Application Settings
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "20"))  # Increased for better coverage
    
    # Validation
    @classmethod
    def validate(cls, require_google_api: bool = True) -> bool:
        """Validate that required configuration is present."""
        if require_google_api and not cls.GOOGLE_API_KEY and not cls.GEMINI_API_KEY:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for text generation. Please set it in your .env file.")
        return True
    
    @classmethod
    def get_chroma_path(cls) -> str:
        """Get absolute path for ChromaDB persistence."""
        return os.path.abspath(cls.CHROMA_PERSIST_DIRECTORY)