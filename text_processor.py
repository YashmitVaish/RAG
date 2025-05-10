from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from .config import TEXT_SPLITTER_CONFIG

class TextProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            separators=TEXT_SPLITTER_CONFIG["separators"]
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.text_splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def create_document(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """Create a Document object from text and metadata."""
        return Document(
            page_content=text,
            metadata=metadata or {}
        )
    
    def process_legal_text(self, text: str) -> List[Document]:
        """Process legal text into document chunks."""
        chunks = self.split_text(text)
        return [
            self.create_document(
                chunk,
                metadata={"source": "legal_text", "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Add text cleaning logic here
        # This is a placeholder implementation
        return text.strip()
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Implement key phrase extraction
        # This is a placeholder implementation
        return [] 