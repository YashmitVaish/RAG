from typing import List, Dict, Any
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

class EmbeddingManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embedding model instance."""
        return self.embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # This is specific to the model being used
        return 768  # for all-mpnet-base-v2 