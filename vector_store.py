from typing import List, Dict, Any, Optional
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from .config import PINECONE_API_KEY, VECTOR_STORE_CONFIG
from .embeddings import EmbeddingManager

class VectorStoreManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> PineconeVectorStore:
        """Initialize the vector store with embeddings."""
        return PineconeVectorStore(
            index_name=VECTOR_STORE_CONFIG["index_name"],
            embedding=self.embedding_manager.get_embeddings(),
            pinecone_api_key=PINECONE_API_KEY
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vector_store.add_documents(documents)
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Perform similarity search on the vector store."""
        search_kwargs = VECTOR_STORE_CONFIG["search_kwargs"].copy()
        if k is not None:
            search_kwargs["k"] = k
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
            
        return self.vector_store.similarity_search(
            query,
            **search_kwargs
        )
    
    def get_retriever(self):
        """Get a retriever instance for the vector store."""
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=VECTOR_STORE_CONFIG["search_kwargs"]
        )
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store."""
        self.vector_store.delete(ids) 