from typing import List, Dict, Any, Optional
from .vector_store import VectorStoreManager
from .llm_manager import LLMManager
from .text_processor import TextProcessor
from .embeddings import EmbeddingManager

class RAGManager:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
    
    def process_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a query using the RAG pipeline."""
        # Get retriever
        retriever = self.vector_store.get_retriever()
        
        # Create retrieval chain
        retrieval_chain = self.llm_manager.create_retrieval_chain(retriever)
        
        # Process the query
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": chat_history or []
        })
        
        return {
            "answer": response["answer"],
            "sources": [doc.metadata for doc in response["context"]]
        }
    
    def analyze_legal_document(self, text: str) -> Dict[str, Any]:
        """Analyze a legal document using the RAG pipeline."""
        # Process the text
        documents = self.text_processor.process_legal_text(text)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Analyze using LLM
        analysis = self.llm_manager.analyze_legal_text(text)
        
        return {
            "analysis": analysis,
            "document_chunks": len(documents)
        }
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """Add documents to the vector store."""
        documents = [
            self.text_processor.create_document(
                text,
                metadata=meta or {}
            )
            for text, meta in zip(texts, metadata or [{}] * len(texts))
        ]
        self.vector_store.add_documents(documents)
    
    def search_similar_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        documents = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store."""
        self.vector_store.delete_documents(ids)
    
    def get_document_analysis(self, text: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a document."""
        # Process text
        documents = self.text_processor.process_legal_text(text)
        
        # Get embeddings
        embeddings = self.embedding_manager.embed_documents(
            [doc.page_content for doc in documents]
        )
        
        # Analyze with LLM
        analysis = self.llm_manager.analyze_legal_text(text)
        
        return {
            "analysis": analysis,
            "embeddings": embeddings,
            "chunks": len(documents),
            "key_phrases": self.text_processor.extract_key_phrases(text)
        } 