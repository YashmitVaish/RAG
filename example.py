from rag_manager import RAGManager

def main():
    # Initialize RAG manager
    rag = RAGManager()
    
    # Example legal text
    legal_text = """
    The Competition Act, 2002 is an Act of the Parliament of India which governs Indian competition law. 
    It replaced the archaic The Monopolies and Restrictive Trade Practices Act, 1969. 
    Under this legislation, the Competition Commission of India was established to prevent activities that have an adverse effect on competition in India.
    """
    
    # Add document to the system
    rag.add_documents([legal_text], [{"source": "competition_act", "type": "legislation"}])
    
    # Process a query
    query = "What is the purpose of the Competition Act?"
    response = rag.process_query(query)
    
    print("Query:", query)
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
    
    # Analyze a legal document
    analysis = rag.analyze_legal_document(legal_text)
    print("\nDocument Analysis:")
    print(analysis)

if __name__ == "__main__":
    main() 