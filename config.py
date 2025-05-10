from decouple import config
from typing import Dict, Any

# API Keys
PINECONE_API_KEY = config('PINECONE_API_KEY')
GROQ_API_KEY = config('GROQ_API_KEY')
OPENAI_API_KEY = config('OPENAI_API_KEY')

# Model Configurations
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3-8b-8192"

# Vector Store Configurations
VECTOR_STORE_CONFIG: Dict[str, Any] = {
    "index_name": "search-engine",
    "search_kwargs": {
        "k": 5,
        "score_threshold": 0.5
    }
}

# Text Processing Configurations
TEXT_SPLITTER_CONFIG: Dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
}

# Prompt Templates
LEGAL_ASSISTANT_PROMPT = """
You are an intelligent legal assistant specializing in commercial law and case analysis.
Use the provided context to answer the user's question accurately and professionally.

Context: {context}

Question: {input}

Guidelines:
1. Focus on legal principles and relevant statutes
2. Provide clear, concise answers
3. Include relevant citations when applicable
4. Maintain professional legal language
5. If uncertain, acknowledge limitations

Response:
"""

LEGAL_ANALYSIS_PROMPT = """
Analyze the following legal text and extract:
1. Key legal principles
2. Relevant statutes
3. Case precedents
4. Legal arguments
5. Potential outcomes

Text: {input}

Provide a structured analysis with:
1. Main legal issues
2. Applicable laws
3. Supporting arguments
4. Potential implications
""" 