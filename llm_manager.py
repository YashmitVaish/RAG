from typing import List, Dict, Any, Optional
from langchain_groqimport ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .config import GROQ_API_KEY, LLM_MODEL, LEGAL_ASSISTANT_PROMPT, LEGAL_ANALYSIS_PROMPT

class LLMManager:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompts = self._initialize_prompts()
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the LLM model."""
        return ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.1  # Low temperature for more focused responses
        )
    
    def _initialize_prompts(self) -> Dict[str, ChatPromptTemplate]:
        
        return {
            "legal_assistant": ChatPromptTemplate.from_template(LEGAL_ASSISTANT_PROMPT),
            "legal_analysis": ChatPromptTemplate.from_template(LEGAL_ANALYSIS_PROMPT)
        }
    
    def create_document_chain(self, prompt_type: str = "legal_assistant"):
        
        return create_stuff_documents_chain(
            self.llm,
            self.prompts[prompt_type]
        )
    
    def create_retrieval_chain(self, retriever, prompt_type: str = "legal_assistant"):
     
        document_chain = self.create_document_chain(prompt_type)
        return create_retrieval_chain(retriever, document_chain)
    
    def generate_response(self, prompt: str) -> str:
   
        return self.llm.invoke(prompt).content
    
    def analyze_legal_text(self, text: str) -> Dict[str, Any]:
     
        prompt = self.prompts["legal_analysis"].format(input=text)
        response = self.generate_response(prompt)
        return self._parse_legal_analysis(response)
    
    def _parse_legal_analysis(self, response: str) -> Dict[str, Any]:
       
        # Implement parsing logic based on the expected response format
        # This is a placeholder implementation
        return {
            "analysis": response,
            "structured_data": {}  # Add structured parsing logic
        } 