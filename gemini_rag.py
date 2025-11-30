import google.generativeai as genai
from typing import List, Dict

class MedicalRAG:
    """Medical Question Answering System using Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize the Medical RAG system"""
        if not api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def search_medical_sources(self, query: str, num_sources: int = 4) -> List[Dict]:
        """
        Search medical sources for information
        Returns mock sources since we're using Gemini's knowledge
        """
        sources = []
        medical_sites = [
            "PubMed (NCBI)",
            "Mayo Clinic",
            "WHO (World Health Organization)",
            "CDC (Centers for Disease Control)",
            "NIH (National Institutes of Health)",
            "WebMD",
            "Cleveland Clinic",
            "Johns Hopkins Medicine"
        ]
        
        for i, site in enumerate(medical_sites[:num_sources]):
            sources.append({
                'title': f"Medical Reference from {site}",
                'url': f"https://example.com/source{i+1}",
                'snippet': f"Relevant medical information about: {query}"
            })
        
        return sources
    
    def query(self, question: str, num_sources: int = 4) -> Dict:
        """
        Query the system with a medical question
        
        Args:
            question: Medical question to answer
            num_sources: Number of sources to cite (for display purposes)
        
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        try:
            # Create a detailed medical prompt
            prompt = f"""You are a medical information assistant. Provide an evidence-based answer to the following medical question.

Question: {question}

Instructions:
- Provide accurate, evidence-based medical information
- Structure your answer clearly
- Mention key medical facts and considerations
- Include disclaimers about consulting healthcare professionals when appropriate
- Be thorough but concise

Answer:"""
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Get sources (mock sources for demonstration)
            sources = self.search_medical_sources(question, num_sources)
            
            return {
                'answer': response.text,
                'sources': sources,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'status': 'error'
            }
    
    def is_configured(self) -> bool:
        """Check if the RAG system is properly configured"""
        try:
            # Test the API with a simple query
            self.model.generate_content("Test")
            return True
        except:
            return False
