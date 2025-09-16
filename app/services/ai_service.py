import logging
from typing import Tuple
from app.services.classifier import EmailClassifier
from app.services.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.response_generator = ResponseGenerator()
    
    async def classify_email(self, text: str) -> Tuple[str, float]:
        """Classifica email (interface principal)"""
        return await self.classifier.classify(text)
    
    async def generate_response(self, category: str, original_text: str) -> str:
        """Gera resposta (interface principal)"""
        return await self.response_generator.generate_response(category, original_text)

# Instância global
ai_service = AIService()