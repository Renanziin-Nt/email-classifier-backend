import logging
from typing import Tuple
import requests
from app.models.ml_model import MLModel
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

class EmailClassifier:
    def __init__(self):
        self.ml_model = MLModel()
        self.hf_api_key = os.getenv('HF_API_KEY')
        self.hf_api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    
    async def classify(self, text: str) -> Tuple[str, float]:
        """Classifica email usando estratégia hierárquica"""
        try:
            # 1. Tentar Hugging Face API
            if self.hf_api_key:
                result = await self._classify_with_hf(text)
                if result:
                    return result
            
            # 2. Usar modelo ML local
            return await self._classify_with_ml(text)
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            return await self._fallback_classification(text), 0.6
    
    async def _classify_with_hf(self, text: str):
        """Classificação com Hugging Face API"""
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            payload = {"inputs": text[:512], "parameters": {"wait_for_model": True}}
            
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()[0]
                if result['label'] in ['LABEL_0', 'LABEL_1']:
                    return "Produtivo", result['score']
                else:
                    return "Improdutivo", result['score']
        except Exception as e:
            logger.warning(f"HF API falhou: {e}")
        return None
    
    async def _classify_with_ml(self, text: str) -> Tuple[str, float]:
        """Classificação com modelo ML local"""
        try:
            # Tenta carregar modelo salvo primeiro
            if not self.ml_model.is_trained:
                if not self.ml_model.load():
                    self.ml_model.train()
                    self.ml_model.save()
            
            return self.ml_model.predict(text)
            
        except Exception as e:
            logger.error(f"Erro ML local: {e}")
            raise
    
    async def _fallback_classification(self, text: str) -> str:
        """Classificação fallback heurística"""
        text_lower = text.lower()
        
        productive_words = ['problema', 'erro', 'solicitação', 'pedido', 'status', 'suporte', 'urgente']
        unproductive_words = ['obrigado', 'agradeço', 'parabéns', 'feliz', 'natal', 'cumprimentos']
        
        productive_count = sum(1 for word in productive_words if word in text_lower)
        unproductive_count = sum(1 for word in unproductive_words if word in text_lower)
        
        return "Produtivo" if productive_count > unproductive_count else "Improdutivo"