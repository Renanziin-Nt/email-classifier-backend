import logging
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        self.gemini_available = False
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Configura Gemini API"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel(os.getenv('GEMINI_MODEL'))
                self.gemini_available = True
                logger.info("Gemini configurado com sucesso")
        except Exception as e:
            logger.warning(f"Gemini não disponível: {e}")
    
    async def generate_response(self, category: str, original_text: str) -> str:
        """Gera resposta para o email"""
        try:
            if self.gemini_available:
                return await self._generate_with_gemini(category, original_text)
            else:
                return self._generate_with_template(category, original_text)
                
        except Exception as e:
            logger.error(f"Erro na geração de resposta: {e}")
            return self._fallback_response(category)
    
    async def _generate_with_gemini(self, category: str, text: str) -> str:
        """Gera resposta usando Gemini API"""
        try:
            prompt = self._build_gemini_prompt(category, text)
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Erro no Gemini: {e}")
            return self._generate_with_template(category, text)
    
    def _build_gemini_prompt(self, category: str, text: str) -> str:
        """Constrói prompt para Gemini"""
        if category == "Produtivo":
            return f"""Gere resposta profissional para email de suporte:

            EMAIL: {text[:800]}

            DIRETRIZES:
            - Português formal brasileiro
            - Demonstrar empatia e compreensão
            - Oferecer solução ou encaminhamento claro
            - Incluir prazos realistas
            - Máximo 100 palavras

            RESPOSTA:"""
        else:
            return f"""Gere resposta cortês para mensagem social:

            EMAIL: {text[:800]}

            DIRETRIZES:
            - Português educado
            - Agradecimento genuíno
            - Brevidade com elegância
            - Máximo 50 palavras

            RESPOSTA:"""
    
    def _generate_with_template(self, category: str, text: str) -> str:
        """Gera resposta usando templates inteligentes"""
        text_lower = text.lower()
        
        templates = {
            "Produtivo": self._get_productive_template(text_lower),
            "Improdutivo": self._get_unproductive_template(text_lower)
        }
        
        return templates[category]
    
    def _get_productive_template(self, text_lower: str) -> str:
        """Template para emails produtivos"""
        if any(kw in text_lower for kw in ['pedido', 'solicitação', 'status']):
            return "Agradecemos seu contato. Seu pedido está em processamento. Retornaremos com atualizações em breve."
        elif any(kw in text_lower for kw in ['problema', 'erro', 'bug']):
            return "Obrigado por relatar o problema. Nossa equipe técnica foi acionada e retornará em breve."
        elif any(kw in text_lower for kw in ['pagamento', 'fatura', 'cobrança']):
            return "Agradecemos sua mensagem. Nossa equipe financeira analisará e retornará em até 24h."
        else:
            return "Agradecemos seu contato. Nossa equipe analisará sua solicitação e retornará em breve."
    
    def _get_unproductive_template(self, text_lower: str) -> str:
        """Template para emails improdutivos"""
        if any(kw in text_lower for kw in ['natal', 'ano novo']):
            return "Agradecemos as felicitações! Desejamos um excelente final de ano!"
        elif any(kw in text_lower for kw in ['obrigado', 'agradeço']):
            return "Obrigado pelo feedback! Ficamos felizes em ajudar."
        else:
            return "Agradecemos suas gentis palavras! Desejamos um ótimo dia."
    
    def _fallback_response(self, category: str) -> str:
        """Resposta fallback"""
        return "Agradecemos seu contato. Retornaremos em breve." if category == "Produtivo" else "Agradecemos suas gentis palavras."