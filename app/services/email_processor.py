import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import io
import re
import logging

logger = logging.getLogger(__name__)

# Download recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class EmailProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('portuguese') + stopwords.words('english'))

    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extrai texto de PDF"""
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                
                return "\n".join(text_parts) if text_parts else ""
                
        except Exception as e:
            logger.error(f"Erro na extração de PDF: {e}")
            raise ValueError(f"Erro ao processar PDF: {str(e)}")

    def extract_text_from_txt(self, content: bytes) -> str:
        """Extrai texto de arquivo TXT"""
        try:
            return content.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            logger.error(f"Erro na leitura de TXT: {e}")
            raise ValueError(f"Erro ao ler arquivo texto: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Limpeza básica de texto"""
        if not text:
            return ""
        
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# Instância global
email_processor = EmailProcessor()