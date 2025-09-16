import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def check_available_models():
    """Verifica quais modelos estão disponíveis"""
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        print("🔍 Verificando modelos disponíveis...")
        models = genai.list_models()
        
        print("\n📋 MODELOS DISPONÍVEIS:")
        print("=" * 50)
        
        for model in models:
            if 'gemini' in model.name.lower():
                print(f"🔹 {model.name}")
                print(f"   Suporta generateContent: {'generateContent' in model.supported_generation_methods}")
                print()
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Erro ao verificar modelos: {e}")
        print("Verifique sua API key e conexão com a internet")

if __name__ == "__main__":
    check_available_models()