import asyncio
from app.services.ai_service import ProfessionalAIService

async def test_ml():
    ai = ProfessionalAIService()
    
    # Teste com emails que NÃO ESTÃO no treinamento
    test_emails = [
        "Oi, preciso resetar minha senha porque esqueci",
        "Parabéns pelo aniversário da empresa!",
        "O sistema está dando erro 404 na página de login",
        "Só passando para agradecer a ajuda de vocês",
        "Como faço para solicitar um reembolso?"
    ]
    
    for email in test_emails:
        category, confidence = await ai.classify_email(email)
        print(f"📧 '{email}'")
        print(f"   → {category} ({confidence:.2f} de confiança)")
        print()

asyncio.run(test_ml())