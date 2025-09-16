from pydantic import BaseModel, Field
from typing import Optional

class EmailRequest(BaseModel):
    text: str

class EmailResponse(BaseModel):
    category: str
    suggested_response: str
    confidence: float = Field(..., ge=0, le=1)
    processed_text: str
    original_length: int

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Produtivo",
                "suggested_response": "Agradecemos seu contato...",
                "confidence": 0.85,
                "processed_text": "Preciso de ajuda com erro...",
                "original_length": 150
            }
        }