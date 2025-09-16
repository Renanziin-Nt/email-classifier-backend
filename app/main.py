from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.ai_service import ai_service
from app.services.email_processor import email_processor
import logging
import io
import inspect
import uuid
import asyncio
from typing import Dict, Optional
from enum import Enum
import uvicorn
import time
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
request_timestamps = {}
app = FastAPI(title="Email Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('FRONTEND_URL')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600
)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING_TEXT = "extracting_text"
    CLASSIFYING = "classifying"
    GENERATING_RESPONSE = "generating_response"
    COMPLETED = "completed"
    FAILED = "failed"

class EmailRequest(BaseModel):
    text: str

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    current_step: str
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None

class EmailResponse(BaseModel):
    category: str
    suggested_response: str
    confidence: float
    processed_text: str
    original_length: int

jobs_storage: Dict[str, dict] = {}

async def update_job_status(job_id: str, status: JobStatus, progress: int, message: str, result: dict = None, error: str = None):
    if job_id in jobs_storage:
        jobs_storage[job_id].update({
            "status": status,
            "progress": progress,
            "current_step": message,
            "message": message,
            "result": result,
            "error": error
        })
        logger.info(f"📊 Job {job_id[:8]}: {status} - {message} ({progress}%)")

async def process_email_job(job_id: str, file_content: bytes = None, file_info: dict = None, text_content: str = None):
    try:
        logger.info(f"🚀 Iniciando job {job_id[:8]}")
        await update_job_status(job_id, JobStatus.PROCESSING, 10, "Iniciando processamento...")
        
        clean_text = ""
        
        if file_content is not None:
            await update_job_status(job_id, JobStatus.EXTRACTING_TEXT, 20, f"Extraindo texto do arquivo...")
            
            content_type = file_info.get("content_type", "").lower()
            filename = file_info.get("filename", "").lower()
            
            is_pdf = "pdf" in content_type or filename.endswith(".pdf")
            is_txt = content_type.startswith("text") or filename.endswith(".txt")
            
            if not (is_pdf or is_txt):
                raise Exception("Apenas arquivos .txt ou .pdf são permitidos")
            
            await asyncio.sleep(0.2)
            
            if is_pdf:
                extractor = getattr(email_processor, "extract_text_from_pdf")
            else:
                extractor = getattr(email_processor, "extract_text_from_txt")
            
            result = extractor(file_content)
            if inspect.isawaitable(result):
                clean_text = await result
            else:
                clean_text = result
                
            await update_job_status(job_id, JobStatus.EXTRACTING_TEXT, 40, f"Texto extraído: {len(clean_text)} caracteres")
        else:
            clean_text = text_content
            await update_job_status(job_id, JobStatus.PROCESSING, 30, f"Texto recebido: {len(clean_text)} caracteres")
        
        if not clean_text or len(clean_text.strip()) < 5:
            raise Exception("Texto muito curto ou vazio")
        
        await asyncio.sleep(0.2)
        
        await update_job_status(job_id, JobStatus.CLASSIFYING, 50, "Classificando email com IA...")
        
        async def _maybe_call(func, *args, **kwargs):
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                await asyncio.sleep(0.1)
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
        
        await update_job_status(job_id, JobStatus.CLASSIFYING, 60, "Processando com modelo de IA...")
        
        classification = await _maybe_call(ai_service.classify_email, clean_text)
        
        await update_job_status(job_id, JobStatus.CLASSIFYING, 70, "Classificação concluída!")
        
        if isinstance(classification, tuple) and len(classification) == 2:
            category, confidence = classification
        elif isinstance(classification, dict):
            category = classification.get("category")
            confidence = classification.get("confidence")
        else:
            category = classification
            confidence = None
        
        await update_job_status(job_id, JobStatus.GENERATING_RESPONSE, 80, "Gerando resposta sugerida...")
        
        await asyncio.sleep(0.1)
        
        suggested_response = await _maybe_call(ai_service.generate_response, category, clean_text)
        
        await update_job_status(job_id, JobStatus.GENERATING_RESPONSE, 95, "Finalizando processamento...")
        
        result = {
            "category": category,
            "suggested_response": suggested_response,
            "confidence": confidence,
            "processed_text": clean_text[:100] + "..." if len(clean_text) > 100 else clean_text,
            "original_length": len(clean_text)
        }
        
        await update_job_status(job_id, JobStatus.COMPLETED, 100, "Processamento concluído!", result=result)
        logger.info(f"✅ Job {job_id[:8]} concluído com sucesso!")
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"💥 Erro no job {job_id[:8]}: {error_msg}")
        await update_job_status(job_id, JobStatus.FAILED, 0, "Erro no processamento", error=error_msg)

@app.post("/classify-email", response_model=JobResponse)
async def classify_email(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    text: str = Form(None),
    request: EmailRequest = Body(None)
):
    try:
        job_id = str(uuid.uuid4())
        
        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0,
            "current_step": "Job criado",
            "message": "Processamento iniciado",
            "result": None,
            "error": None
        }
        
        file_content = None
        file_info = None
        text_content = None
        
        if file is not None:
            file_content = await file.read()
            file_info = {
                "filename": file.filename,
                "content_type": file.content_type
            }
        elif request is not None and getattr(request, "text", None):
            text_content = request.text.strip()
        elif text and text.strip():
            text_content = text.strip()
        else:
            raise HTTPException(status_code=400, detail="Forneça um arquivo ou texto para classificação")
        
        background_tasks.add_task(
            process_email_job,
            job_id=job_id,
            file_content=file_content,
            file_info=file_info,
            text_content=text_content
        )
        
        logger.info(f"🎯 Job {job_id[:8]} criado e adicionado à fila")
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job criado com sucesso. Use o job_id para verificar o status."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro ao criar job")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    endpoint = request.url.path
    
    if "/job-status/" in endpoint:
        current_time = time.time()
        key = f"{client_ip}:{endpoint}"
        
        if key in request_timestamps:
            last_request = request_timestamps[key]
            if current_time - last_request < 0.5:
                logger.warning(f"⚠️  Rate limit exceeded for {client_ip} on {endpoint}")
                
        request_timestamps[key] = current_time
    
    response = await call_next(request)
    return response

@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    job_data = jobs_storage[job_id]
    
    logger.info(f"📋 Status requisitado para job {job_id[:8]}: {job_data['status']} ({job_data['progress']}%)")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        current_step=job_data["current_step"],
        message=job_data["message"],
        result=job_data["result"],
        error=job_data["error"]
    )

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws/job-status/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job_data = jobs_storage.get(job_id)

            if job_data:
                await websocket.send_json(job_data)

                # Se o job terminou, encerrar a conexão
                if job_data["status"] in ["completed", "failed"]:
                    logger.info(f"🔌 Encerrando WS do job {job_id[:8]} - status final: {job_data['status']}")
                    break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"⚠️ Cliente desconectado do job {job_id[:8]}")
    except Exception as e:
        logger.error(f"❌ Erro no WS do job {job_id[:8]}: {e}")
    finally:
        await websocket.close()
        logger.info(f"✅ WebSocket fechado para job {job_id[:8]}")


@app.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    if job_id in jobs_storage:
        del jobs_storage[job_id]
        logger.info(f"🧹 Job {job_id[:8]} removido da memória")
        return {"message": "Job removido com sucesso"}
    else:
        logger.warning(f"❌ Job {job_id[:8]} não encontrado para remoção")
        raise HTTPException(status_code=404, detail="Job não encontrado")

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "600"
        }
    )

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Max-Age"] = "600"
    
    return response

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "active_jobs": len(jobs_storage),
        "jobs": [f"{k[:8]}..." for k in jobs_storage.keys()]
    }

@app.get("/")
async def root():
    return {
        "message": "Email Classifier API with Async Jobs",
        "version": "1.0.0",
        "endpoints": {
            "classify": "POST /classify-email",
            "job_status": "GET /job-status/{job_id}",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_delay=0.5,
        timeout_keep_alive=300
    )