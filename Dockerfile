FROM python:3.12-slim

WORKDIR /app

# instalar dependências do sistema necessárias para numpy, scikit-learn e pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# copiar requirements e instalar
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# copiar código do app
COPY . .

# expor a porta do FastAPI
EXPOSE 8000

# usar gunicorn + uvicorn workers no Railway
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "2", "--threads", "8"]
