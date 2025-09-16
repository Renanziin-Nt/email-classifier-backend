FROM python:3.12-slim

WORKDIR /app

# instalar dependências do sistema necessárias para pdfplumber e amigos
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# atualizar pip e instalar numpy / scikit-learn primeiro via wheels oficiais
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir numpy==1.24.3 scikit-learn==1.3.2

# agora instalar o resto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "2", "--threads", "8"]
