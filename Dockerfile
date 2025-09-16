FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema
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

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir numpy==1.24.3 scikit-learn==1.3.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expor a porta (Railway vai usar a variável PORT)
EXPOSE $PORT

# Comando corrigido - usando a variável PORT do ambiente
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --workers 2 --threads 8"]