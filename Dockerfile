# ============================================================
# Dockerfile - Insight into RAG
# CPU-only image: torch CPU (~200MB thay vì 1GB CUDA)
# BAAI/bge-m3 được cache vào image để không download lại mỗi lần start
# ============================================================
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    g++ \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch CPU (nhẹ ~200MB) trước, sau đó cài requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download BAAI/bge-m3 vào image để container start không cần mạng
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" || true

# Copy toàn bộ mã nguồn
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
