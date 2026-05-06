FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 && \
    pip install -r requirements.txt

# Pre-download CLIP ViT-B/32 weights into the image so cold starts are fast.
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='/root/.cache/clip')"

COPY app.py inference.py ./
COPY model_code/ ./model_code/
COPY assets/ ./assets/

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
