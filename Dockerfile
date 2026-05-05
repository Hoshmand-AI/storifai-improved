FROM python:3.11-slim

WORKDIR /app

# Install numpy first to avoid version conflicts
RUN pip install --no-cache-dir "numpy<2"

# Install PyTorch CPU
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    google-cloud-storage \
    transformers \
    ftfy \
    regex

# Copy application code
COPY . .

ENV PORT=8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
