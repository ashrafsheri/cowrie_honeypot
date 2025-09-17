# Multi-stage Docker build for LogBERT training
FROM python:3.9-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy source code
COPY --chown=app:app . .

# Training stage
FROM base AS training

# Set environment for training
ENV PYTHONPATH=/home/app
ENV PREFECT_API_URL=http://localhost:4200/api

# Install additional training dependencies
USER root
RUN pip install --no-cache-dir \
    prefect==2.14.0 \
    torch>=2.3.0 \
    transformers>=4.30.0 \
    datasets>=2.12.0 \
    accelerate>=0.20.0 \
    scikit-learn>=1.0.0 \
    drain3>=0.9.5 \
    pandas>=1.3.0 \
    numpy>=1.20.0 \
    pyyaml>=6.0.0

USER app

# Training entrypoint with MPS support
CMD ["python", "-m", "train_cowrie"]

# Inference stage  
FROM base AS inference

# Set environment for inference
ENV PYTHONPATH=/home/app
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

# Install inference dependencies
USER root
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    python-multipart>=0.0.6

USER app

# Expose inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Inference entrypoint
CMD ["uvicorn", "inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage (includes both training and inference dependencies)
FROM base AS development

USER root

# Install all dependencies for development
RUN pip install --no-cache-dir \
    prefect==2.14.0 \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    datasets>=2.12.0 \
    accelerate>=0.20.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    python-multipart>=0.0.6 \
    jupyter>=1.0.0 \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0

USER app

# Development entrypoint (bash for interactive development)
CMD ["/bin/bash"]