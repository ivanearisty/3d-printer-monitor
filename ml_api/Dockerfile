FROM python:3.11-slim

# Install system dependencies for OpenCV and model loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files from root
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY ml_api/ .

# Download model weights
RUN mkdir -p /model_cache/ml_api/onnx && \
    echo 'Downloading failure detection AI model (ONNX format)...' && \
    curl -o /model_cache/ml_api/onnx/model-weights.onnx \
    https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx

EXPOSE 3333

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:3333", "--workers", "1", "--timeout", "120", "server:app"]
