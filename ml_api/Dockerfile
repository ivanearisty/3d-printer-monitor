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

# Download default model weights (Obico YOLOv4) unless a local model is provided.
# To use a custom-trained YOLO11 model, place model-weights.onnx in ml_api/model/
# before building, and the download will be skipped.
RUN mkdir -p /model_cache/ml_api/onnx && \
    if [ -f /app/model/model-weights.onnx ]; then \
        echo 'Using local model-weights.onnx from ml_api/model/'; \
        cp /app/model/model-weights.onnx /model_cache/ml_api/onnx/model-weights.onnx; \
    else \
        echo 'No local model found. Downloading default Obico YOLOv4 model...'; \
        curl -o /model_cache/ml_api/onnx/model-weights.onnx \
        https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx; \
    fi

EXPOSE 3333

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:3333", "--workers", "1", "--timeout", "120", "server:app"]
