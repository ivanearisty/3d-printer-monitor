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

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model weights
RUN mkdir -p /model_cache/ml_api/onnx && \
    echo 'Downloading failure detection AI model (ONNX format)...' && \
    curl -o /model_cache/ml_api/onnx/model-weights.onnx \
    https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx

EXPOSE 3333

CMD ["gunicorn", "--bind", "0.0.0.0:3333", "--workers", "1", "--timeout", "120", "server:app"]
