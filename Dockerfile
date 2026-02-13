FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using pip (avoid external ghcr dependency on `uv`)
# Pip will read `pyproject.toml` and install declared dependencies.
RUN pip install --no-cache-dir .

# Copy application code
COPY main.py ./
COPY stream_analyzer/ ./stream_analyzer/

# Run main.py - restart policy in docker-compose handles crashes
CMD ["python", "main.py"]
