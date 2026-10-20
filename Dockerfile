# Use official lightweight Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system build dependencies (needed by some scientific Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, FastAPI app, and packaged model artifact
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# If the model artifact wasn't pre-built locally, train it during image build
RUN if [ ! -f models/heart_disease_pipeline.pkl ]; then python -m src.train; fi

# Expose FastAPI port
EXPOSE 8000

# Container healthcheck hitting FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
 