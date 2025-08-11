# Production Dockerfile for Railway Deployment
# This serves as a fallback if NIXPACKS fails
FROM python:3.11-slim-bookworm

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    ENVIRONMENT=production

# Set work directory
WORKDIR /app

# Install system dependencies required for PDF generation and other features
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 appuser

# Copy application code
COPY . .

# Ensure backend directory is in the right place
RUN if [ ! -d "/app/backend" ]; then \
        echo "Error: backend directory not found in expected location"; \
        ls -la /app/; \
        exit 1; \
    fi

# Set proper ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

# Expose port (Railway will set the PORT env var)
EXPOSE 8000

# Start the application
CMD cd backend && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
