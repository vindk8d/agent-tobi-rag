# Multi-stage Dockerfile optimized for WeasyPrint on Railway
FROM python:3.12-slim as base

# Install system dependencies for WeasyPrint
RUN apt-get update && apt-get install -y \
    # Build tools
    gcc g++ make pkg-config \
    # WeasyPrint core dependencies
    libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev \
    libffi-dev shared-mime-info \
    # GObject and GLib
    libglib2.0-dev libgobject-introspection-dev \
    # Text rendering
    libharfbuzz-dev libfribidi-dev \
    # Font support
    fontconfig libfontconfig1-dev libfreetype6-dev \
    # Image formats
    libjpeg-dev libpng-dev zlib1g-dev \
    # XML processing
    libxml2-dev libxslt1-dev \
    # Additional libraries
    libgirepository1.0-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health', timeout=10)" || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE $PORT

# Start command
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1