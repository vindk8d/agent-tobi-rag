#!/bin/bash
# Simple start script that bypasses WeasyPrint issues
export PORT=${PORT:-8000}
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT