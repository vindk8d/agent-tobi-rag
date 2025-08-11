#!/bin/bash
# Startup script for Railway with WeasyPrint library setup

echo "=== Setting up WeasyPrint libraries at runtime ==="

# Find and link all required libraries
for dir in /nix/store/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done

# Also add standard locations
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Test WeasyPrint import
echo "=== Testing WeasyPrint import ==="
python -c "import weasyprint; print('WeasyPrint imported successfully!')" 2>&1 || echo "WeasyPrint import failed at runtime"

# Start the application
echo "=== Starting application ==="
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
