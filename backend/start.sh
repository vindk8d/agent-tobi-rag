#!/bin/bash
# Startup script for Railway with WeasyPrint library setup

echo "=== Setting up WeasyPrint libraries at runtime ==="

# Create directory for symlinks
mkdir -p /app/libs

# Find and symlink specific libraries WeasyPrint needs
echo "Creating library symlinks..."

# Find libgobject and create the symlink WeasyPrint expects
for lib in /nix/store/*/lib/libgobject-2.0.so*; do
    if [ -f "$lib" ]; then
        ln -sf "$lib" /app/libs/libgobject-2.0-0
        echo "Linked libgobject: $lib -> /app/libs/libgobject-2.0-0"
        break
    fi
done

# Find libcairo
for lib in /nix/store/*/lib/libcairo.so*; do
    if [ -f "$lib" ]; then
        ln -sf "$lib" /app/libs/libcairo.so.2
        echo "Linked libcairo: $lib -> /app/libs/libcairo.so.2"
        break
    fi
done

# Find libpango
for lib in /nix/store/*/lib/libpango-1.0.so*; do
    if [ -f "$lib" ]; then
        ln -sf "$lib" /app/libs/libpango-1.0.so.0
        echo "Linked libpango: $lib -> /app/libs/libpango-1.0.so.0"
        break
    fi
done

# Set library paths - put our symlink directory first
export LD_LIBRARY_PATH="/app/libs:$LD_LIBRARY_PATH"

# Also add all Nix store library paths
for dir in /nix/store/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$dir"
    fi
done

echo "LD_LIBRARY_PATH includes /app/libs with symlinks"

# List our symlinks
echo "=== Library symlinks created ==="
ls -la /app/libs/

# Test WeasyPrint import
echo "=== Testing WeasyPrint import ==="
python -c "import weasyprint; print('SUCCESS: WeasyPrint imported!')" 2>&1

# Start the application
echo "=== Starting application ==="
echo "PORT environment variable: $PORT"
echo "Will listen on port: ${PORT:-8000}"

# Start uvicorn with the PORT Railway provides (venv should already be activated)
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
