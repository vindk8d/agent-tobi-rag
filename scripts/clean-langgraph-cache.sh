#!/bin/bash

# LangGraph Cache Cleanup Script
# Use this when you encounter checkpoint compatibility issues

echo "🧹 Cleaning LangGraph Studio cache..."

# Remove all checkpoint files
if [ -d ".langgraph_api" ]; then
    echo "Removing .langgraph_api/*.pckl files..."
    rm -rf .langgraph_api/*.pckl
    echo "✅ Checkpoint files removed"
else
    echo "ℹ️ .langgraph_api directory not found"
fi

# Remove any other potential cache files
echo "Checking for other cache files..."
find . -name "*.pckl" -type f -exec rm {} \; 2>/dev/null || true

echo "🎉 Cache cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Restart LangGraph Studio"
echo "2. If you still see errors, run: pip install -r requirements.txt --upgrade" 