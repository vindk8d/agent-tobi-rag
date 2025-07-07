#!/bin/bash

# RAG-Tobi Development Environment Teardown Script

set -e

echo "🧹 Tearing down RAG-Tobi development environment..."

# Stop and remove containers
echo "⏹️  Stopping containers..."
docker-compose down

# Option to remove volumes
read -p "🗑️  Do you want to remove volumes (this will delete all data)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing volumes..."
    docker-compose down -v
    
    # Remove additional data directories
    read -p "🗑️  Do you want to remove local data directories (logs, uploads)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing local data directories..."
        rm -rf logs uploads data
    fi
fi

# Option to remove images
read -p "🗑️  Do you want to remove Docker images? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing Docker images..."
    docker-compose down --rmi all
fi

# Clean up dangling images and containers
echo "🧹 Cleaning up dangling Docker resources..."
docker system prune -f

echo ""
echo "✅ Development environment teardown complete!"
echo ""
echo "🔧 To restart the environment, run:"
echo "   ./scripts/dev-setup.sh" 