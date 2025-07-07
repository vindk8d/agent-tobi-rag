#!/bin/bash

# RAG-Tobi Development Environment Setup Script

set -e

echo "🚀 Setting up RAG-Tobi development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f env-template.txt ]; then
        cp env-template.txt .env
        echo "✅ .env file created from template. Please update it with your actual values."
    else
        echo "❌ env-template.txt not found. Please create your .env file manually."
        exit 1
    fi
else
    echo "✅ .env file already exists"
fi

# Create docker-compose.override.yml if it doesn't exist
if [ ! -f docker-compose.override.yml ]; then
    if [ -f docker-compose.override.yml.example ]; then
        echo "📝 Creating docker-compose.override.yml from example..."
        cp docker-compose.override.yml.example docker-compose.override.yml
        echo "✅ docker-compose.override.yml created. Customize as needed."
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p uploads

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting development environment..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

# Health checks
echo "🔍 Checking service health..."

# Check backend
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend might not be ready yet. Check logs with: docker-compose logs backend"
fi

# Check frontend
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend is accessible"
else
    echo "⚠️  Frontend might not be ready yet. Check logs with: docker-compose logs frontend"
fi

# Check Redis
if docker-compose exec redis redis-cli ping &> /dev/null; then
    echo "✅ Redis is healthy"
else
    echo "⚠️  Redis might not be ready yet. Check logs with: docker-compose logs redis"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Available services:"
echo "   - Backend API: http://localhost:8000"
echo "   - Frontend: http://localhost:3000"
echo "   - Redis: localhost:6379"
echo "   - pgAdmin (optional): http://localhost:5050"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f [service]"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart [service]"
echo "   - Rebuild services: docker-compose build [service]"
echo "   - Run tools: docker-compose --profile tools up -d"
echo ""
echo "📖 Don't forget to:"
echo "   1. Update your .env file with actual API keys"
echo "   2. Run database migrations if needed"
echo "   3. Install frontend dependencies if developing locally" 