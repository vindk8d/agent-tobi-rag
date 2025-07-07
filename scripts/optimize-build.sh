#!/bin/bash
# Build optimization script for RAG-Tobi

set -e

echo "🚀 Optimizing Docker builds for RAG-Tobi..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Clean up old containers and images
print_status "Cleaning up old containers and images..."
docker-compose down --remove-orphans 2>/dev/null || true
docker system prune -f

# Remove unused images to save space
print_status "Removing unused images..."
docker image prune -f

# Build with BuildKit for better caching
print_status "Enabling Docker BuildKit for faster builds..."
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build images with better caching
print_status "Building images with optimization..."
docker-compose build --parallel --progress=plain

# Verify build success
if [ $? -eq 0 ]; then
    print_success "Build completed successfully!"
    
    # Show image sizes
    print_status "Image sizes:"
    docker images | grep -E "(rag-tobi|REPOSITORY)"
    
    # Show build cache usage
    print_status "Build cache usage:"
    docker system df
    
    echo ""
    print_success "✅ Build optimization complete!"
    print_status "You can now start the services with: docker-compose up -d"
else
    print_error "Build failed. Please check the errors above."
    exit 1
fi 