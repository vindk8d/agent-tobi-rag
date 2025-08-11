#!/bin/bash

# Railway Deployment Helper Script
# This script helps prepare the application for Railway deployment

set -e

echo "🚂 Preparing Agent Tobi RAG for Railway deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓ $1 exists${NC}"
    else
        echo -e "${RED}✗ $1 is missing${NC}"
        exit 1
    fi
}

# Function to check if environment variable is set
check_env() {
    if [ -z "${!1}" ]; then
        echo -e "${YELLOW}⚠ $1 is not set${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 is set${NC}"
        return 0
    fi
}

echo ""
echo "📋 Checking required files..."
check_file "railway.json"
check_file "backend/Dockerfile.prod"
check_file "frontend/Dockerfile.prod"
check_file "backend/.dockerignore"
check_file "frontend/.dockerignore"
check_file "backend/requirements.txt"
check_file "frontend/package.json"

echo ""
echo "🔧 Building production images locally (test)..."

# Test backend build
echo "Building backend..."
cd backend
docker build -f Dockerfile.prod -t agent-tobi-backend:prod . || {
    echo -e "${RED}Backend build failed!${NC}"
    exit 1
}
cd ..

# Test frontend build
echo "Building frontend..."
cd frontend
docker build -f Dockerfile.prod -t agent-tobi-frontend:prod . || {
    echo -e "${RED}Frontend build failed!${NC}"
    exit 1
}
cd ..

echo -e "${GREEN}✓ Local builds successful!${NC}"

echo ""
echo "📝 Deployment Checklist:"
echo "1. [ ] Create a new Railway project"
echo "2. [ ] Add Redis service from Railway dashboard"
echo "3. [ ] Create backend service:"
echo "   - Connect your GitHub repo"
echo "   - Set root directory: /backend"
echo "   - Set Dockerfile path: backend/Dockerfile.prod"
echo "   - Add environment variables from docs/deployment/railway-environment-variables.md"
echo "4. [ ] Create frontend service:"
echo "   - Connect your GitHub repo"
echo "   - Set root directory: /frontend"
echo "   - Set Dockerfile path: frontend/Dockerfile.prod"
echo "   - Add environment variables from docs/deployment/railway-environment-variables.md"
echo "5. [ ] Update CORS and API URLs after Railway assigns domains"
echo "6. [ ] Configure custom domains (optional)"

echo ""
echo "🔗 Useful Railway CLI commands:"
echo "  railway login          # Login to Railway"
echo "  railway link           # Link to existing project"
echo "  railway up             # Deploy current directory"
echo "  railway logs           # View logs"
echo "  railway variables      # Manage environment variables"

echo ""
echo -e "${GREEN}✅ Pre-deployment checks complete!${NC}"
echo "Ready to deploy to Railway. Follow the checklist above."