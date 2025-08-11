#!/bin/bash

# Railway Deployment Script for RAG-Tobi
# This script helps deploy the application to Railway with proper checks

set -e  # Exit on any error

echo "🚀 Railway Deployment Script for RAG-Tobi"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if Railway CLI is installed
check_railway_cli() {
    print_status "Checking Railway CLI installation..."
    if ! command -v railway &> /dev/null; then
        print_error "Railway CLI is not installed!"
        echo "Install it with: npm install -g @railway/cli"
        exit 1
    fi
    print_success "Railway CLI is installed"
}

# Check if user is logged in
check_railway_auth() {
    print_status "Checking Railway authentication..."
    if ! railway whoami &> /dev/null; then
        print_error "Not logged in to Railway!"
        echo "Login with: railway login"
        exit 1
    fi
    print_success "Authenticated with Railway"
}

# Validate required files
check_required_files() {
    print_status "Checking required deployment files..."
    
    required_files=(
        "railway.json"
        "nixpacks.toml" 
        "Dockerfile"
        "requirements.txt"
        "backend/main.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required file missing: $file"
            exit 1
        fi
    done
    
    print_success "All required files present"
}

# Check environment variables template
check_env_template() {
    print_status "Checking environment configuration..."
    
    if [[ ! -f "env-template.txt" ]]; then
        print_warning "env-template.txt not found"
    else
        print_success "Environment template available"
    fi
    
    if [[ -f ".env" ]]; then
        print_warning ".env file found - make sure not to commit it!"
    fi
}

# Validate requirements.txt
check_requirements() {
    print_status "Validating requirements.txt..."
    
    # Check if requirements.txt has pinned versions
    if grep -q ">=" requirements.txt; then
        print_warning "Found unpinned versions in requirements.txt"
        print_warning "Consider pinning versions for stable deployments"
    else
        print_success "Requirements.txt has pinned versions"
    fi
}

# Pre-deployment checks
run_pre_deployment_checks() {
    print_status "Running pre-deployment checks..."
    
    # Check Python syntax
    if command -v python3 &> /dev/null; then
        if python3 -m py_compile backend/main.py; then
            print_success "Python syntax check passed"
        else
            print_error "Python syntax errors found!"
            exit 1
        fi
    fi
    
    # Check if git repo is clean (optional warning)
    if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        if [[ -n $(git status --porcelain) ]]; then
            print_warning "Git working directory is not clean"
            print_warning "Consider committing changes before deployment"
        else
            print_success "Git working directory is clean"
        fi
    fi
}

# Deploy to Railway
deploy_to_railway() {
    print_status "Deploying to Railway..."
    
    # Ask for confirmation
    read -p "Are you ready to deploy? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled by user"
        exit 0
    fi
    
    # Deploy
    if railway deploy; then
        print_success "Deployment initiated successfully!"
        print_status "Monitor deployment progress in Railway dashboard"
    else
        print_error "Deployment failed!"
        exit 1
    fi
}

# Show post-deployment information
show_post_deployment_info() {
    echo ""
    print_success "Deployment script completed!"
    echo ""
    print_status "Next steps:"
    echo "1. Monitor deployment in Railway dashboard"
    echo "2. Check health endpoint: https://tobi-backend-production.up.railway.app/health"
    echo "3. Verify all environment variables are set"
    echo "4. Test API endpoints"
    echo "5. Frontend URL: https://tobi-frontend-production.up.railway.app/"
    echo ""
    print_status "Useful commands:"
    echo "  railway logs        - View application logs"
    echo "  railway variables   - Manage environment variables"
    echo "  railway open        - Open Railway dashboard"
    echo "  railway shell       - SSH into running container"
}

# Main execution
main() {
    check_railway_cli
    check_railway_auth
    check_required_files
    check_env_template
    check_requirements
    run_pre_deployment_checks
    deploy_to_railway
    show_post_deployment_info
}

# Run main function
main "$@"
