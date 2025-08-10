# Railway Deployment Documentation

This folder contains all documentation and resources needed to deploy Agent Tobi RAG to Railway.

## Files Overview

### 📚 Main Guide
- **[railway-deployment-guide.md](./railway-deployment-guide.md)** - Comprehensive deployment guide with architecture overview and step-by-step instructions

### 🔧 Configuration
- **[railway-environment-variables.md](./railway-environment-variables.md)** - Complete list of environment variables for backend and frontend services

### ✅ Deployment Process
- **[railway-deployment-checklist.md](./railway-deployment-checklist.md)** - Step-by-step checklist to ensure successful deployment

## Quick Start

1. **Prepare for deployment**:
   ```bash
   ./scripts/deploy-to-railway.sh
   ```

2. **Follow the deployment guide**: Start with `railway-deployment-guide.md`

3. **Use the checklist**: Track your progress with `railway-deployment-checklist.md`

4. **Configure environment**: Copy variables from `railway-environment-variables.md`

## Architecture Summary

```
Railway Cloud
├── Frontend Service (Next.js)
│   └── Port 3000
├── Backend Service (FastAPI)
│   └── Port 8000
└── Redis Service
    └── Internal networking only

Supabase Cloud
└── PostgreSQL with pgvector
```

## Key Points

- Only 2 ports exposed publicly (frontend: 3000, backend: 8000)
- Development tools (Mailhog, Redis Commander) excluded from production
- Separate Dockerfiles for development (`Dockerfile.dev`) and production (`Dockerfile.prod`)
- Health check endpoints configured for both services
- Environment variables managed through Railway dashboard

## Support

- Railway Issues: Check Railway Discord or documentation
- Application Issues: Check application logs in Railway dashboard
- Database Issues: Check Supabase dashboard