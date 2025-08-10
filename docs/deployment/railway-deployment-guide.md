# Railway Deployment Guide for Agent Tobi RAG

## Overview
This guide provides step-by-step instructions for deploying the Agent Tobi RAG application to Railway, using Supabase Cloud as the database provider.

## Architecture
The deployment consists of three main services on Railway:
- **Frontend Service**: Next.js application (Port 3000)
- **Backend Service**: FastAPI with LangChain/LangGraph (Port 8000)
- **Redis Service**: For caching and session management (Internal only)

Supabase Cloud provides:
- PostgreSQL database with pgvector extension
- Authentication and authorization
- Real-time subscriptions

**Note:** Only 2 ports are exposed publicly:
- Frontend: Port 3000 (or Railway's assigned port)
- Backend: Port 8000 (or Railway's assigned port)

Development tools like Mailhog, Redis Commander, and debugging ports are NOT included in production.

## Prerequisites
1. Railway account
2. Supabase Cloud project (already configured)
3. OpenAI API key
4. Git repository with the application code

## Deployment Steps

### 1. Create Production Dockerfiles

#### Backend Production Dockerfile (`backend/Dockerfile.prod`)
```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Only expose the main API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Production Dockerfile (`frontend/Dockerfile.prod`)
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

COPY --from=builder /app/package*.json ./
RUN npm ci --only=production
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/public ./public

USER nextjs

# Only expose the Next.js port
EXPOSE 3000

CMD ["npm", "start"]
```

### 2. Create Railway Configuration Files

#### Root `railway.json`
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 3. Environment Variables for Railway

#### Backend Service Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.3

# Supabase Configuration
SUPABASE_URL=https://ovppegnboovxussasjff.supabase.co
SUPABASE_ANON_KEY=<your-supabase-anon-key>
SUPABASE_SERVICE_KEY=<your-supabase-service-key>
SUPABASE_DB_PASSWORD=<your-supabase-db-password>

# Redis Configuration (Railway internal URL)
REDIS_URL=${{Redis.REDIS_URL}}

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_CORS_ORIGINS=["https://your-frontend-domain.railway.app"]

# System Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=100
RESPONSE_TIMEOUT_SECONDS=30

# RAG Configuration
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_SIMILARITY_THRESHOLD=0.8
RAG_MAX_RETRIEVED_DOCUMENTS=10
RAG_EMBEDDING_BATCH_SIZE=100

# Optional: LangSmith for monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=salesperson-copilot-rag
```

#### Frontend Service Environment Variables
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-backend-domain.railway.app

# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://ovppegnboovxussasjff.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-anon-key>

# NextAuth Configuration
NEXTAUTH_SECRET=<generate-a-secure-secret>
NEXTAUTH_URL=https://your-frontend-domain.railway.app
```

### 4. Railway Deployment Process

1. **Create New Project in Railway**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Empty Project"

2. **Add Redis Service**
   - Click "New Service"
   - Select "Database" → "Redis"
   - Railway will automatically provision Redis

3. **Deploy Backend Service**
   - Click "New Service" → "GitHub Repo"
   - Select your repository
   - Set Root Directory: `/backend`
   - Set Dockerfile Path: `./Dockerfile.prod` (relative to backend directory)
   - Add all backend environment variables
   - Deploy
   
   **Note**: The `requirements.txt` file must be present in the backend directory

4. **Deploy Frontend Service**
   - Click "New Service" → "GitHub Repo"
   - Select your repository
   - Set Root Directory: `/frontend`
   - Set Dockerfile Path: `./Dockerfile.prod` (relative to frontend directory)
   - Add all frontend environment variables
   - Deploy

5. **Configure Service URLs**
   - Backend: Generate domain or use custom domain
   - Frontend: Generate domain or use custom domain
   - Update CORS origins in backend env vars
   - Update API URL in frontend env vars

### 5. Post-Deployment Configuration

1. **Update CORS Settings**
   - In backend environment variables, update `FASTAPI_CORS_ORIGINS` to include your frontend URL

2. **Configure Health Checks**
   - Backend health check endpoint: `/health`
   - Frontend health check endpoint: `/api/health`

3. **Set Up Custom Domains (Optional)**
   - Add custom domains in Railway service settings
   - Update DNS records
   - Update environment variables with new domains

### 6. Database Migration
Since we're using Supabase Cloud, the database is already set up. Ensure:
- All migrations in `/supabase/migrations/` have been applied
- pgvector extension is enabled
- Connection string in Railway uses Supabase credentials

### 7. Monitoring and Maintenance

1. **Enable Railway Metrics**
   - CPU and Memory usage monitoring
   - Request/Response metrics
   - Error tracking

2. **Set Up Alerts**
   - Configure alerts for service failures
   - Set up uptime monitoring

3. **Scaling Configuration**
   - Adjust replica count as needed
   - Configure autoscaling rules

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies are listed
   - Check Railway build logs

2. **Connection Issues**
   - Verify environment variables
   - Check CORS configuration
   - Ensure services are using correct internal URLs

3. **Performance Issues**
   - Monitor Redis connection
   - Check Supabase query performance
   - Review Railway metrics

### Debug Commands
```bash
# Check service logs
railway logs -s <service-name>

# SSH into service (if enabled)
railway shell -s <service-name>

# View environment variables
railway variables -s <service-name>
```

## Security Considerations

1. **Environment Variables**
   - Never commit sensitive keys to repository
   - Use Railway's environment variable management
   - Rotate keys regularly

2. **Network Security**
   - Use HTTPS for all public endpoints
   - Configure proper CORS policies
   - Only 2 ports exposed publicly (Frontend: 3000, Backend: 8000)
   - Redis communicates internally via Railway's private network

3. **Database Security**
   - Use connection pooling
   - Enable SSL for database connections
   - Regular backups via Supabase

## Cost Optimization

1. **Service Configuration**
   - Start with 1 replica per service
   - Scale based on actual usage
   - Use Railway's sleep feature for dev environments

2. **Resource Limits**
   - Set appropriate CPU/Memory limits
   - Monitor usage patterns
   - Optimize container sizes

## Maintenance

1. **Regular Updates**
   - Keep dependencies updated
   - Monitor security advisories
   - Plan maintenance windows

2. **Backup Strategy**
   - Supabase handles database backups
   - Store application configs in git
   - Document environment variables

## Additional Resources

### Deployment Files
- **Environment Variables**: See `railway-environment-variables.md`
- **Deployment Checklist**: See `railway-deployment-checklist.md`
- **Deployment Script**: Run `./scripts/deploy-to-railway.sh` for pre-deployment checks

### Support Resources
- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Supabase Documentation: https://supabase.com/docs
- Project Repository: [Your GitHub repo URL]