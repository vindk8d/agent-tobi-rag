# Railway Deployment Guide for RAG-Tobi

This guide covers deploying the RAG-Tobi application to Railway using NIXPACKS for optimal performance and reliability.

## 🚀 Quick Deployment

1. **Connect to Railway**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login to Railway
   railway login
   
   # Deploy from current directory
   railway deploy
   ```

2. **The deployment will automatically use NIXPACKS** as configured in `railway.json`

## 📋 Prerequisites

- Railway account (free tier available)
- Supabase project for database
- OpenAI API key
- All environment variables configured (see below)

## 🔧 Configuration Files

### 1. railway.json
- Configures NIXPACKS as the build system
- Sets up health checks and deployment parameters
- Optimized for Python FastAPI applications

### 2. nixpacks.toml
- Custom NIXPACKS configuration
- Installs system dependencies for PDF generation
- Sets up Python environment with proper paths

### 3. Dockerfile (Fallback)
- Production-ready fallback if NIXPACKS fails
- Multi-stage build for optimal size
- Security hardening with non-root user

## 🌍 Environment Variables

Set these in your Railway project dashboard:

### Required Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_COMPLEX_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.3

# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
SUPABASE_DB_PASSWORD=your_database_password

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
PYTHONPATH=backend
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Optional Variables
```bash
# Frontend URL for CORS (configured for your production deployment)
FRONTEND_URL=https://tobi-frontend-production.up.railway.app

# LangSmith (for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=salesperson-copilot-rag

# Redis (if using external Redis)
REDIS_URL=redis://your-redis-instance

# Performance Tuning
MAX_CONCURRENT_REQUESTS=100
RESPONSE_TIMEOUT_SECONDS=30
```

## 🏗️ Build Process

### NIXPACKS Build Steps
1. **Setup Phase**: Installs system dependencies (Python, Cairo, Pango, etc.)
2. **Install Phase**: Installs Python packages from requirements.txt
3. **Build Phase**: Compiles Python bytecode for faster startup
4. **Start Phase**: Launches uvicorn server

### System Dependencies Installed
- Python 3.11
- Cairo (for PDF generation)
- Pango (for text rendering)
- GDK-Pixbuf (for image handling)
- FontConfig (for font management)
- GCC/Build tools

## 📊 Monitoring & Health Checks

### Health Check Endpoint
- **URL**: `/health`
- **Method**: GET
- **Response**: JSON with service status, database health, and performance metrics

### Example Health Response
```json
{
  "success": true,
  "data": {
    "service": "RAG-Tobi API",
    "status": "healthy",
    "timestamp": 1704067200,
    "response_time_ms": 45.23,
    "checks": {
      "database": "healthy",
      "environment": "production",
      "port": "8000",
      "python_path": "backend"
    },
    "version": "1.0.0"
  }
}
```

### Current Production URLs
- **Backend**: https://tobi-backend-production.up.railway.app
- **Frontend**: https://tobi-frontend-production.up.railway.app
- **Health Check**: https://tobi-backend-production.up.railway.app/health

## 🔒 Security Features

### CORS Configuration
- Automatically configured for Railway domains
- Removes localhost origins in production
- Supports custom frontend URLs via environment variables

### Security Headers
- Non-root user execution
- Minimal system surface
- Pinned dependency versions

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check system dependencies in nixpacks.toml
   - Verify requirements.txt has pinned versions
   - Review build logs in Railway dashboard

2. **Database Connection Issues**
   - Verify Supabase credentials
   - Check SUPABASE_DB_PASSWORD is correct
   - Ensure Supabase allows connections from Railway IPs

3. **Health Check Failures**
   - Check `/health` endpoint response
   - Verify all required environment variables are set
   - Review application logs

4. **CORS Errors**
   - Set FRONTEND_URL environment variable
   - Verify Railway domain is in allowed origins
   - Check browser developer console for exact error

### Debug Commands
```bash
# Check deployment logs
railway logs

# SSH into running container (if needed)
railway shell

# Check environment variables
railway variables

# Redeploy with latest changes
railway deploy --detach
```

## 🔄 CI/CD Integration

### Automatic Deployments
Railway can automatically deploy on:
- Git push to main branch
- Pull request merges
- Manual triggers

### Environment Promotion
1. **Development**: Deploy from feature branches
2. **Staging**: Deploy from main branch
3. **Production**: Manual promotion or tag-based deployment

## 📈 Performance Optimization

### Build Optimizations
- Pinned dependency versions for consistent builds
- Compiled Python bytecode for faster startup
- Minimal system dependencies
- Multi-layer Docker caching (fallback)

### Runtime Optimizations
- Single worker process (suitable for Railway's container model)
- Optimized CORS settings
- Enhanced health checks with timing
- Structured logging for better monitoring

## 💰 Cost Optimization

### Railway Pricing Considerations
- **Free Tier**: 500 hours/month, good for development
- **Pro Plan**: $5/month, unlimited usage
- **Resource Limits**: Monitor CPU and memory usage

### Cost-Saving Tips
1. Use Railway's sleep feature for development environments
2. Optimize container resource allocation
3. Monitor database connection pooling
4. Use Redis for caching if needed

## 🔗 Related Services

### Recommended Railway Services
- **PostgreSQL**: For additional database needs
- **Redis**: For caching and session storage
- **Frontend**: Deploy Next.js frontend separately

### External Integrations
- **Supabase**: Primary database and auth
- **OpenAI**: AI/ML services
- **LangSmith**: Monitoring and observability

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [NIXPACKS Documentation](https://nixpacks.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Supabase Integration](https://supabase.com/docs)

## 🆘 Support

For deployment issues:
1. Check Railway dashboard logs
2. Review this guide's troubleshooting section
3. Consult Railway community forums
4. Check project documentation in `/docs/` folder

---

**Last Updated**: January 2025
**Railway CLI Version**: Latest
**NIXPACKS Version**: Latest
