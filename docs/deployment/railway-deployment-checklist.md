# Railway Deployment Checklist

## Pre-Deployment Verification

- [ ] Run `./scripts/deploy-to-railway.sh` to verify all files are ready
- [ ] Ensure all sensitive data is removed from code
- [ ] Verify `.env` is in `.gitignore`
- [ ] Commit all changes to Git

## Railway Setup

### 1. Create Railway Project
- [ ] Go to [railway.app](https://railway.app)
- [ ] Click "New Project"
- [ ] Select "Empty Project"
- [ ] Name your project (e.g., "agent-tobi-rag")

### 2. Add Redis Service
- [ ] In Railway dashboard, click "+ New"
- [ ] Select "Database" → "Add Redis"
- [ ] Redis will be automatically provisioned
- [ ] Note: Redis URL will be available as `${{Redis.REDIS_URL}}`

### 3. Deploy Backend Service
- [ ] Click "+ New" → "GitHub Repo"
- [ ] Select your repository
- [ ] Configure service:
  - **Service Name**: `backend`
  - **Root Directory**: `/backend`
  - **Build Command**: (leave empty - uses Dockerfile)
  - **Start Command**: (leave empty - uses Dockerfile)
  - **Dockerfile Path**: `backend/Dockerfile.prod`
- [ ] Add environment variables from `railway-environment-variables.md`
- [ ] Deploy

### 4. Deploy Frontend Service
- [ ] Click "+ New" → "GitHub Repo"
- [ ] Select your repository
- [ ] Configure service:
  - **Service Name**: `frontend`
  - **Root Directory**: `/frontend`
  - **Build Command**: (leave empty - uses Dockerfile)
  - **Start Command**: (leave empty - uses Dockerfile)
  - **Dockerfile Path**: `frontend/Dockerfile.prod`
- [ ] Add environment variables from `railway-environment-variables.md`
- [ ] Deploy

## Post-Deployment Configuration

### 5. Update Domain-Dependent Variables
- [ ] Get backend URL from Railway (e.g., `backend-production-xxx.up.railway.app`)
- [ ] Get frontend URL from Railway (e.g., `frontend-production-xxx.up.railway.app`)
- [ ] Update backend service variables:
  - `FASTAPI_CORS_ORIGINS=["https://frontend-production-xxx.up.railway.app"]`
- [ ] Update frontend service variables:
  - `NEXT_PUBLIC_API_URL=https://backend-production-xxx.up.railway.app`
  - `NEXTAUTH_URL=https://frontend-production-xxx.up.railway.app`
- [ ] Redeploy both services

### 6. Verify Deployment
- [ ] Check backend health: `https://backend-url/health`
- [ ] Check frontend health: `https://frontend-url/api/health`
- [ ] Test basic functionality
- [ ] Check Railway logs for any errors

### 7. Configure Custom Domains (Optional)
- [ ] Add custom domain in Railway service settings
- [ ] Update DNS records (CNAME to Railway domain)
- [ ] Update environment variables with new domains
- [ ] Enable HTTPS (automatic with Railway)

## Monitoring & Maintenance

### 8. Set Up Monitoring
- [ ] Enable Railway metrics
- [ ] Set up uptime monitoring (e.g., UptimeRobot)
- [ ] Configure error alerts
- [ ] Set up backup strategy for Supabase

### 9. Production Readiness
- [ ] Test all API endpoints
- [ ] Verify Supabase connection
- [ ] Check Redis connectivity
- [ ] Test document upload functionality
- [ ] Verify RAG search functionality

## Troubleshooting Commands

```bash
# View logs
railway logs -s backend
railway logs -s frontend

# Connect to service
railway link
railway shell

# View/Update environment variables
railway variables -s backend
railway variables -s frontend

# Restart service
railway restart -s backend
railway restart -s frontend
```

## Rollback Plan

If deployment fails:
1. Check Railway build logs
2. Verify environment variables
3. Check Dockerfile syntax
4. Review recent commits
5. Rollback to previous deployment in Railway dashboard

## Security Checklist

- [ ] All API keys are in environment variables
- [ ] CORS is properly configured
- [ ] HTTPS is enabled
- [ ] Database connections use SSL
- [ ] No debug mode in production
- [ ] Error messages don't expose sensitive info

## Performance Optimization

- [ ] Enable Redis connection pooling
- [ ] Configure proper replica counts
- [ ] Set up CDN for static assets
- [ ] Enable gzip compression
- [ ] Optimize Docker image sizes

## Final Steps

- [ ] Document the deployment in team wiki
- [ ] Share access with team members
- [ ] Set up deployment notifications
- [ ] Schedule regular security updates