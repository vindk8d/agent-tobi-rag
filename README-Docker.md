# Docker Development Environment

This document describes how to set up and use the Docker-based development environment for RAG-Tobi.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0 or higher
- Git (for cloning the repository)

## Quick Start

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd rag-tobi
   ```

2. **Set up environment variables:**
   ```bash
   cp env-template.txt .env
   # Edit .env file with your actual API keys and configuration
   ```

3. **Run the automated setup script:**
   ```bash
   ./scripts/dev-setup.sh
   ```

4. **Access the application:**
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:3000
   - Redis: localhost:6379
   - pgAdmin (optional): http://localhost:5050

## Manual Setup

If you prefer to run commands manually:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services Overview

### Core Services

- **Backend** (`backend`): FastAPI application with RAG functionality
- **Frontend** (`frontend`): Next.js web application
- **Redis** (`redis`): Caching and session storage

### Development Tools (Optional)

- **pgAdmin** (`pgadmin`): Database administration interface
- **MailHog** (`mailhog`): Email testing tool (via override file)
- **Redis Commander** (`redis-commander`): Redis administration interface (via override file)

## Configuration

### Environment Variables

All services use the `.env` file for configuration. Key variables include:

- `OPENAI_API_KEY`: OpenAI API key for embeddings and chat
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_ANON_KEY`: Supabase anonymous key
- `SUPABASE_SERVICE_KEY`: Supabase service role key
- `LANGSMITH_API_KEY`: LangSmith API key for tracing

### Custom Configuration

Create a `docker-compose.override.yml` file for custom local settings:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

## Development Workflow

### Starting Development

```bash
# Start all services
docker-compose up -d

# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f backend
```

### Code Changes

- **Backend**: Changes are automatically reloaded via volume mounts
- **Frontend**: Changes are automatically reloaded via Next.js hot reload
- **Configuration**: Restart services after changing environment variables

### Database Operations

```bash
# Access PostgreSQL via pgAdmin
open http://localhost:5050

# Run database migrations (when backend supports it)
docker-compose exec backend python -m alembic upgrade head
```

### Testing

```bash
# Run backend tests
docker-compose exec backend pytest

# Run frontend tests
docker-compose exec frontend npm test

# Run linting
docker-compose exec backend flake8
docker-compose exec frontend npm run lint
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: If ports 3000 or 8000 are in use, modify the port mappings in `docker-compose.override.yml`

2. **Permission errors**: Ensure your user has Docker permissions
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Build failures**: Clear Docker cache and rebuild
   ```bash
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

4. **Database connection errors**: Verify Supabase credentials in `.env`

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Resetting Environment

```bash
# Complete reset (removes all data)
./scripts/dev-teardown.sh

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Production Considerations

This Docker setup is optimized for development. For production deployment:

1. Use production Dockerfiles with optimized builds
2. Set up proper secrets management
3. Configure reverse proxy (nginx/traefik)
4. Set up monitoring and logging
5. Use Docker Swarm or Kubernetes for orchestration

## Useful Commands

```bash
# Start with specific profile
docker-compose --profile tools up -d

# Scale services
docker-compose up -d --scale backend=2

# Execute commands in containers
docker-compose exec backend bash
docker-compose exec frontend sh

# View resource usage
docker stats

# Clean up unused resources
docker system prune -f
``` 