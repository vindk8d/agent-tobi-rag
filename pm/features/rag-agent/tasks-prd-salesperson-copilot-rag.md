# Task List: Salesperson Copilot Agent with RAG

Based on PRD: `prd-salesperson-copilot-rag.md`

## Relevant Files

### Backend/Agent Infrastructure
- `backend/main.py` - FastAPI main application entry point with proper project structure
- `backend/config.py` - Environment configuration and settings
- `backend/database.py` - Supabase client setup and database connections
- `backend/api/` - API routes and endpoints organization
- `backend/models/` - Pydantic models for data validation
- `backend/models/base.py` - Base model classes with common functionality
- `backend/models/conversation.py` - Conversation and chat models
- `backend/models/document.py` - Document and file management models
- `backend/models/datasource.py` - Data source and document upload models
- `backend/agents/` - LangGraph agent components (structure initialized)
- `backend/rag/` - RAG system components
- `backend/rag/embeddings.py` - OpenAI embedding generation
- `backend/rag/document_loader.py` - Modular LangChain-based loader for PDFs, Word docs, and other document formats
- `backend/rag/vector_store.py` - Supabase vector store operations for upsert, similarity, and hybrid search
- `backend/rag/pipeline.py` - Document processing pipeline: load, chunk, embed, and store documents
- `backend/rag/retriever.py` - Semantic retriever for similarity search with configurable threshold and source attribution
- ~~`backend/scrapers/` - Web scraping infrastructure (DEPRIORITIZED)~~
- ~~`backend/scrapers/web_scraper.py` - Web scraping utility (DEPRIORITIZED)~~
- `backend/telegram/` - Telegram bot integration (structure initialized)
- `backend/monitoring/` - Monitoring and observability (structure initialized)
- `backend/tests/` - Test files for all backend components (structure initialized)
- `backend/monitoring/scheduler.py` - Daily refresh scheduler for document sources with error handling

### Frontend/Dashboard
- `frontend/app/` - Next.js 14 app router structure with TypeScript
- `frontend/app/page.tsx` - Modern dashboard page with system status and features overview
- `frontend/app/layout.tsx` - Professional layout with header, footer, and responsive design
- `frontend/app/globals.css` - Tailwind CSS with custom components and theme variables
- `frontend/components/` - Reusable React components (structure ready for future components)
- `frontend/lib/` - Utility functions and API clients
- `frontend/lib/api.ts` - Comprehensive API client for backend communication
- `frontend/lib/supabase.ts` - Supabase client configuration with helper functions
- `frontend/lib/utils.ts` - Common utility functions for formatting, validation, and DOM operations
- `frontend/types/` - Complete TypeScript type definitions
- `frontend/types/index.ts` - Type definitions for all models, API responses, and UI components
- `frontend/tailwind.config.js` - Professional blue/white theme with custom animations
- `frontend/postcss.config.js` - PostCSS configuration for Tailwind CSS
- `frontend/tsconfig.json` - TypeScript configuration with path mapping
- `frontend/package.json` - Dependencies with Tailwind CSS, Supabase, and formatting tools
- `frontend/.prettierrc` - Code formatting configuration

### Database/Migrations
- `supabase/migrations/` - Database migration files
- `supabase/migrations/001_initial_schema.sql` - Initial database schema with core tables for data sources, documents, conversations, and logging
- `supabase/migrations/002_vector_extensions.sql` - Vector search setup with pgvector extension and similarity search functions
- `supabase/migrations/003_rag_tables.sql` - RAG-specific tables for conflicts, metrics, feedback, and proactive suggestions

### Configuration
- `docker-compose.yml` - Local development environment
- `requirements.txt` - Python dependencies with OpenAI, LangChain, and all required packages
- `package.json` - Node.js dependencies
- `env-template.txt` - Environment variable template with backend and Next.js frontend configurations
- `backend/config.py` - Configuration management with validation functions for all services including Next.js
- `backend/rag/embeddings.py` - OpenAI text embedding integration with rate limiting and error handling
- `frontend/.env.local.example` - Next.js specific environment template with NEXT_PUBLIC_ variables
- `pytest.ini` - Python testing configuration

### Notes

- Use LangChain's modular architecture with separate components for loaders, splitters, and retrievers
- Implement LangGraph's state-based agent design for conversation management
- Follow Supabase best practices for vector embeddings and RLS policies
- Integrate LangSmith from the beginning for comprehensive tracing
- Use Next.js 14 app router for modern React patterns
- **Website Scraping Deprioritized:** Due to complexity of data gathering, HTML/JS parsing, and scraping methodologies, focus on document upload as primary data source
- **Dependency Management**: Heavy dependencies (~~Playwright~~, Telegram bot) were removed from initial requirements.txt to optimize build times. They will be added back when their respective tasks are started (see task-specific notes).

## Tasks

- [x] 1.0 Setup Technical Infrastructure and Environment
  - [x] 1.1 Initialize Supabase project with vector extensions (pgvector) and create database schema
  - [x] 1.2 Configure OpenAI API keys and set up embedding model (text-embedding-3-small)
  - [x] 1.3 Set up LangSmith account and configure tracing API keys for monitoring
  - [x] 1.4 Create environment configuration files (.env.example, config.py) with all required variables
  - [x] 1.5 Set up local development environment with Docker Compose for services orchestration
  - [x] 1.6 Initialize backend FastAPI project structure with proper dependency management
  - [x] 1.7 Initialize Next.js 14 frontend with TypeScript and Tailwind CSS configuration

- [x] 2.0 Build Core RAG System ~~with Web Scraping~~ and Document Processing
  - [x] 2.1 Implement LangChain document loaders for PDFs, Word docs, and other document formats using appropriate loaders
  - [x] 2.2 Set up text splitting strategy using RecursiveCharacterTextSplitter with optimal chunk sizes (1000-2000 tokens)
  - [x] 2.3 Implement OpenAI embeddings integration with proper error handling and rate limiting
  - [x] 2.4 Create Supabase vector store operations with proper indexing and similarity search functions
  - ~~[x] 2.5 Build web scraping infrastructure using BeautifulSoup4 and Playwright for dynamic content~~ (DEPRIORITIZED)
  - [x] 2.6 Implement document processing pipeline with immediate embedding upon upload
  - [x] 2.7 Create semantic retriever with configurable similarity thresholds and source attribution
  - [x] 2.8 Set up daily refresh scheduler for document sources with error handling

- [x] 3.0 Create Web Management Dashboard with Next.js
  - [x] 3.1 Create frontend interface to upload PDFs and documents and trigger processing (for testing RAG pipeline)
  - [x] 3.2 Set up Next.js 14 app router structure with TypeScript and proper folder organization
  - [x] 3.3 Create Supabase client configuration for frontend with proper environment variables
  - ~~[x] 3.4 Build data source management interface for adding/removing websites and viewing status~~ (DEPRIORITIZED)
  - [x] 3.5 Implement drag-and-drop document upload with progress indicators and validation
  - [x] 3.6 Create status dashboard showing embedding quality metrics and indexing statistics
  - ~~[x] 3.7 Build alert system for displaying website accessibility issues and scraping failures~~ (DEPRIORITIZED)
  - [x] 3.8 Design responsive UI using Tailwind CSS with professional blue/white theme
  - [x] 3.9 Implement real-time updates for document processing status using Supabase realtime subscriptions

- [ ] 4.0 Develop LangGraph Agent Architecture and Conversation Management
  - [ ] 4.1 Design LangGraph state schema with conversation memory, user context, and retrieved documents
  - [ ] 4.2 Implement agent graph with nodes for: retrieval, generation, conflict detection, and response formatting
  - [ ] 4.3 Create custom tools for semantic search, source attribution, and conflict logging
  - [ ] 4.4 Set up persistent conversation memory using LangGraph's built-in persistence with Supabase backend
  - [ ] 4.5 Implement proactive suggestion logic based on conversation context and available documents
  - [ ] 4.6 Add conflict detection logic that logs to console and responds appropriately to users
  - [ ] 4.7 Configure LangChain Expression Language (LCEL) chains for optimized performance
  - [ ] 4.8 Implement graceful fallback responses when no relevant information is found

- [ ] 5.0 Implement Monitoring, Testing, and Deployment Infrastructure
  - [ ] 5.1 Integrate LangSmith tracing throughout the agent pipeline for comprehensive monitoring
  - [ ] 5.2 Set up custom metrics tracking for response accuracy, query resolution, and performance
  - [ ] 5.3 Implement comprehensive error handling with proper logging and alerting mechanisms
  - [ ] 5.4 Create unit tests for RAG components using pytest with proper mocking of external APIs
  - [ ] 5.5 Set up integration tests for agent workflows and conversation flows
  - [ ] 5.6 Configure performance monitoring with response time tracking and concurrent user support
  - [ ] 5.7 Implement rate limiting and API security measures for production deployment
  - [ ] 5.8 Set up automated backup strategies for vector embeddings and conversation history

- [ ] 6.0 Build Temporary Chat Interface for Testing and Development
  - [ ] 6.1 Create simple React chat component with message history and real-time updates
  - [ ] 6.2 Implement WebSocket or SSE connection for streaming agent responses
  - [ ] 6.3 Add debugging features showing retrieved documents, confidence scores, and processing steps
  - [ ] 6.4 Create test scenarios for different query types and edge cases
  - [ ] 6.5 Implement conversation reset and context management for testing different scenarios
  - [ ] 6.6 Add developer tools for inspecting agent state, memory, and decision processes
  - [ ] 6.7 Create performance testing interface to validate concurrent user support (up to 100 users)

- [ ] 7.0 Integrate Telegram Bot Interface with Production Deployment
  - [ ] 7.1 Set up Telegram Bot API and configure webhook endpoints for secure message handling
    - **NOTE**: Add back Telegram bot dependency to requirements.txt when starting this task: `python-telegram-bot==20.7`
  - [ ] 7.2 Implement Telegram bot handlers with proper message parsing and user session management
  - [ ] 7.3 Adapt agent responses for Telegram format with inline keyboards for proactive suggestions
  - [ ] 7.4 Configure production deployment with proper scaling and load balancing
  - [ ] 7.5 Implement user authentication and session persistence across Telegram conversations
  - [ ] 7.6 Set up production monitoring and alerting for bot uptime and response performance
  - [ ] 7.7 Create deployment scripts and CI/CD pipeline for automated updates
  - [ ] 7.8 Hide/remove temporary chat interface and finalize production configuration 