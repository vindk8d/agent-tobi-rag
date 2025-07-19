# Task List: Salesperson Copilot Agent with RAG

Based on PRD: `prd-salesperson-copilot-rag.md`

## 🎯 Agent Design Principle

**Keep agent logic simple yet effective** - The goal is to achieve the needed functionalities with as little complexity as needed to keep our agent effective, lightweight and traceable. Prioritize clarity over complexity.

## Relevant Files

#### Backend/Agents
- `backend/agents/tobi_sales_copilot/rag_agent.py` - Main RAG agent implementation using LangGraph StateGraph with proper message persistence via PostgreSQL checkpointer
- `backend/agents/tobi_sales_copilot/state.py` - Simplified AgentState schema with essential persistent fields only (messages, conversation_id, user_id, retrieved_docs, sources, conversation_summary)
- `backend/agents/memory.py` - Consolidated memory system combining short-term (LangGraph checkpointer), long-term (Supabase Store), and conversation consolidation
- `backend/agents/memory_nodes.py` - Simplified LangGraph memory nodes (context retrieval and storage only)
- `backend/agents/tools.py` - RAG tools for document retrieval and CRM tools for business data queries with comprehensive fallback handling

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
- `supabase/migrations/20250115000000_create_crm_sales_tables.sql` - Comprehensive CRM sales management schema with 8 core tables for branches, employees, customers, vehicles, opportunities, transactions, pricing, and activities
- `supabase/migrations/20250115000001_insert_sample_crm_data.sql` - Sample CRM data with realistic sales scenarios, employee hierarchy, and customer interactions
- `supabase/database_schema.md` - Complete database schema documentation including CRM tables and relationships
- `docs/crm-sales-system.md` - Comprehensive CRM system documentation with usage examples and business logic
- `supabase/migrations/20250120000000_add_long_term_memory_tables.sql` - Long-term memory database schema following LangGraph Store best practices with semantic search support
- `supabase/migrations/20250120000001_optimize_memory_performance.sql` - Performance optimization for long-term memory with vector indexes, query optimization, and maintenance functions

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
- **CRM Integration Completed:** Comprehensive CRM database schema with 8 core tables enables natural language queries for sales data, pricing, inventory, customer management, and performance analytics
- **State Schema Simplified:** Removed redundant variables (user_query, conversation_history, response, response_metadata, current_step, error_message, timestamp, confidence_score, retrieval_metadata) to focus on essential persistent memory fields: messages, conversation_id, user_id, retrieved_docs, sources, conversation_summary (for periodic context retention)
- **Conversation Summary Strategy:** Implemented hybrid approach with periodic LLM-based summary generation every 8-10 messages stored in state schema (not cluttering messages) with automatic persistence via LangGraph checkpointing and database metadata storage for long-term context retention
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

- [x] 4.0 Implement CRM Database Integration for Sales Data Access
  - [x] 4.1 Create comprehensive CRM database schema with 8 core tables: branches, employees, customers, vehicles, opportunities, transactions, pricing, and activities
  - [x] 4.2 Implement CRM migration files with proper relationships, indexes, and constraints for automotive sales management
  - [x] 4.3 Create sample CRM data migration with realistic sales scenarios, employee hierarchy, and customer interactions
  - [x] 4.4 Develop natural language to SQL query engine for safe CRM data access with security validation
  - [x] 4.5 Implement query_crm_data tool with comprehensive schema documentation and business context understanding
  - [x] 4.6 Create CRM query capabilities for pricing, inventory, sales performance, customer management, and opportunity tracking
  - [x] 4.7 Integrate CRM tool with existing RAG agent architecture for seamless document and CRM data access
  - [x] 4.8 Implement comprehensive CRM documentation with usage examples and business logic explanations

- [x] 5.0 Develop Basic LangGraph Agent with Conversational RAG
  - [x] 5.1 Design minimal LangGraph state schema with user query, retrieved documents, and conversation context
  - [x] 5.2 Implement simple agent graph with core nodes: retrieval, generation, and response formatting
  - [x] 5.3 Create basic tools for semantic search and source attribution (without conflict logging)
  - [x] 5.4 Set up LangSmith tracing integration for all agent components and conversation flows
      - [x] 5.5 Implement hybrid conversation memory system using LangGraph persistence best practices with periodic conversation summarization:
    - [x] 5.5.1 Configure LangGraph built-in persistence layer (SqliteSaver/PostgresSaver) for short-term memory with thread-based conversation management and enhanced state schema including conversation_summary field
    - [x] 5.5.2 Implement ConversationMemoryManager class with sliding window context management (configurable size, default 10 messages) and periodic summary generation (every 8-10 messages) using LLM-based summarization
    - [x] 5.5.3 Set up automatic checkpointing and state persistence between agent steps using LangGraph's persistence layer with conversation_summary field included in state serialization
    - [x] 5.5.4 Implement long-term memory integration following LangGraph Store best practices:
       - **Database Schema:** Long-term memory tables with vector embeddings, conversation summaries, and access patterns
       - **LangGraph Store Implementation:** Custom SupabaseLongTermMemoryStore extending BaseStore interface
       - **Conversation Consolidation:** Automated transition from short-term to long-term memory with LLM summarization
       - **Enhanced Memory Manager:** Integration with long-term memory capabilities and semantic search
       - **Simplified State Management:** Clean AgentState schema with only essential persistent fields:
         - Core: messages, conversation_id, user_id (LangGraph essentials)
         - RAG: retrieved_docs, sources (current session context)
         - Summary: conversation_summary (for long conversations)
         - Transient data (context, preferences, stats) retrieved by nodes when needed
       - **Memory Nodes:** LangGraph nodes for context retrieval, consolidation, and user preference management
       - **Performance Optimization:** Vector indexes, query optimization, and maintenance functions
    - [x] 5.6 Configure LangChain Expression Language (LCEL) chains for retrieval and generation
    - [x] 5.7 Implement graceful fallback responses when no relevant information is found, including comprehensive SQL query failure handling
    - [x] 5.8 Test end-to-end conversation flow with document retrieval, CRM data queries, and response generation
    - [x] 5.9 Create Memory Debug Interface for Testing and Monitoring Persistent Memory Functionalities
       - **Frontend Location:** `/frontend/app/memorycheck` - Dedicated debugging interface for memory system validation
       - **Chat Interface:** Interactive chat component connected to messages table for real-time conversation testing
       - **User Data Display:** Container showing comprehensive user information from users, customers, and related CRM tables
       - **Memory Visualization:** Real-time display of long_term_memories table entries filtered by selected user
       - **Memory Details:** Show namespace structure, trigger conditions, storage timestamps, and access patterns
       - **Conversation Summaries:** Display periodic conversation consolidation and LLM-generated summaries
       - **Debugging Tools:** Memory trigger testing, manual consolidation controls, and storage pattern analysis
       - **User Selection:** Dropdown to switch between different users for comparative memory analysis
       - **Real-time Updates:** Live updates using Supabase subscriptions for memory changes and conversation activity

- [x] 6.0 Quota Usage Optimization and Rate Limiting
  - [x] 6.1 Implement token usage tracking and monitoring system for OpenAI API calls
  - [x] 6.2 Add rate limiting and quota management to prevent API overuse
  - [x] 6.3 Optimize embedding generation with batch processing and caching strategies
  - [x] 6.4 Implement conversation memory management to reduce context window usage
  - [x] 6.5 Add embedding deduplication to avoid re-processing identical content
  - [ ] 6.6 Implement dynamic model selection based on task complexity (GPT-3.5-turbo for simple queries, GPT-4 for complex reasoning)
  - [ ] 6.7 Implement smart caching for frequently accessed documents and embeddings
  - [ ] 6.8 Add request throttling and queue management for high-volume usage
  - [ ] 6.9 Create quota usage alerts and automatic fallback mechanisms
  - [ ] 6.10 Implement embedding compression techniques to reduce storage costs
  - [ ] 6.11 Add usage analytics and cost optimization recommendations

- [ ] 7.0 Extend Agent with Advanced Features (Post-Basic RAG)
  - [ ] 7.1 Add conflict detection logic that logs to console and responds appropriately to users
  - [ ] 7.2 Implement proactive suggestion logic based on conversation context and available documents
  - [ ] 7.3 Enhance tools with conflict logging capabilities
  - [ ] 7.4 Optimize performance with advanced LCEL patterns and caching strategies

- [ ] 8.0 Implement Monitoring, Testing, and Deployment Infrastructure
      - [ ] 8.1 Integrate LangSmith tracing throughout the agent pipeline for comprehensive monitoring (RAG and SQL operations)
  - [ ] 8.2 Set up custom metrics tracking for response accuracy, query resolution, and performance
  - [ ] 8.3 Implement comprehensive error handling with proper logging and alerting mechanisms
      - [ ] 8.4 Create unit tests for RAG components and CRM/SQL tools using pytest with proper mocking of external APIs
      - [ ] 8.5 Set up integration tests for agent workflows and conversation flows (including RAG and CRM data queries)
      - [ ] 8.6 Configure performance monitoring with response time tracking for both RAG and SQL operations, and concurrent user support
  - [ ] 8.7 Implement rate limiting and API security measures for production deployment
  - [ ] 8.8 Set up automated backup strategies for vector embeddings and conversation history

- [ ] 9.0 Build Temporary Chat Interface for Testing and Development
  - [ ] 9.1 Create simple React chat component with message history and real-time updates
  - [ ] 9.2 Implement WebSocket or SSE connection for streaming agent responses
  - [ ] 9.3 Add debugging features showing retrieved documents, confidence scores, and processing steps
      - [ ] 9.4 Create test scenarios for different query types and edge cases (document queries, CRM data queries, mixed scenarios)
  - [ ] 9.5 Implement conversation reset and context management for testing different scenarios
  - [ ] 9.6 Add developer tools for inspecting agent state, memory, and decision processes
  - [ ] 9.7 Create performance testing interface to validate concurrent user support (up to 100 users)

- [ ] 10.0 Integrate Telegram Bot Interface with Production Deployment
  - [ ] 10.1 Set up Telegram Bot API and configure webhook endpoints for secure message handling
    - **NOTE**: Add back Telegram bot dependency to requirements.txt when starting this task: `python-telegram-bot==20.7`
  - [ ] 10.2 Implement Telegram bot handlers with proper message parsing and user session management
  - [ ] 10.3 Adapt agent responses for Telegram format with inline keyboards for proactive suggestions
  - [ ] 10.4 Configure production deployment with proper scaling and load balancing
  - [ ] 10.5 Implement user authentication and session persistence across Telegram conversations
  - [ ] 10.6 Set up production monitoring and alerting for bot uptime and response performance
  - [ ] 10.7 Create deployment scripts and CI/CD pipeline for automated updates
  - [ ] 10.8 Hide/remove temporary chat interface and finalize production configuration
