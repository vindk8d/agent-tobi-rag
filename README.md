# Salesperson Copilot Agent with RAG

## Overview
The Salesperson Copilot Agent is an AI-powered assistant designed to help salespeople quickly access consolidated information from multiple departments (marketing, finance, operations) via a single interface. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, real-time responses about company products, services, promotions, and inventory.

- **User Interface:** Telegram bot for salespeople, web-based dashboard for admins
- **Backend:** Python (FastAPI), LangChain, LangGraph, OpenAI GPT, Supabase
- **Frontend:** Next.js, Tailwind CSS

## Key Features
- Unified conversational interface for salespeople (Telegram)
- Web dashboard for admins to manage document uploads
- Document processing and indexing (PDF, Word, etc.)
- Real-time, semantic search using OpenAI embeddings
- Daily automated data refresh
- Conversation history and proactive suggestions
- Source attribution and conflict detection
- Monitoring and logging (LangSmith, Supabase)

**Note:** Website scraping functionality has been deprioritized due to the complexity of gathering relevant data from websites, HTML/JS parsing challenges, and the need for complex scraping methodologies. The system focuses on document upload as the primary data source.

## Goals
- Reduce information retrieval time for salespeople
- Improve sales support with up-to-date, accurate info
- Consolidate data sources into a single source of truth
- Enhance sales productivity
- Ensure data freshness with daily updates

## Architecture
- **Frontend:** Next.js 14+, TypeScript, Tailwind CSS
- **Backend:** FastAPI microservices, LangChain, LangGraph
- **Database:** Supabase (PostgreSQL + vector extensions)
- **LLM:** OpenAI GPT-4, text-embedding-3-small
- **Monitoring:** LangSmith
- **Queue:** Redis (for Telegram concurrency)
- **Document Processing:** Python libraries for PDF, Word, and other formats

## Functional Highlights
- Document upload and ingestion (admin dashboard)
- Embedding and indexing of all content
- Semantic search for user queries
- Telegram bot with fast, context-aware responses
- Conversation history and context management
- Proactive suggestions and error handling
- Logging of queries, responses, and conflicts
- Source attribution for all answers

## Setup (High-Level)
1. **Clone the repo:**
   ```bash
   git clone https://github.com/vindk8d/agent-tobi-rag.git
   ```
2. **Backend:**
   - Python 3.10+, install dependencies from `backend/requirements.txt`
   - Configure environment variables (see `env-template.txt`)
   - Run FastAPI server
3. **Frontend:**
   - Node.js 18+, install dependencies from `frontend/package.json`
   - Run Next.js dev server
4. **Supabase:**
   - Set up a Supabase project
   - Apply migrations from `supabase/migrations/`
5. **Telegram Bot:**
   - Create a bot via BotFather, set token in env
6. **Admin Dashboard:**
   - Access via web frontend for data source management

## Contributing
- Please see the PRD in `tasks/prd-salesperson-copilot-rag.md` for detailed requirements and open questions.
- Issues and feature requests welcome!

## License
MIT 