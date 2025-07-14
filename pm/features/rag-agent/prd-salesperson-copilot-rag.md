# Product Requirements Document: Salesperson Copilot Agent with RAG

## Introduction/Overview

The Salesperson Copilot Agent is an AI-powered assistant designed to help salespeople access consolidated information from across multiple departments (marketing, finance, operations) through a single interface. The system uses Retrieval-Augmented Generation (RAG) to provide accurate, real-time responses about company products, services, promotions, and inventory.

The copilot operates through a Telegram bot interface for salespeople and includes a web-based management dashboard for administrators. The system primarily focuses on document upload and indexing to build a comprehensive knowledge base that salespeople can query conversationally.

**Note:** Website scraping functionality has been deprioritized due to the complexity of gathering relevant data from websites, HTML/JS parsing challenges, and the need for complex scraping methodologies. The system will focus on document upload (PDFs, Word docs, etc.) as the primary data source.

**Problem Statement:** Salespeople currently need to contact multiple departments or search through various systems to get information needed to close sales, leading to delays and missed opportunities.

**Solution:** A unified AI copilot that consolidates all sales-relevant information from uploaded documents into a single, conversational interface accessible via Telegram.

## Goals

1. **Reduce Information Retrieval Time:** Decrease the time salespeople spend gathering information from multiple sources from hours to seconds
2. **Improve Sales Support:** Provide accurate, up-to-date information about products, promotions, and inventory in real-time
3. **Consolidate Data Sources:** Create a single source of truth for sales-relevant information across the organization
4. **Enhance Sales Productivity:** Enable salespeople to focus on selling rather than information gathering
5. **Ensure Data Freshness:** Maintain current information through automated daily updates from source systems

## User Stories

### Salesperson Stories
- **As a salesperson**, I want to ask "What's the latest promotion on Tesla Model 3?" so that I can provide current offers to my clients
- **As a salesperson**, I want to query "How many BMW X5s do we have in stock?" so that I can give accurate availability information to customers
- **As a salesperson**, I want to ask "Show me all hot leads assigned to me this week" so that I can prioritize my follow-up activities
- **As a salesperson**, I want to query "What's the average deal size for SUVs in the last quarter?" so that I can set realistic expectations with customers
- **As a salesperson**, I want to ask "Which customers in my territory haven't been contacted in the last 30 days?" so that I can maintain regular touchpoints
- **As a salesperson**, I want to query "What vehicles are available in my branch with current pricing?" so that I can provide accurate inventory and pricing information
- **As a salesperson**, I want the copilot to proactively suggest relevant information during our conversation so that I don't miss important details that could help close the sale
- **As a salesperson**, I want to access the copilot through Telegram so that I can get information quickly while on calls or meetings with clients
- **As a salesperson**, I want my conversation history to be preserved so that I can reference previous queries and build context over time

### Administrator Stories
- **As a sales manager**, I want to upload documents to the system so that the copilot has access to the latest marketing materials and product information
- **As an IT admin**, I want to upload internal documents to the system so that sales materials are available to the copilot
- **As a sales manager**, I want to monitor which documents are successfully indexed so that I can ensure all relevant information is available
- **As an admin**, I want to receive alerts when document processing fails so that I can address data source issues promptly

## Functional Requirements

### Core RAG System
1. ~~The system must scrape and index content from websites provided by administrators~~ (DEPRIORITIZED)
2. The system must support document upload functionality for PDFs, Word documents, and other text-based files
3. The system must immediately embed, index, and store data as soon as a document is uploaded via the frontend
4. The system must refresh data from all document sources at least once daily
5. The system must generate embeddings for all indexed content using OpenAI's embedding models
6. The system must store embeddings and metadata in Supabase for efficient retrieval
7. The system must use semantic search to find relevant information based on user queries

### CRM Data Access System
8. The system must provide secure SQL query generation capabilities to answer questions about CRM data (branches, employees, customers, vehicles, opportunities, transactions, pricing, and activities)
9. The system must restrict SQL access to READ-ONLY operations on CRM tables only (no DROP, DELETE, UPDATE, or access to RAG system tables)
10. The system must validate all generated SQL queries against a whitelist of allowed operations and table access patterns
11. The system must handle high-cardinality columns (customer names, vehicle models, employee names) using semantic search for proper noun validation
12. The system must implement query complexity limits to prevent resource-intensive operations (maximum 10 JOINs, no nested subqueries beyond 2 levels)
13. The system must log all SQL queries and responses for security auditing and performance monitoring

### Telegram Bot Interface
14. The system must provide a Telegram bot interface for salespeople to interact with the copilot
15. The system must respond to user queries within 3 seconds under normal conditions for document queries and 5 seconds for complex SQL queries
16. The system must maintain conversation history and context between sessions
17. The system must support up to 100 concurrent users without performance degradation
18. The system must gracefully handle cases where no relevant information is found by stating "I don't know"
19. The system must proactively suggest relevant information when contextually appropriate
20. The system must automatically determine whether to use document search or CRM data queries based on the user's question context

### Web Management Dashboard
21. The system must provide a web-based dashboard for administrators to manage data sources
22. The dashboard must display all indexed documents with their status (successful/failed)
23. The dashboard must show embedding quality metrics and indexing statistics
24. ~~The dashboard must provide functionality to add new website URLs for scraping~~ (DEPRIORITIZED)
25. The dashboard must support document upload with drag-and-drop functionality
26. ~~The dashboard must display alerts when websites become inaccessible or fail to scrape~~ (DEPRIORITIZED)
27. The dashboard must provide a CRM data overview showing table row counts, data freshness, and SQL query performance metrics

### Data Management
28. The system must log all user queries and responses for analysis
29. The system must log conflicting information from different sources to console for debugging purposes and inform the user that conflicting information was found and the question cannot be answered at this time
30. The system must provide source attribution for all responses (document sources for RAG queries, table sources for CRM queries)
31. ~~The system must handle website structure changes gracefully with appropriate error handling~~ (DEPRIORITIZED)
32. The system must maintain audit logs of all SQL queries with execution time, affected rows, and user context for security monitoring

### Technical Architecture
33. The system must use LangChain and LangGraph for agent architecture and conversation management
34. The system must integrate with LangSmith for tracing and monitoring
35. The system must use OpenAI GPT models for natural language generation and SQL query generation
36. The system must use Supabase for data persistence and user session management
37. The frontend must be built using Next.js and Tailwind CSS
38. The system must implement LangChain's SQL Database toolkit with proper security configurations and query validation
39. The system must use LangChain's semantic retrieval for high-cardinality column validation (customer names, vehicle models, employee names)

## Non-Goals (Out of Scope)

- **Multi-language Support:** Initial version will only support English
- **Content Filtering:** No content approval process for uploaded documents (future feature)
- **User Authentication:** No login system for the web dashboard (future feature)
- **Human-in-the-Loop Escalation:** No escalation paths to human experts (future feature)
- **Mobile App:** Only Telegram bot interface, no dedicated mobile application
- **Role-based Access Control:** No territory or product-specific data filtering for salespeople
- **Website Scraping:** Due to complexity of data gathering, HTML/JS parsing, and scraping methodologies (DEPRIORITIZED)
- **Real-time Document Monitoring:** No immediate alerts for document changes (daily refresh only)
- **Advanced Analytics:** No detailed usage analytics or sales performance correlation (future feature)

## Design Considerations

### Frontend Dashboard
- **Framework:** Next.js with Tailwind CSS for responsive, modern UI
- **Layout:** Single-page application with sidebar navigation
- **Color Scheme:** Professional blue/white theme consistent with business applications
- **Components:** Document upload area, data source management table, status indicators, alert notifications
- **Responsive Design:** Desktop-optimized (primary users are administrators at workstations)

### Telegram Bot UX
- **Conversational Flow:** Natural language interface with quick response buttons for common queries
- **Message Format:** Structured responses with clear source attribution
- **Error Handling:** Friendly error messages with suggestions for alternative queries
- **Proactive Suggestions:** Context-aware recommendations presented as optional buttons

## Technical Considerations

### Architecture Stack
- **Frontend:** Next.js 14+ with TypeScript, Tailwind CSS for styling
- **Backend:** Python-based microservices using FastAPI
- **Agent Framework:** LangChain for RAG pipeline, LangGraph for conversation management
- **Database:** Supabase (PostgreSQL) with vector extensions for embeddings
- **LLM Provider:** OpenAI GPT-4 for generation, text-embedding-3-small for embeddings
- **Monitoring:** LangSmith for agent tracing and performance monitoring
- **Message Queue:** Redis for handling concurrent Telegram bot requests
- **Document Processing:** Python libraries for PDF, Word, and other document formats

### Performance Requirements
- **Response Time:** <3 seconds for 95% of queries
- **Concurrent Users:** Support 100 simultaneous Telegram conversations
- **Data Refresh:** Complete daily refresh of all sources within 4-hour maintenance window
- **Uptime:** 99.5% availability during business hours

### Security Considerations
- **API Security:** Rate limiting on Telegram bot endpoints
- **Data Privacy:** No storage of customer PII in conversation logs
- **Access Control:** IP whitelisting for admin dashboard (initially)
- **Data Encryption:** All data encrypted at rest and in transit

## Success Metrics

### Primary Metrics
1. **Salesperson Satisfaction Score:** Target >4.0/5.0 rating from regular user surveys
2. **Response Accuracy:** Target >90% accuracy based on salesperson feedback
3. **Query Resolution Rate:** Target >85% of queries answered without escalation
4. **Response Time:** Target <3 seconds for 95% of queries

### Secondary Metrics
1. **System Adoption:** Target 80% of sales team using the copilot weekly within 3 months
2. **Data Freshness:** Target <24 hours lag between document updates and availability in system
3. **Document Coverage:** Target >95% of uploaded documents successfully processed daily
4. **Conversation Engagement:** Target >3 queries per conversation session

## Open Questions

1. **Data Source Prioritization:** How should the system handle conflicting information from sources of different authority levels?
2. **Usage Limits:** Should there be daily/hourly query limits per salesperson to manage costs?
3. **Conversation Context:** How long should conversation context be maintained (days/weeks)?
4. **Error Recovery:** What should be the retry policy for failed document processing attempts?
5. **Performance Monitoring:** What specific LangSmith metrics should trigger alerts for system administrators?
6. **Document Versioning:** How should the system handle updates to uploaded documents?
7. **Telegram Bot Security:** What measures should be implemented to prevent unauthorized access to the bot?
8. **SQL Query Complexity:** What are the optimal limits for SQL query complexity to balance functionality with performance and security?
9. **CRM Data Privacy:** Should certain CRM data fields (salaries, personal customer information) be restricted from SQL queries?
10. **Cross-Data Source Queries:** How should the system handle questions that require both document search and CRM data analysis? 