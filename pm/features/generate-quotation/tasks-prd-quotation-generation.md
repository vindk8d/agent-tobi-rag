# Task List: AI Agent Quotation Generation System

Based on the PRD for the AI Agent Quotation Generation System, this document outlines the implementation tasks required to build the feature.

## Relevant Files

- `backend/agents/tools.py` - Add the generate_quotation tool and database helper functions (includes GenerateQuotationParams BaseModel, generate_quotation tool with @tool decorator, _create_quotation_preview helper, intelligent vehicle parsing functions, HITL resume logic)
- `backend/agents/tools.py` - Add helper functions: _lookup_customer, _lookup_vehicle_by_criteria, _lookup_current_pricing, _lookup_employee_details
- `backend/core/pdf_generator.py` - Comprehensive PDF generation module with WeasyPrint, template rendering, and async support
- `backend/core/storage.py` - New module for Supabase quotations bucket integration
- `backend/monitoring/quotation_cleanup.py` - Cleanup job to remove expired quotation PDFs and update DB
- `backend/monitoring/scheduler.py` - Scheduler wired to run quotation cleanup on an interval
- `backend/agents/tobi_sales_copilot/agent.py` - Add generate_quotation to employee agent toolset
- `supabase/migrations/20250808001737_create_quotations_table.sql` - Database table for quotation tracking and metadata with RLS policies, indexes, and auto-generated quotation numbers
- `supabase/migrations/20250808002000_create_quotations_storage_policies.sql` - Storage helper functions for quotation PDF management and signed URL generation
- `scripts/create-quotations-bucket.js` - Node.js script to create quotations storage bucket using Supabase service key
- `templates/quotation_template.html` - Professional HTML template with company branding, vehicle specs, pricing breakdown, and signatures
- `templates/quotation_styles.css` - Print-optimized CSS with responsive design, dark mode support, and accessibility features
- `frontend/app/dev/pdf-test/page.tsx` - Comprehensive React testing interface with tabbed forms, sample data, and real-time preview
- `backend/api/test_pdf_generation.py` - FastAPI endpoints for PDF generation, HTML preview, sample data, and health checks
- `tests/test_quotation_generation.py` - Comprehensive tests for quotation generation flow
- `tests/test_pdf_generation.py` - Unit tests for PDF generation functionality
- `tests/test_quotation_storage.py` - Tests for Supabase storage integration
 - `tests/test_quotation_cleanup_simulation.py` - E2E simulation test for upload + cleanup correctness and safety
 - `tests/test_database_helpers.py` - Unit tests for database helper functions (lookup customer/vehicle/pricing/employee)
 - `evaluation_generate_quotation_hitl.md` - HITL architecture compliance evaluation and identified issues

### Notes

- WeasyPrint requires specific HTML/CSS structure for professional PDF output
- Supabase storage integration should follow existing document upload patterns
- HITL flows must integrate with existing request_approval() and request_input() patterns
- Database helper functions should be reusable across multiple tools
- Tests should cover both successful flows and error scenarios
- **CRITICAL**: HITL resume logic is required for proper agent architecture compliance (identified in evaluation_generate_quotation_hitl.md)

## Tasks

- [x] 1.0 Database Infrastructure Setup
  - [x] 1.1 Create quotations table migration with fields: id, customer_id, employee_id, vehicle_specs, pricing_data, pdf_url, created_at, expires_at, status
  - [x] 1.2 Add proper indexes for customer_id, employee_id, and created_at for performance
  - [x] 1.3 Set up RLS policies for quotations table (employees can only access their own quotations)
  - [x] 1.4 Create quotations storage bucket in Supabase with proper access policies
  - [x] 1.5 Test database migration and storage bucket setup

- [x] 2.0 PDF Generation System Implementation
  - [x] 2.1 Install and configure WeasyPrint dependency in requirements.txt
  - [x] 2.2 Create professional HTML quotation template with company branding sections
  - [x] 2.3 Develop CSS styles for print-optimized layout with proper typography and spacing
  - [x] 2.4 Implement async PDF generation function using WeasyPrint with error handling
  - [x] 2.5 Add template data injection for customer info, vehicle specs, pricing, and employee details
  - [x] 2.6 Create frontend PDF testing interface with sample data input forms
  - [x] 2.7 Add backend API endpoint for PDF generation testing (/api/test-pdf-generation)
  - [x] 2.8 Test PDF generation with various sample data scenarios and validate output quality

- [x] 3.0 Supabase Storage Integration
  - [x] 3.1 Implement PDF upload function to quotations bucket with proper metadata
  - [x] 3.2 Create signed URL generation for secure, time-limited quotation sharing (24-48 hours)
  - [x] 3.3 Add file naming convention: quotation_{customer_id}_{timestamp}.pdf
  - [x] 3.4 Implement error handling for storage failures and retry logic
  - [x] 3.5 Add quotation cleanup job for expired files (optional background task)
  - [x] 3.6 Simulate upload and cleanup to validate deletion safety and reliability
    - [x] 3.6.1 Upload a test PDF using `upload_quotation_pdf()` and record returned `path`
    - [x] 3.6.2 Create or upsert a `quotations` row with `pdf_url` pointing to `/quotations/{path}` and `expires_at` in the past; status in [draft|pending|sent|viewed|expired|rejected]
    - [x] 3.6.3 Run `cleanup_expired_quotations()` and assert the file at `path` is deleted and DB updated (`pdf_url` NULL, `status` = expired)
    - [x] 3.6.4 Safety check: create control rows/files that must NOT be deleted (e.g., `status = accepted`, or `expires_at` in future) and verify they remain intact
    - [x] 3.6.5 Edge case: set a malformed `pdf_url` (no `/quotations/` segment) and verify cleanup skips deletion
    - [x] 3.6.6 Clean up test artifacts (delete test DB rows if needed)

- [x] 4.0 Database Helper Functions Development
  - [x] 4.1 Implement _lookup_customer() with flexible search (name, email, phone, company)
  - [x] 4.2 Create _lookup_vehicle_by_criteria() for vehicle specs, availability, and stock queries
  - [x] 4.3 Develop _lookup_current_pricing() for real-time pricing, discounts, and add-ons
  - [x] 4.4 Build _lookup_employee_details() for salesperson information and branch data
  - [x] 4.5 Add comprehensive error handling and data validation for all helper functions
  - [x] 4.6 Implement lightweight query optimization (keep code simple)
    - [x] 4.6.1 Replace any remaining `select("*")` with explicit column lists in helpers (e.g., pricing)
    - [x] 4.6.2 Apply limits/order only when semantically correct (e.g., single-record lookups like pricing); avoid limits that could truncate user-facing results
  - [x] 4.7 Generate comprehensive tests for helper functions per PRD (validate inputs, edge cases, and expected outputs)
    - [x] 4.7.1 Test scaffolding & fixtures
      - [x] Create test scaffolding with env guards (skip when `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` are missing)
      - [x] Prefer lightweight fixtures or rely on seeded CRM data; avoid heavy setup/teardown
    - [x] 4.7.2 `_lookup_customer()` tests
      - [x] Exact match by UUID, email, phone, and full name
      - [x] Partial/fuzzy name and company search; multi‑word queries
      - [x] Phone normalization (formatted vs digits‑only)
      - [x] Email domain search behavior
      - [x] Not‑found handling and returned `None`
      - [x] Performance: repeated calls do not error and return promptly
    - [x] 4.7.3 `_lookup_vehicle_by_criteria()` tests
      - [x] Filter by make/model/type/year/color
      - [x] Availability/stock filters and ranges
      - [x] Combined criteria correctness; deterministic ordering if applicable
      - [x] Not‑found handling
    - [x] 4.7.4 `_lookup_current_pricing()` tests
      - [x] Base price, discounts, insurance, LTO fees, add‑ons; computed totals align with PRD expectations
      - [x] Active/valid pricing selection; handles missing/legacy fields gracefully
      - [x] Edge cases: zero/negative discounts, empty add‑ons
    - [x] 4.7.5 `_lookup_employee_details()` tests
      - [x] Lookup by id/email/name; includes branch details
      - [x] Not‑found handling
    - [x] 4.7.6 Error‑handling & resilience
      - [x] Invalid inputs (empty/whitespace, malformed ids) return safe results
      - [x] Simulated DB error surfaces user‑friendly messages without crashes (monkeypatch client to raise)
    - [x] 4.7.7 Non‑functional checks
      - [x] Read‑only behavior (no writes performed by helpers)
      - [x] Connection stability under repeated calls (no leaks/exceptions)

- [ ] 5.0 Generate Quotation Tool Implementation
  - [x] 5.1 Create generate_quotation tool with proper @tool decorator and BaseModel schema
  - [x] 5.2 Implement conversation context extraction using extract_fields_from_conversation()
  - [x] 5.3 Add HITL flow for missing information gathering using request_input()
  - [x] 5.3.1 Fix HITL resume logic for proper agent architecture compliance
    - [x] 5.3.1.1 Add resume parameters to generate_quotation tool signature (state preservation)
    - [x] 5.3.1.2 Implement resume detection logic at tool start (check for HITL continuation)
    - [x] 5.3.1.3 Add complete state preservation in all HITL contexts (intermediate results)
    - [x] 5.3.1.4 Implement multi-step HITL flow support (sequential interactions)
    - [x] 5.3.1.5 Add resume logic for customer lookup HITL flow
    - [x] 5.3.1.6 Add resume logic for vehicle requirements HITL flow  
    - [x] 5.3.1.7 Add resume logic for employee data HITL flow
    - [x] 5.3.1.8 Add resume logic for missing information HITL flow
    - [x] 5.3.1.9 Add resume logic for pricing issues HITL flow
    - [x] 5.3.1.10 Test resume logic with sample HITL scenarios
  - [x] 5.4 Implement quotation preview and confirmation using request_approval() pattern
  - [ ] 5.5 Integrate all helper functions for data retrieval and validation
  - [ ] 5.6 Add comprehensive error handling and user-friendly error messages
  - [ ] 5.7 Implement employee-only access control using existing user verification patterns
  - [x] 5.8 Replace hard-coded vehicle parsing with intelligent LLM-based approach
    - [x] 5.8.1 Create _parse_vehicle_requirements_with_llm() function for intelligent parsing
    - [x] 5.8.2 Implement _get_available_makes_and_models() for dynamic inventory lookup
    - [x] 5.8.3 Add _enhance_vehicle_criteria_with_fuzzy_matching() for better matching
    - [x] 5.8.4 Create _find_closest_match() helper for string similarity matching
    - [x] 5.8.5 Add _generate_inventory_suggestions() for dynamic fallback messages
    - [x] 5.8.6 Remove hard-coded if/elif vehicle parsing logic (lines 2677-2707)
    - [x] 5.8.7 Replace static fallback message with dynamic inventory-aware message
    - [x] 5.8.8 Update generate_quotation tool to use new intelligent parsing functions

- [ ] 6.0 Agent Integration and Testing
  - [ ] 6.1 Add generate_quotation tool to employee agent toolset in agent.py
  - [ ] 6.2 Update agent tool descriptions to clearly differentiate from simple_query_crm_data
  - [ ] 6.3 Create comprehensive unit tests for all helper functions
  - [ ] 6.4 Develop integration tests for complete quotation generation flow
  - [ ] 6.5 Test HITL flows with various scenarios (missing data, approval/rejection)
  - [ ] 6.6 Validate PDF quality and professional appearance with real CRM data
  - [ ] 6.7 Performance testing for PDF generation and storage operations
  - [ ] 6.8 Security testing for storage access controls and signed URLs