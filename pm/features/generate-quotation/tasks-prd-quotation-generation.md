# Task List: AI Agent Quotation Generation System

Based on the PRD for the AI Agent Quotation Generation System, this document outlines the implementation tasks required to build the feature.

## Relevant Files

- `backend/agents/tools.py` - Add the generate_quotation tool and database helper functions
- `backend/agents/tools.py` - Add helper functions: _lookup_customer, _lookup_vehicle_by_criteria, _lookup_current_pricing, _lookup_employee_details
- `backend/core/pdf_generator.py` - Comprehensive PDF generation module with WeasyPrint, template rendering, and async support
- `backend/core/storage.py` - New module for Supabase quotations bucket integration
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

### Notes

- WeasyPrint requires specific HTML/CSS structure for professional PDF output
- Supabase storage integration should follow existing document upload patterns
- HITL flows must integrate with existing request_approval() and request_input() patterns
- Database helper functions should be reusable across multiple tools
- Tests should cover both successful flows and error scenarios

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

- [ ] 3.0 Supabase Storage Integration
  - [ ] 3.1 Implement PDF upload function to quotations bucket with proper metadata
  - [ ] 3.2 Create signed URL generation for secure, time-limited quotation sharing (24-48 hours)
  - [ ] 3.3 Add file naming convention: quotation_{customer_id}_{timestamp}.pdf
  - [ ] 3.4 Implement error handling for storage failures and retry logic
  - [ ] 3.5 Add quotation cleanup job for expired files (optional background task)

- [ ] 4.0 Database Helper Functions Development
  - [ ] 4.1 Implement _lookup_customer() with flexible search (name, email, phone, company)
  - [ ] 4.2 Create _lookup_vehicle_by_criteria() for vehicle specs, availability, and stock queries
  - [ ] 4.3 Develop _lookup_current_pricing() for real-time pricing, discounts, and add-ons
  - [ ] 4.4 Build _lookup_employee_details() for salesperson information and branch data
  - [ ] 4.5 Add comprehensive error handling and data validation for all helper functions
  - [ ] 4.6 Implement connection pooling and query optimization for performance

- [ ] 5.0 Generate Quotation Tool Implementation
  - [ ] 5.1 Create generate_quotation tool with proper @tool decorator and BaseModel schema
  - [ ] 5.2 Implement conversation context extraction using extract_fields_from_conversation()
  - [ ] 5.3 Add HITL flow for missing information gathering using request_input()
  - [ ] 5.4 Implement quotation preview and confirmation using request_approval() pattern
  - [ ] 5.5 Integrate all helper functions for data retrieval and validation
  - [ ] 5.6 Add comprehensive error handling and user-friendly error messages
  - [ ] 5.7 Implement employee-only access control using existing user verification patterns

- [ ] 6.0 Agent Integration and Testing
  - [ ] 6.1 Add generate_quotation tool to employee agent toolset in agent.py
  - [ ] 6.2 Update agent tool descriptions to clearly differentiate from simple_query_crm_data
  - [ ] 6.3 Create comprehensive unit tests for all helper functions
  - [ ] 6.4 Develop integration tests for complete quotation generation flow
  - [ ] 6.5 Test HITL flows with various scenarios (missing data, approval/rejection)
  - [ ] 6.6 Validate PDF quality and professional appearance with real CRM data
  - [ ] 6.7 Performance testing for PDF generation and storage operations
  - [ ] 6.8 Security testing for storage access controls and signed URLs