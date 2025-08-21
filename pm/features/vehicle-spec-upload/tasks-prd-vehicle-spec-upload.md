# Task List: Vehicle Specification Upload System

Based on PRD: `prd-vehicle-spec-upload.md`

## Relevant Files

- `supabase/migrations/20250120000003_create_vehicle_spec_schema.sql` - **CREATED** - Migration to establish document_chunks table and vehicle relationships
- `backend/api/documents.py` - **EXTENDED** - Added vehicle endpoints to existing document router
- `tests/test_vehicle_specification_api.py` - **CREATED** - Comprehensive test suite for vehicle API endpoints (16 tests, all passing)
- `backend/rag/pipeline.py` - **MODIFY EXISTING** - Extend existing pipeline with vehicle_id support
- `backend/rag/document_loader.py` - **MODIFY EXISTING** - Add section-based chunking option to existing functions
- `backend/core/storage.py` - **EXTEND EXISTING** - Add vehicle path generation to existing storage functions
- `backend/models/document.py` - **EXTEND EXISTING** - Add vehicle fields to existing document models
- `frontend/app/dev/specupload/page.tsx` - **CREATED** - Vehicle specification upload interface with vehicle dropdown, file upload, progress tracking, and confirmation dialogs
- `frontend/app/dev/specupload/components/VehicleDropdown.tsx` - **NOT NEEDED** - Vehicle dropdown integrated directly into main page component
- `frontend/lib/api.ts` - **EXTENDED** - Added vehicle API methods (getVehicles, getVehicleSpecification, uploadVehicleSpecification, deleteVehicleSpecification)
- `frontend/app/dev/page.tsx` - **EXTENDED** - Added Vehicle Spec Upload tool to dev tools navigation
- `frontend/lib/fileUtils.ts` - **CREATED** - Shared file upload utilities extracted from manage page (validation, drag-and-drop, progress tracking)
- `backend/services/cleanup_service.py` - **NEW** - Automated cleanup service using existing patterns

### Notes

- Unit tests should be placed alongside the code files they are testing
- Use `npx jest [optional/path/to/test/file]` to run frontend tests
- Use `pytest [optional/path/to/test/file]` to run backend tests
- Migration must be applied before any other development work begins

## Tasks

- [x] 1.0 Database Schema Migration and Setup
  - [x] 1.1 Create new migration file `20250120000003_create_vehicle_spec_schema.sql`
  - [x] 1.2 Add document_chunks table with proper structure (id, data_source_id, vehicle_id, title, content, document_type, chunk_index, status, word_count, character_count, metadata, timestamps)
  - [x] 1.3 Add vehicle_id foreign key column to documents table
  - [x] 1.4 Add vehicle_id foreign key column to document_chunks table
  - [x] 1.5 Update embeddings table to reference document_chunk_id instead of document_id
  - [x] 1.6 Add 'vehicle_specification' to document_type enum
  - [x] 1.7 Create necessary indexes for performance (vehicle_id, document_type, status)
  - [x] 1.8 Apply migration using Supabase CLI (`supabase db push` or `supabase migration up`)
  - [x] 1.9 Verify schema changes and test foreign key relationships using Supabase MCP tools

- [x] 2.0 Backend API Development
  - [x] 2.1 Extend existing `backend/api/documents.py` with vehicle-specific endpoints
  - [x] 2.2 Add `GET /api/vehicles` endpoint to existing documents router for vehicle dropdown
  - [x] 2.3 Modify existing `ProcessUploadedRequest` to include optional `vehicle_id` parameter
  - [x] 2.4 Extend `process_uploaded_document()` function to handle vehicle-specific processing
  - [x] 2.5 Add vehicle document checking logic to existing document management functions
  - [x] 2.6 Reuse existing `delete_data_source()` function for vehicle document removal
  - [x] 2.7 Extend existing Pydantic models in `backend/models/document.py` with vehicle fields
  - [x] 2.8 Add vehicle-specific tests to existing `backend/api/documents.test.py`
  - [x] 2.9 Test extended endpoints using Supabase MCP for data verification

- [x] 3.0 Document Processing Pipeline Enhancement
  - [x] 3.1 Extend existing `DocumentProcessingPipeline.process_file()` to accept vehicle_id parameter
  - [x] 3.2 Modify existing `split_documents()` function in `document_loader.py` to support section-based chunking
  - [x] 3.3 Update existing `_store_document_chunk()` method to include vehicle_id foreign key
  - [x] 3.4 Enhance existing chunk metadata to include vehicle context for better embeddings
  - [x] 3.5 Extend existing `process_document_background()` to handle vehicle-specific data source creation
  - [x] 3.6 Reuse existing `reprocess_data_source()` logic for document replacement workflow
  - [x] 3.7 Modify existing embedding generation to include vehicle metadata for filtered retrieval
  - [x] 3.8 Add vehicle context injection as optional parameter to existing chunking functions
  - [x] 3.9 Extend existing pipeline unit tests to cover vehicle-specific processing
  - [x] 3.10 Test enhanced pipeline with vehicle context using existing test framework

- [ ] 4.0 Frontend Upload Interface Development
  - [x] 4.1 Create main upload page at `frontend/app/dev/specupload/page.tsx` using existing dev page patterns
  - [x] 4.2 Reuse existing file upload logic from `frontend/app/dev/manage/page.tsx` for vehicle specs
  - [x] 4.3 Extract and reuse existing `validateFile()` function from manage page
  - [x] 4.4 Adapt existing drag-and-drop upload interface for vehicle-specific use
  - [x] 4.5 Reuse existing upload progress and status tracking components
  - [x] 4.6 Build `VehicleDropdown.tsx` component using existing dropdown patterns
  - [x] 4.7 Extend existing `APIClient` methods in `frontend/lib/api.ts` for vehicle endpoints
  - [x] 4.8 Reuse existing error handling and user feedback patterns from manage page
  - [x] 4.9 Adapt existing file type validation and size limits for vehicle specifications
  - [x] 4.10 Test upload interface using existing upload testing patterns

- [ ] 5.0 Storage Management and Cleanup System
  - [x] 5.1 Extend existing `backend/core/storage.py` with vehicle-specific path generation functions
  - [x] 5.2 Reuse existing Supabase Storage patterns for organized vehicle folder structure
  - [x] 5.3 Adapt existing file backup patterns from quotation system for document replacement
  - [x] 5.4 Create cleanup service using existing storage management patterns
  - [x] 5.5 Reuse existing background task scheduling patterns for automated cleanup
  - [ ] 5.6 Extend existing logging patterns for cleanup audit trail
  - [ ] 5.7 Add vehicle storage tests to existing storage test suite
  - [ ] 5.8 Test backup and cleanup using existing Supabase Storage test patterns
  - [ ] 5.9 Verify storage organization using existing storage verification methods
