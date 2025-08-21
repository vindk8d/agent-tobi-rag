# Product Requirements Document: Vehicle Specification Upload System

## Introduction/Overview

The Vehicle Specification Upload System enables sales teams to upload and manage comprehensive vehicle specification documents for the RAG-powered sales copilot. This feature allows users to upload detailed vehicle documentation (specifications, features, technical details) that will be processed, chunked, and embedded to enhance the AI agent's ability to answer customer queries about specific vehicles with accurate, up-to-date information.

**Problem:** Currently, the RAG system has general vehicle data but lacks detailed specification documents that sales agents need to answer specific customer questions about vehicle features, technical specifications, and capabilities.

**Goal:** Create a streamlined upload system that processes vehicle specification documents into searchable, contextual knowledge for the sales copilot.

## Goals

1. **Enable vehicle-specific document uploads** - Sales teams can upload specification documents for individual vehicles
2. **Maintain data integrity** - One authoritative specification document per vehicle with backup/versioning
3. **Optimize RAG retrieval** - Vehicle specifications are properly chunked and embedded for precise question answering
4. **Provide intuitive interface** - Simple dropdown selection and upload process
5. **Ensure data reliability** - Direct foreign key relationships between vehicles and their documentation

## User Stories

1. **As a sales manager**, I want to upload the latest Ford Bronco specification sheet so that the AI agent can answer detailed customer questions about its features and capabilities.

2. **As a sales agent**, I want to quickly find which vehicles have specification documents uploaded so I know when the AI can provide detailed technical answers.

3. **As a sales manager**, I want to replace outdated vehicle specifications with new versions while keeping a backup of the previous version for reference.

4. **As a customer service representative**, I want the AI to access accurate vehicle specifications when customers ask detailed questions about engine power, safety features, or dimensions.

5. **As a system administrator**, I want old replaced documents to be automatically cleaned up after 30 days to manage storage costs.

## Functional Requirements

### Core Upload Functionality
1. **Vehicle Selection Interface** - The system must provide a dropdown menu populated with all available vehicles from the vehicles table (brand, model, year, variant).

2. **File Upload Support** - The system must accept text files including PDF, Word documents (.docx), Markdown (.md), and plain text (.txt) files up to 50MB.

3. **Document Replacement Logic** - The system must detect existing vehicle specifications and prompt for confirmation before overwriting with a clear warning message.

4. **Backup Management** - The system must move replaced documents to a `documents/vehicles/{vehicle_id}/replaced/` folder with timestamp prefixes before uploading new documents.

5. **Storage Organization** - The system must store vehicle documents in organized paths: `documents/vehicles/{vehicle_id}/current/{filename}`.

### Document Processing
6. **Section-Based Chunking** - The system must split vehicle specification documents by major sections (H2 headers: ##) to preserve semantic meaning.

7. **Vehicle Context Enhancement** - The system must add vehicle identification (brand, model, year) to each document chunk for improved retrieval accuracy.

8. **Foreign Key Relationships** - The system must store `vehicle_id` as a foreign key in both `documents` and `document_chunks` tables for direct relational mapping.

9. **Embedding Generation** - The system must generate vector embeddings for each document chunk with vehicle-specific metadata for filtered retrieval.

10. **Data Source Creation** - The system must automatically create vehicle-specific data sources with naming convention: "{Brand} {Model} {Year} Specifications".

### Data Management
11. **Automatic Cleanup** - The system must automatically delete files from the `replaced/` folder after 30 days to manage storage costs.

12. **Document Type Classification** - The system must classify uploaded vehicle documents with document_type = 'vehicle_specification'.

13. **Upload Progress Tracking** - The system must provide real-time upload progress and processing status feedback to users.

14. **Error Handling** - The system must handle upload failures gracefully and provide clear error messages for file size limits, unsupported formats, or processing errors.

## Non-Goals (Out of Scope)

1. **Multiple documents per vehicle** - This feature supports only one comprehensive specification document per vehicle
2. **Access control/permissions** - Authentication and authorization are deprioritized for initial implementation
3. **Document versioning UI** - No interface for browsing historical versions of replaced documents
4. **Quotation system integration** - Direct integration with quotation generation is not included in this scope
5. **Bulk upload functionality** - Mass upload of multiple vehicle specifications is not supported
6. **Document editing capabilities** - No in-system editing of uploaded specifications
7. **Document approval workflows** - No review/approval process for uploaded specifications

## Design Considerations

### Frontend Interface
- **Location**: `/frontend/app/dev/specupload/page.tsx`
- **Layout**: Clean, minimal interface following existing dev page patterns
- **Vehicle Dropdown**: Searchable dropdown with format: "Brand Model Year (Variant)"
- **Upload Area**: Drag-and-drop file upload with file type validation
- **Confirmation Modal**: Clear warning when replacing existing documents
- **Status Display**: Upload progress bar and processing status indicators

### Backend API Endpoints
- `GET /api/vehicles` - List all available vehicles for dropdown population
- `POST /api/vehicles/{vehicle_id}/specification` - Upload vehicle specification document
- `GET /api/vehicles/{vehicle_id}/specification` - Check if vehicle has existing specification
- `DELETE /api/vehicles/{vehicle_id}/specification` - Remove vehicle specification (moves to backup)

## Technical Considerations

### Database Schema
- **Leverage existing tables** - Use current `documents`, `document_chunks`, and `embeddings` tables with vehicle_id foreign keys
- **Migration dependency** - Verify `document_chunks` table exists from previous migration `20250109000000_restructure_for_clarity.sql`
- **Referential integrity** - Cascade deletes when vehicles are removed

### Document Processing Pipeline
- **Extend existing pipeline** - Modify `DocumentProcessingPipeline` in `backend/rag/pipeline.py`
- **Section-aware chunking** - Split on H2 headers (`## `) to preserve document structure
- **Vehicle context injection** - Prepend vehicle information to chunk content for better embeddings
- **Metadata enhancement** - Add vehicle_id to embedding metadata for filtered retrieval

### Storage Infrastructure
- **Use existing 'documents' bucket** - No new storage buckets needed
- **Organized folder structure** - Vehicle-specific subfolders for easy management
- **Backup retention** - 30-day retention policy for replaced documents

### RAG Enhancement
- **Vehicle-scoped retrieval** - Filter embeddings by vehicle_id before semantic search
- **Contextual responses** - Include vehicle context in all specification-related answers
- **Reliable mapping** - Direct foreign key relationships ensure accurate vehicle-document associations

## Success Metrics

1. **Upload Success Rate** - 95%+ successful uploads without errors
2. **Processing Time** - Vehicle specifications processed and available for RAG queries within 2 minutes of upload
3. **RAG Accuracy** - AI agent provides accurate vehicle-specific answers when specifications are uploaded
4. **Storage Efficiency** - Automated cleanup reduces storage costs by removing old backup files
5. **User Experience** - Sales team adoption rate of 80%+ for uploading vehicle specifications

## Open Questions

1. **File naming conventions** - Should we enforce specific naming patterns for uploaded files or allow any filename?
2. **Document validation** - Should we validate document content to ensure it contains vehicle specifications?
3. **Notification system** - Should users be notified when document processing completes?
4. **Integration timeline** - When should this system integrate with the quotation generation feature?
5. **Batch operations** - Future consideration for bulk upload/replacement of multiple vehicle specifications?

## Implementation Notes

### Database Changes Required
```sql
-- Add vehicle_id foreign key to documents table
ALTER TABLE documents ADD COLUMN vehicle_id UUID REFERENCES vehicles(id) ON DELETE CASCADE;
CREATE INDEX idx_documents_vehicle_id ON documents(vehicle_id);

-- Add vehicle_id foreign key to document_chunks table  
ALTER TABLE document_chunks ADD COLUMN vehicle_id UUID REFERENCES vehicles(id) ON DELETE CASCADE;
CREATE INDEX idx_document_chunks_vehicle_id ON document_chunks(vehicle_id);

-- Add vehicle_specification to document_type enum
ALTER TYPE document_type ADD VALUE 'vehicle_specification';
```

### Storage Folder Structure
```
documents/
  vehicles/
    {vehicle_id}/
      current/
        {original_filename}
      replaced/
        {timestamp}_{original_filename}
```

### Cleanup Schedule
- **Automated task**: Daily check for files in `replaced/` folders older than 30 days
- **Deletion process**: Permanent removal from Supabase Storage
- **Logging**: Track cleanup operations for audit purposes

---

*This PRD focuses on creating a reliable, simple vehicle specification upload system that enhances the RAG capabilities of the sales copilot through proper document organization, processing, and retrieval.*
