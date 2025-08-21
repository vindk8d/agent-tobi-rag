-- Fix the sync_document_chunk_vehicle_id function to properly handle different tables
-- The issue is that the function tries to access NEW.document_id even when triggered on the documents table

CREATE OR REPLACE FUNCTION sync_document_chunk_vehicle_id()
RETURNS TRIGGER AS $$
BEGIN
    -- When document vehicle_id changes, update all its chunks
    -- This applies when the trigger is fired on the documents table
    IF TG_OP = 'UPDATE' AND TG_TABLE_NAME = 'documents' AND OLD.vehicle_id IS DISTINCT FROM NEW.vehicle_id THEN
        UPDATE document_chunks 
        SET vehicle_id = NEW.vehicle_id 
        WHERE document_id = NEW.id;
    END IF;
    
    -- When inserting a new chunk, inherit vehicle_id from document
    -- This applies when the trigger is fired on the document_chunks table
    IF TG_OP = 'INSERT' AND TG_TABLE_NAME = 'document_chunks' AND NEW.document_id IS NOT NULL THEN
        SELECT vehicle_id INTO NEW.vehicle_id 
        FROM documents 
        WHERE id = NEW.document_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
