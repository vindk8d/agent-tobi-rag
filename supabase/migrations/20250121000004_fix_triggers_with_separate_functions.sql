-- Fix the trigger issue by creating separate, focused functions
-- This is safer and more maintainable than one multi-purpose function

-- Function 1: Handle document vehicle_id updates (propagate to chunks)
CREATE OR REPLACE FUNCTION sync_document_vehicle_id_to_chunks()
RETURNS TRIGGER AS $$
BEGIN
    -- Only handle document updates where vehicle_id changes
    IF TG_OP = 'UPDATE' AND OLD.vehicle_id IS DISTINCT FROM NEW.vehicle_id THEN
        UPDATE document_chunks 
        SET vehicle_id = NEW.vehicle_id 
        WHERE document_id = NEW.id;
        
        RAISE LOG 'Updated % chunks for document % with new vehicle_id %', 
            (SELECT COUNT(*) FROM document_chunks WHERE document_id = NEW.id),
            NEW.id, 
            NEW.vehicle_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function 2: Handle chunk insertion (inherit vehicle_id from document)
CREATE OR REPLACE FUNCTION inherit_vehicle_id_from_document()
RETURNS TRIGGER AS $$
BEGIN
    -- Only handle chunk inserts where document_id is provided
    IF TG_OP = 'INSERT' AND NEW.document_id IS NOT NULL THEN
        -- Get vehicle_id from parent document
        SELECT vehicle_id INTO NEW.vehicle_id 
        FROM documents 
        WHERE id = NEW.document_id;
        
        RAISE LOG 'Chunk % inherited vehicle_id % from document %', 
            NEW.id, 
            NEW.vehicle_id, 
            NEW.document_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop the old problematic function and triggers
DROP TRIGGER IF EXISTS trigger_sync_document_vehicle_id ON documents;
DROP TRIGGER IF EXISTS trigger_inherit_document_vehicle_id ON document_chunks;
DROP FUNCTION IF EXISTS sync_document_chunk_vehicle_id();

-- Create new, focused triggers
CREATE TRIGGER trigger_sync_document_vehicle_id
    AFTER UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION sync_document_vehicle_id_to_chunks();

CREATE TRIGGER trigger_inherit_document_vehicle_id
    BEFORE INSERT ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION inherit_vehicle_id_from_document();
