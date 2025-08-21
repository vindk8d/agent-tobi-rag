-- Recreate the triggers to ensure they use the updated function
-- Drop and recreate both triggers

DROP TRIGGER IF EXISTS trigger_sync_document_vehicle_id ON documents;
DROP TRIGGER IF EXISTS trigger_inherit_document_vehicle_id ON document_chunks;

-- Recreate the triggers
CREATE TRIGGER trigger_sync_document_vehicle_id
    AFTER UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION sync_document_chunk_vehicle_id();

CREATE TRIGGER trigger_inherit_document_vehicle_id
    BEFORE INSERT ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION sync_document_chunk_vehicle_id();
