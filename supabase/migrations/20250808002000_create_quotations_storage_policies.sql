-- Migration: Create Storage Helper Functions for Quotations
-- This migration creates helper functions for quotations storage management
-- Note: The quotations bucket has been created via API with the following settings:
-- - Name: quotations
-- - Public: false (private bucket) 
-- - File size limit: 10MB
-- - Allowed MIME types: application/pdf
-- Storage policies are managed at the application level for better flexibility

-- Create a function to generate signed URLs for quotations
CREATE OR REPLACE FUNCTION generate_quotation_signed_url(
    quotation_id UUID,
    expires_in_hours INTEGER DEFAULT 48
)
RETURNS TEXT AS $$
DECLARE
    employee_id UUID;
    file_path TEXT;
    signed_url TEXT;
BEGIN
    -- Get the employee_id and pdf_url for the quotation
    SELECT q.employee_id, q.pdf_url
    INTO employee_id, file_path
    FROM quotations q
    WHERE q.id = quotation_id;
    
    -- Check if quotation exists and user has access
    IF employee_id IS NULL THEN
        RAISE EXCEPTION 'Quotation not found or access denied';
    END IF;
    
    -- Verify the requesting user is the employee who created the quotation
    IF NOT EXISTS (
        SELECT 1 FROM employees e
        WHERE e.id = employee_id
        AND e.email = (SELECT email FROM users WHERE id = auth.uid())
    ) THEN
        RAISE EXCEPTION 'Access denied: You can only access quotations you created';
    END IF;
    
    -- Extract the storage path from the full URL
    -- Assuming pdf_url format: https://[project].supabase.co/storage/v1/object/public/quotations/path
    file_path := REGEXP_REPLACE(file_path, '^.*/quotations/', '');
    
    -- Note: This is a placeholder for the actual signed URL generation
    -- In practice, you would use the Supabase client library to generate signed URLs
    -- This function serves as documentation for the intended behavior
    RETURN 'signed_url_placeholder_' || file_path || '_expires_' || (NOW() + INTERVAL '1 hour' * expires_in_hours)::text;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION generate_quotation_signed_url(UUID, INTEGER) TO authenticated;

-- Comments for documentation
COMMENT ON FUNCTION generate_quotation_signed_url(UUID, INTEGER) IS 'Generate time-limited signed URLs for quotation PDFs with proper access control';

-- Create a helper function to get the proper storage path for quotations
CREATE OR REPLACE FUNCTION get_quotation_storage_path(
    employee_id UUID,
    quotation_number TEXT
)
RETURNS TEXT AS $$
BEGIN
    -- Return path format: {employee_id}/quotation_{quotation_number}_{timestamp}.pdf
    RETURN employee_id::text || '/quotation_' || quotation_number || '_' || EXTRACT(EPOCH FROM NOW())::bigint || '.pdf';
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION get_quotation_storage_path(UUID, TEXT) TO authenticated;

COMMENT ON FUNCTION get_quotation_storage_path(UUID, TEXT) IS 'Generate consistent storage paths for quotation PDFs organized by employee';