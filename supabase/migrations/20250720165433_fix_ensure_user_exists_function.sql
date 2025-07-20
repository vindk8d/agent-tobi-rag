-- Fix ensure_user_exists function after user_id → username rename
-- This function was referencing the old user_id column

CREATE OR REPLACE FUNCTION public.ensure_user_exists(input_user_id text)
RETURNS uuid
LANGUAGE plpgsql
AS $function$
DECLARE
    found_user_id UUID;
    employee_data RECORD;
BEGIN
    -- Check if user already exists - look by both username and id
    SELECT id INTO found_user_id 
    FROM users 
    WHERE username = input_user_id OR id::text = input_user_id;
    
    IF found_user_id IS NOT NULL THEN
        RETURN found_user_id;
    END IF;
    
    -- Try to find matching employee by email
    SELECT id, name, email INTO employee_data 
    FROM employees 
    WHERE email = input_user_id OR email = input_user_id || '@system.local';
    
    -- Create user with employee link if found
    IF employee_data.id IS NOT NULL THEN
        INSERT INTO users (username, employee_id, email, display_name, user_type)
        VALUES (
            input_user_id,
            employee_data.id,
            COALESCE(employee_data.email, input_user_id),
            employee_data.name,
            'employee'
        )
        RETURNING id INTO found_user_id;
    ELSE
        -- Create generic user
        INSERT INTO users (username, email, display_name, user_type)
        VALUES (
            input_user_id,
            CASE 
                WHEN input_user_id LIKE '%@%' THEN input_user_id 
                ELSE input_user_id || '@system.local' 
            END,
            input_user_id,
            'employee'
        )
        RETURNING id INTO found_user_id;
    END IF;
    
    RETURN found_user_id;
END;
$function$;

-- Add comment to clarify the function's purpose
COMMENT ON FUNCTION ensure_user_exists(text) IS 'Ensure user exists in users table. Accepts either users.id (UUID) or username (string). Creates user if not found.';
