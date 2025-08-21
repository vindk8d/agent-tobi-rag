-- Add Ford Bronco Outer Banks with Sasquatch Package to vehicles table
-- Based on the specification document we've been working with

-- First check if Ford Bronco already exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM vehicles 
        WHERE brand = 'Ford' 
        AND model = 'Bronco' 
        AND year = 2024 
        AND type = 'suv'
    ) THEN
        INSERT INTO vehicles (
            brand,
            model, 
            year,
            type,
            engine_type,
            variant,
            key_features,
            color,
            acceleration,
            power_ps,
            torque_nm,
            transmission,
            is_available,
            stock_quantity,
            is_autohub
        ) VALUES (
            'Ford',
            'Bronco',
            2024,
            'suv',
            'Gasoline',
            'Outer Banks with Sasquatch Package',
            'Advanced 4x4 with Automatic On-Demand Engagement, Terrain Management System with G.O.A.T. Modes, BILSTEIN Position-Sensitive Shock Absorbers, Electronic-Locking Front and Rear Axle, Removable Roof and Doors, 35-inch Mud-Terrain Tires, 850mm Water Wading Capability',
            'Cactus Gray',
            null, -- acceleration not specified in the document
            335, -- 335 PS @ 5,570 rpm
            555, -- 555 Nm @ 3,000 rpm  
            '10-Speed Automatic Transmission',
            true,
            2, -- reasonable stock quantity
            true -- assuming this is an AutoHub vehicle
        );
        RAISE NOTICE 'Ford Bronco 2024 SUV added successfully';
    ELSE
        RAISE NOTICE 'Ford Bronco 2024 SUV already exists, skipping insert';
    END IF;
END $$;
