-- Populate users table with employee users (employee_id set, customer_id null)
INSERT INTO users (user_id, employee_id, customer_id, email, display_name, user_type, is_active, created_at, updated_at) VALUES
-- Employee users
('alex.thompson', 'db8cdcda-3c96-4e35-8fe4-860f927b9395', NULL, 'alex.thompson@company.com', 'Alex Thompson', 'employee', true, NOW(), NOW()),
('amanda.white', '8764cfb9-bca1-4bfc-9f26-b85ea4c5a36e', NULL, 'amanda.white@company.com', 'Amanda White', 'employee', true, NOW(), NOW()),
('carlos.rodriguez', 'ecb3092d-58da-43d7-8d5f-27ade0cbd0d4', NULL, 'carlos.rodriguez@company.com', 'Carlos Rodriguez', 'employee', true, NOW(), NOW()),
('christopher.lee', '59b46c2a-81e4-4440-807e-c78593eefb58', NULL, 'christopher.lee@company.com', 'Christopher Lee', 'employee', true, NOW(), NOW()),
('david.brown', 'd193a8ac-e124-4fb3-88eb-d1bdc8eba79c', NULL, 'david.brown@company.com', 'David Brown', 'employee', true, NOW(), NOW()),
('david.chen', '8a994198-73e9-46da-8a3f-eab65938bf16', NULL, 'david.chen@company.com', 'David Chen', 'employee', true, NOW(), NOW()),
('emily.rodriguez', 'cebe246f-7650-4405-ab06-7714c2c8c25f', NULL, 'emily.rodriguez@company.com', 'Emily Rodriguez', 'employee', true, NOW(), NOW()),
('emma.wilson', 'e1682094-c232-4b3d-b1d8-de98de4427e5', NULL, 'emma.wilson@company.com', 'Emma Wilson', 'employee', true, NOW(), NOW()),
('james.anderson', '3bfe59a3-e22b-4401-9161-6090c788326e', NULL, 'james.anderson@company.com', 'James Anderson', 'employee', true, NOW(), NOW()),
('jennifer.taylor', 'cb951bd6-083a-4463-89a2-6e91149e47af', NULL, 'jennifer.taylor@company.com', 'Jennifer Taylor', 'employee', true, NOW(), NOW()),
('john.smith', 'cd15ede1-8102-4831-8695-e0374fc160ba', NULL, 'john.smith@company.com', 'John Smith', 'employee', true, NOW(), NOW()),
('kevin.martinez', 'fd90835b-2255-444f-b53c-b9693f87e4dd', NULL, 'kevin.martinez@company.com', 'Kevin Martinez', 'employee', true, NOW(), NOW()),
('lisa.chen', '8dbf462e-0a5b-4150-b0f0-c8175ee6c760', NULL, 'lisa.chen@company.com', 'Lisa Chen', 'employee', true, NOW(), NOW()),
('lisa.wang', 'eece00c2-126e-4c55-b93c-f754a3525bc6', NULL, 'lisa.wang@company.com', 'Lisa Wang', 'employee', true, NOW(), NOW()),
('maria.garcia', 'caf0ebba-1e05-4a5a-b7a9-7874b0765e4b', NULL, 'maria.garcia@company.com', 'Maria Garcia', 'employee', true, NOW(), NOW()),
('michael.johnson', '9fa0ad0f-0ad0-42f2-a3b7-97e3a930dd32', NULL, 'michael.johnson@company.com', 'Michael Johnson', 'employee', true, NOW(), NOW()),
('mike.davis', '6edb95c4-7a98-4a6a-a8b9-e7dbd8023490', NULL, 'mike.davis@company.com', 'Mike Davis', 'employee', true, NOW(), NOW()),
('robert.davis', 'ca2727c3-68c0-4e9f-a2f4-8532a79586d3', NULL, 'robert.davis@company.com', 'Robert Davis', 'employee', true, NOW(), NOW()),
('sarah.johnson', 'aca0db3a-1e5f-49f1-a39e-59252950ef2a', NULL, 'sarah.johnson@company.com', 'Sarah Johnson', 'employee', true, NOW(), NOW()),
('sarah.wilson', '3f599d6f-607f-4afe-b83c-7e57d0e3d3dc', NULL, 'sarah.wilson@company.com', 'Sarah Wilson', 'employee', true, NOW(), NOW()),

-- Customer users (customer_id set, employee_id null)
('alice.johnson.customer', NULL, '2d91dc76-3cf0-42d9-a7ff-2f511334658a', 'alice.johnson@email.com', 'Alice Johnson', 'customer', true, NOW(), NOW()),
('bob.smith.customer', NULL, 'dcb2838f-a93c-4b12-94e3-909cd734013e', 'bob.smith@personal.com', 'Bob Smith', 'customer', true, NOW(), NOW()),
('carol.thompson.customer', NULL, 'c11bff5a-207c-409f-90c5-ce446de91d88', 'carol.thompson@logistics.com', 'Carol Thompson', 'customer', true, NOW(), NOW()),
('daniel.wilson.customer', NULL, 'ed545112-f687-42b8-95de-8891cf28e1f3', 'daniel.wilson@gmail.com', 'Daniel Wilson', 'customer', true, NOW(), NOW()),
('eva.martinez.customer', NULL, 'ab296284-b214-40e2-a961-9084c01be3ab', 'eva.martinez@startup.com', 'Eva Martinez', 'customer', true, NOW(), NOW()),
('frank.davis.customer', NULL, 'bfc306bf-d7bb-403b-9996-ab03576810c5', 'frank.davis@yahoo.com', 'Frank Davis', 'customer', true, NOW(), NOW()),
('grace.lee.customer', NULL, 'f5f96b86-4dc0-45fa-8fbd-b34d28306a19', 'grace.lee@consulting.com', 'Grace Lee', 'customer', true, NOW(), NOW()),
('henry.clark.customer', NULL, 'f81f6038-4c56-4337-b0fd-53307836c793', 'henry.clark@outlook.com', 'Henry Clark', 'customer', true, NOW(), NOW()),

-- Special test user from the original issue (should be able to access as employee)
('123e4567-e89b-12d3-a456-426614174000', 'cd15ede1-8102-4831-8695-e0374fc160ba', NULL, 'test.employee@company.com', 'Test Employee User', 'employee', true, NOW(), NOW()); 