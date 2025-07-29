-- Migration: Populate customer_warmth and customer_warmth_history with sample data
-- This creates realistic warmth data based on existing customer activity patterns

-- First, populate customer_warmth table with current warmth scores
-- High-activity customers (hot/warm)
INSERT INTO customer_warmth (
    customer_id, 
    overall_warmth_score, 
    warmth_level, 
    purchase_probability,
    engagement_score,
    interaction_frequency_score,
    response_rate_score,
    budget_qualification_score,
    timeline_urgency_score,
    decision_making_authority_score,
    demographic_fit_score,
    past_purchase_behavior_score,
    days_since_first_contact,
    days_since_last_interaction,
    expected_purchase_timeframe,
    total_interactions,
    meaningful_interactions,
    last_engagement_type,
    last_engagement_date,
    interested_vehicle_types,
    price_range_min,
    price_range_max,
    preferred_features,
    predicted_purchase_value,
    predicted_purchase_date,
    confidence_level,
    warmth_notes,
    scoring_rationale,
    calculated_by
) VALUES 
-- Jennifer Lee - HOT prospect (191 opportunities)
(
    'e623c347-d88e-4a0b-99ca-4f9f6ad7882e',
    85.5,
    'hot',
    78.0,
    90.0, -- engagement_score
    85.0, -- interaction_frequency_score
    92.0, -- response_rate_score
    75.0, -- budget_qualification_score
    88.0, -- timeline_urgency_score
    80.0, -- decision_making_authority_score
    85.0, -- demographic_fit_score
    70.0, -- past_purchase_behavior_score (new customer)
    45, -- days_since_first_contact
    3, -- days_since_last_interaction
    30, -- expected_purchase_timeframe (days)
    25, -- total_interactions
    18, -- meaningful_interactions
    'financing_inquiry',
    NOW() - INTERVAL '3 days',
    ARRAY['sedan', 'suv'],
    25000,
    45000,
    '{"automatic_transmission": true, "backup_camera": true, "bluetooth": true}',
    35000,
    CURRENT_DATE + INTERVAL '25 days',
    82.0,
    'Highly engaged customer with multiple inquiries and strong purchase intent',
    'High engagement rate and frequent meaningful interactions indicate strong purchase likelihood',
    'cebe246f-7650-4405-ab06-7714c2c8c25f' -- Emily Rodriguez (sales_agent)
),

-- Robert Brown - HOT prospect (160 opportunities + 1120 activities)
(
    'f390205f-9cf2-43fc-9338-3acd1bd90327',
    88.2,
    'scorching',
    85.0,
    95.0, -- engagement_score (highest due to activities)
    88.0, -- interaction_frequency_score
    90.0, -- response_rate_score
    85.0, -- budget_qualification_score
    92.0, -- timeline_urgency_score
    88.0, -- decision_making_authority_score
    82.0, -- demographic_fit_score
    75.0, -- past_purchase_behavior_score
    60, -- days_since_first_contact
    1, -- days_since_last_interaction
    14, -- expected_purchase_timeframe (days)
    45, -- total_interactions
    35, -- meaningful_interactions
    'test_drive',
    NOW() - INTERVAL '1 day',
    ARRAY['pickup', 'suv'],
    30000,
    55000,
    '{"4wd": true, "towing_capacity": true, "leather_seats": true}',
    48000,
    CURRENT_DATE + INTERVAL '10 days',
    90.0,
    'Extremely active customer with recent test drive - ready to purchase',
    'Combination of high activity count and recent test drive indicates immediate purchase intent',
    'db8cdcda-3c96-4e35-8fe4-860f927b9395' -- Alex Thompson (account_executive)
),

-- TechCorp Solutions - WARM business prospect (181 opportunities)
(
    '7287db95-f307-404e-aad9-2b619e3681f1',
    72.8,
    'warm',
    68.0,
    75.0, -- engagement_score
    80.0, -- interaction_frequency_score
    70.0, -- response_rate_score
    90.0, -- budget_qualification_score (business)
    65.0, -- timeline_urgency_score
    95.0, -- decision_making_authority_score (business)
    85.0, -- demographic_fit_score
    60.0, -- past_purchase_behavior_score
    90, -- days_since_first_contact
    7, -- days_since_last_interaction
    60, -- expected_purchase_timeframe (days)
    20, -- total_interactions
    14, -- meaningful_interactions
    'brochure_request',
    NOW() - INTERVAL '7 days',
    ARRAY['van', 'truck'],
    40000,
    80000,
    '{"fleet_discount": true, "commercial_warranty": true, "bulk_purchase": true}',
    65000,
    CURRENT_DATE + INTERVAL '45 days',
    75.0,
    'Business customer evaluating fleet purchase options',
    'Strong budget and authority but longer decision timeline typical for B2B purchases',
    'caf0ebba-1e05-4a5a-b7a9-7874b0765e4b' -- Maria Garcia (account_executive)
),

-- Daniel Wilson - WARM individual prospect (170 opportunities)
(
    'ed545112-f687-42b8-95de-8891cf28e1f3',
    70.5,
    'warm',
    65.0,
    78.0, -- engagement_score
    75.0, -- interaction_frequency_score
    68.0, -- response_rate_score
    72.0, -- budget_qualification_score
    70.0, -- timeline_urgency_score
    85.0, -- decision_making_authority_score
    80.0, -- demographic_fit_score
    65.0, -- past_purchase_behavior_score
    75, -- days_since_first_contact
    5, -- days_since_last_interaction
    45, -- expected_purchase_timeframe (days)
    18, -- total_interactions
    12, -- meaningful_interactions
    'callback_request',
    NOW() - INTERVAL '5 days',
    ARRAY['sedan', 'hatchback'],
    20000,
    35000,
    '{"fuel_efficiency": true, "safety_features": true, "warranty": true}',
    28000,
    CURRENT_DATE + INTERVAL '35 days',
    72.0,
    'Engaged customer comparing options with good purchase intent',
    'Consistent engagement and reasonable budget alignment suggest likely purchase',
    'cb951bd6-083a-4463-89a2-6e91149e47af' -- Jennifer Taylor (sales_agent)
),

-- Global Industries - LUKEWARM business prospect (170 opportunities)
(
    '3f30fe70-ee99-49e9-8ce5-51c5da8234cc',
    58.3,
    'lukewarm',
    52.0,
    60.0, -- engagement_score
    65.0, -- interaction_frequency_score
    55.0, -- response_rate_score
    85.0, -- budget_qualification_score
    45.0, -- timeline_urgency_score
    90.0, -- decision_making_authority_score
    70.0, -- demographic_fit_score
    50.0, -- past_purchase_behavior_score
    120, -- days_since_first_contact
    14, -- days_since_last_interaction
    90, -- expected_purchase_timeframe (days)
    15, -- total_interactions
    8, -- meaningful_interactions
    'website_visit',
    NOW() - INTERVAL '14 days',
    ARRAY['truck', 'van'],
    50000,
    100000,
    '{"commercial_grade": true, "extended_warranty": true}',
    75000,
    CURRENT_DATE + INTERVAL '75 days',
    58.0,
    'Business prospect in early evaluation phase with long timeline',
    'Good budget and authority but low urgency indicates longer sales cycle',
    'fd90835b-2255-444f-b53c-b9693f87e4dd' -- Kevin Martinez (account_executive)
),

-- Mark Johnson - LUKEWARM individual prospect (170 opportunities)
(
    'c6121296-25dc-44c0-8856-6726d242a9bc',
    55.8,
    'lukewarm',
    48.0,
    58.0, -- engagement_score
    60.0, -- interaction_frequency_score
    52.0, -- response_rate_score
    65.0, -- budget_qualification_score
    50.0, -- timeline_urgency_score
    80.0, -- decision_making_authority_score
    75.0, -- demographic_fit_score
    45.0, -- past_purchase_behavior_score
    100, -- days_since_first_contact
    10, -- days_since_last_interaction
    75, -- expected_purchase_timeframe (days)
    12, -- total_interactions
    6, -- meaningful_interactions
    'inquiry',
    NOW() - INTERVAL '10 days',
    ARRAY['suv', 'sedan'],
    25000,
    40000,
    '{"family_friendly": true, "safety_rating": true}',
    32000,
    CURRENT_DATE + INTERVAL '60 days',
    55.0,
    'Casual inquirer still in research phase',
    'Moderate engagement but slower response rate suggests longer consideration period',
    '59b46c2a-81e4-4440-807e-c78593eefb58' -- Christopher Lee (sales_agent)
),

-- Bob Smith - COOL prospect (168 opportunities but declining engagement)
(
    'dcb2838f-a93c-4b12-94e3-909cd734013e',
    42.5,
    'cool',
    35.0,
    45.0, -- engagement_score
    50.0, -- interaction_frequency_score
    40.0, -- response_rate_score
    60.0, -- budget_qualification_score
    30.0, -- timeline_urgency_score
    70.0, -- decision_making_authority_score
    65.0, -- demographic_fit_score
    35.0, -- past_purchase_behavior_score
    150, -- days_since_first_contact
    21, -- days_since_last_interaction
    120, -- expected_purchase_timeframe (days)
    20, -- total_interactions
    5, -- meaningful_interactions
    'website_visit',
    NOW() - INTERVAL '21 days',
    ARRAY['sedan'],
    15000,
    25000,
    '{"basic_features": true, "reliability": true}',
    20000,
    CURRENT_DATE + INTERVAL '90 days',
    45.0,
    'Previously engaged customer with declining interest',
    'Many initial touchpoints but recent low engagement suggests cooling interest',
    'e1682094-c232-4b3d-b1d8-de98de4427e5' -- Emma Wilson (sales_agent)
),

-- Henry Clark - COLD prospect (0 opportunities)
(
    'f81f6038-4c56-4337-b0fd-53307836c793',
    25.2,
    'cold',
    18.0,
    25.0, -- engagement_score
    20.0, -- interaction_frequency_score
    30.0, -- response_rate_score
    40.0, -- budget_qualification_score
    15.0, -- timeline_urgency_score
    60.0, -- decision_making_authority_score
    45.0, -- demographic_fit_score
    20.0, -- past_purchase_behavior_score
    200, -- days_since_first_contact
    45, -- days_since_last_interaction
    180, -- expected_purchase_timeframe (days)
    5, -- total_interactions
    1, -- meaningful_interactions
    'inquiry',
    NOW() - INTERVAL '45 days',
    ARRAY['sedan'],
    12000,
    20000,
    '{"budget_friendly": true}',
    15000,
    CURRENT_DATE + INTERVAL '150 days',
    30.0,
    'Low engagement prospect with budget constraints',
    'Minimal interaction and long silence period suggests low purchase probability',
    '8a994198-73e9-46da-8a3f-eab65938bf16' -- David Chen (sales_agent)
),

-- Eva Martinez/StartUp Dynamics - WARM business prospect (173 opportunities)
(
    'ab296284-b214-40e2-a961-9084c01be3ab',
    68.7,
    'warm',
    62.0,
    72.0, -- engagement_score
    70.0, -- interaction_frequency_score
    65.0, -- response_rate_score
    80.0, -- budget_qualification_score
    75.0, -- timeline_urgency_score
    85.0, -- decision_making_authority_score
    78.0, -- demographic_fit_score
    55.0, -- past_purchase_behavior_score
    80, -- days_since_first_contact
    6, -- days_since_last_interaction
    50, -- expected_purchase_timeframe (days)
    22, -- total_interactions
    16, -- meaningful_interactions
    'financing_inquiry',
    NOW() - INTERVAL '6 days',
    ARRAY['suv', 'sedan'],
    30000,
    50000,
    '{"startup_discount": true, "flexible_payment": true, "professional_image": true}',
    42000,
    CURRENT_DATE + INTERVAL '40 days',
    70.0,
    'Startup founder with growing business needs',
    'Good engagement and startup growth trajectory indicates solid purchase potential',
    '8764cfb9-bca1-4bfc-9f26-b85ea4c5a36e' -- Amanda White (account_executive)
),

-- Grace Lee/Strategic Consulting LLC - ICE COLD business prospect (0 opportunities)
(
    'f5f96b86-4dc0-45fa-8fbd-b34d28306a19',
    15.8,
    'ice_cold',
    8.0,
    15.0, -- engagement_score
    10.0, -- interaction_frequency_score
    20.0, -- response_rate_score
    70.0, -- budget_qualification_score (business has budget)
    5.0, -- timeline_urgency_score
    80.0, -- decision_making_authority_score
    30.0, -- demographic_fit_score
    10.0, -- past_purchase_behavior_score
    300, -- days_since_first_contact
    90, -- days_since_last_interaction
    365, -- expected_purchase_timeframe (days)
    3, -- total_interactions
    0, -- meaningful_interactions
    'website_visit',
    NOW() - INTERVAL '90 days',
    ARRAY['sedan'],
    25000,
    40000,
    '{"business_use": true}',
    30000,
    CURRENT_DATE + INTERVAL '300 days',
    20.0,
    'Dormant business prospect with minimal engagement',
    'Very low engagement despite business status suggests lack of current vehicle needs',
    '3bfe59a3-e22b-4401-9161-6090c788326e' -- James Anderson (sales_agent)
);

-- Now populate customer_warmth_history with historical progression data
-- This shows how warmth scores have changed over time

INSERT INTO customer_warmth_history (
    customer_warmth_id,
    customer_id,
    warmth_score_snapshot,
    warmth_level_snapshot,
    purchase_probability_snapshot,
    change_trigger,
    change_description,
    score_change,
    recorded_by
) VALUES 
-- Jennifer Lee's progression (hot prospect)
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'e623c347-d88e-4a0b-99ca-4f9f6ad7882e'),
    'e623c347-d88e-4a0b-99ca-4f9f6ad7882e',
    60.0,
    'lukewarm',
    45.0,
    'initial_inquiry',
    'Customer first inquiry about sedan options',
    0.0, -- baseline
    'cebe246f-7650-4405-ab06-7714c2c8c25f'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'e623c347-d88e-4a0b-99ca-4f9f6ad7882e'),
    'e623c347-d88e-4a0b-99ca-4f9f6ad7882e',
    72.5,
    'warm',
    58.0,
    'follow_up_call',
    'Positive response to follow-up call, expressed serious interest',
    +12.5,
    'cebe246f-7650-4405-ab06-7714c2c8c25f'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'e623c347-d88e-4a0b-99ca-4f9f6ad7882e'),
    'e623c347-d88e-4a0b-99ca-4f9f6ad7882e',
    85.5,
    'hot',
    78.0,
    'financing_inquiry',
    'Customer initiated financing inquiry - strong purchase signal',
    +13.0,
    'cebe246f-7650-4405-ab06-7714c2c8c25f'
),

-- Robert Brown's progression (scorching prospect)
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'f390205f-9cf2-43fc-9338-3acd1bd90327'),
    'f390205f-9cf2-43fc-9338-3acd1bd90327',
    45.0,
    'lukewarm',
    35.0,
    'website_visit',
    'Multiple website visits and brochure downloads',
    0.0,
    'db8cdcda-3c96-4e35-8fe4-860f927b9395'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'f390205f-9cf2-43fc-9338-3acd1bd90327'),
    'f390205f-9cf2-43fc-9338-3acd1bd90327',
    65.8,
    'warm',
    58.0,
    'sales_call',
    'Engaged sales call with specific vehicle preferences discussed',
    +20.8,
    'db8cdcda-3c96-4e35-8fe4-860f927b9395'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'f390205f-9cf2-43fc-9338-3acd1bd90327'),
    'f390205f-9cf2-43fc-9338-3acd1bd90327',
    88.2,
    'scorching',
    85.0,
    'test_drive',
    'Completed test drive and discussed financing options same day',
    +22.4,
    'db8cdcda-3c96-4e35-8fe4-860f927b9395'
),

-- TechCorp Solutions progression (B2B warm)
(
    (SELECT id FROM customer_warmth WHERE customer_id = '7287db95-f307-404e-aad9-2b619e3681f1'),
    '7287db95-f307-404e-aad9-2b619e3681f1',
    55.0,
    'lukewarm',
    40.0,
    'business_inquiry',
    'Initial fleet vehicle inquiry for company expansion',
    0.0,
    'caf0ebba-1e05-4a5a-b7a9-7874b0765e4b'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = '7287db95-f307-404e-aad9-2b619e3681f1'),
    '7287db95-f307-404e-aad9-2b619e3681f1',
    72.8,
    'warm',
    68.0,
    'proposal_review',
    'Reviewed detailed proposal and requested modifications',
    +17.8,
    'caf0ebba-1e05-4a5a-b7a9-7874b0765e4b'
),

-- Bob Smith's declining progression (cooling prospect)
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'dcb2838f-a93c-4b12-94e3-909cd734013e'),
    'dcb2838f-a93c-4b12-94e3-909cd734013e',
    68.0,
    'warm',
    62.0,
    'initial_contact',
    'Strong initial interest and multiple inquiries',
    0.0,
    'e1682094-c232-4b3d-b1d8-de98de4427e5'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'dcb2838f-a93c-4b12-94e3-909cd734013e'),
    'dcb2838f-a93c-4b12-94e3-909cd734013e',
    55.2,
    'lukewarm',
    48.0,
    'delayed_response',
    'Customer took longer to respond to follow-ups',
    -12.8,
    'e1682094-c232-4b3d-b1d8-de98de4427e5'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'dcb2838f-a93c-4b12-94e3-909cd734013e'),
    'dcb2838f-a93c-4b12-94e3-909cd734013e',
    42.5,
    'cool',
    35.0,
    'minimal_engagement',
    'Limited response to recent outreach attempts',
    -12.7,
    'e1682094-c232-4b3d-b1d8-de98de4427e5'
),

-- Daniel Wilson's steady progression
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'ed545112-f687-42b8-95de-8891cf28e1f3'),
    'ed545112-f687-42b8-95de-8891cf28e1f3',
    58.0,
    'lukewarm',
    45.0,
    'price_inquiry',
    'Asked about pricing for specific models',
    0.0,
    'cb951bd6-083a-4463-89a2-6e91149e47af'
),
(
    (SELECT id FROM customer_warmth WHERE customer_id = 'ed545112-f687-42b8-95de-8891cf28e1f3'),
    'ed545112-f687-42b8-95de-8891cf28e1f3',
    70.5,
    'warm',
    65.0,
    'callback_request',
    'Proactively requested callback to discuss options',
    +12.5,
    'cb951bd6-083a-4463-89a2-6e91149e47af'
);

-- Add some comments for documentation
COMMENT ON TABLE customer_warmth IS 'Tracks current customer warmth scores and engagement metrics. Warmth is measured at customer level, separate from opportunity stages.';

COMMENT ON TABLE customer_warmth_history IS 'Historical tracking of customer warmth changes over time, showing progression and triggers for score changes.'; 