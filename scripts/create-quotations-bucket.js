#!/usr/bin/env node

/**
 * Create Quotations Storage Bucket
 * Uses Supabase Management API with service key to create the quotations bucket
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment variables');
    process.exit(1);
}

// Extract project reference from URL
const projectRef = SUPABASE_URL.replace('https://', '').split('.')[0];

console.log('🚀 Creating quotations storage bucket...');
console.log(`📍 Project: ${projectRef}`);

// Create bucket configuration
const bucketConfig = {
    id: 'quotations',
    name: 'quotations',
    public: false,
    file_size_limit: 10485760, // 10MB
    allowed_mime_types: ['application/pdf']
};

// Prepare the HTTP request
const postData = JSON.stringify(bucketConfig);

const options = {
    hostname: `${projectRef}.supabase.co`,
    port: 443,
    path: '/storage/v1/bucket',
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${SUPABASE_SERVICE_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
        'apikey': SUPABASE_SERVICE_KEY
    }
};

const req = https.request(options, (res) => {
    let data = '';

    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        console.log(`📊 Response Status: ${res.statusCode}`);
        
        if (res.statusCode === 200 || res.statusCode === 201) {
            console.log('✅ Quotations bucket created successfully!');
            console.log('📝 Response:', JSON.parse(data));
        } else if (res.statusCode === 409) {
            console.log('ℹ️  Quotations bucket already exists');
            console.log('📝 Response:', data);
        } else {
            console.log('❌ Failed to create bucket');
            console.log('📝 Response:', data);
            process.exit(1);
        }
    });
});

req.on('error', (e) => {
    console.error('❌ Request failed:', e.message);
    process.exit(1);
});

// Send the request
req.write(postData);
req.end();