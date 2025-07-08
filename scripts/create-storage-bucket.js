/**
 * Script to create the 'documents' storage bucket in Supabase
 * Run this script to set up the storage bucket for document uploads
 */

const { createClient } = require('@supabase/supabase-js');

// Load environment variables
require('dotenv').config();

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY; // Need service key for admin operations

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing required environment variables:');
  console.error('   SUPABASE_URL and SUPABASE_SERVICE_KEY must be set');
  console.error('   Add them to your .env file');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function createDocumentsBucket() {
  console.log('🚀 Creating documents storage bucket...');
  
  try {
    // Create the bucket
    const { data, error } = await supabase.storage.createBucket('documents', {
      public: false,
      allowedMimeTypes: [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/markdown',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.ms-powerpoint',
        'image/jpeg',
        'image/png',
        'image/webp'
      ],
      fileSizeLimit: 52428800 // 50MB
    });

    if (error) {
      if (error.message.includes('already exists')) {
        console.log('✅ Bucket "documents" already exists');
      } else {
        console.error('❌ Error creating bucket:', error.message);
        return;
      }
    } else {
      console.log('✅ Successfully created bucket "documents"');
    }

    // Verify the bucket exists
    const { data: buckets, error: listError } = await supabase.storage.listBuckets();
    
    if (listError) {
      console.error('❌ Error listing buckets:', listError.message);
      return;
    }

    const documentsBucket = buckets.find(bucket => bucket.name === 'documents');
    
    if (documentsBucket) {
      console.log('✅ Bucket verification successful');
      console.log('📁 Bucket details:', {
        name: documentsBucket.name,
        public: documentsBucket.public,
        file_size_limit: documentsBucket.file_size_limit,
        allowed_mime_types: documentsBucket.allowed_mime_types
      });
    } else {
      console.error('❌ Bucket not found in list');
    }

    console.log('\n🎉 Storage setup complete!');
    console.log('You can now upload documents through the frontend at localhost:3000/manage');

  } catch (error) {
    console.error('❌ Unexpected error:', error);
  }
}

// Run the function
createDocumentsBucket(); 