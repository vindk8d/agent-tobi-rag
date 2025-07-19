#!/usr/bin/env node

/**
 * Database Connection Test Script for Memory Debug Interface
 * 
 * This script tests:
 * 1. Supabase connection from frontend environment
 * 2. Checks for existing users/customers data
 * 3. Validates memory tables exist
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '../frontend/.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase environment variables!');
  console.log('Create frontend/.env.local with:');
  console.log('NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co');
  console.log('NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function testConnection() {
  console.log('🔍 Testing Supabase Connection...\n');

  try {
    // Test 1: Basic connection
    console.log('1. Testing basic connection...');
    const { data, error } = await supabase.from('customers').select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('❌ Connection failed:', error.message);
      return false;
    }
    console.log('✅ Connection successful!\n');

    // Test 2: Check for users table
    console.log('2. Checking for users table...');
    const { data: usersData, error: usersError } = await supabase
      .from('users')
      .select('id, email, display_name, user_type, created_at')
      .limit(5);

    if (usersError) {
      if (usersError.message.includes('relation "users" does not exist')) {
        console.log('⚠️  Users table does not exist, will use customers table');
      } else {
        console.error('❌ Users table error:', usersError.message);
      }
    } else {
      console.log(`✅ Found ${usersData?.length || 0} users in users table`);
      if (usersData?.length > 0) {
        console.log('   Sample users:', usersData.slice(0, 2).map(u => `${u.email} (${u.user_type || 'user'})`));
      }
    }

    // Test 3: Check customers table (fallback)
    console.log('\n3. Checking customers table...');
    const { data: customersData, error: customersError } = await supabase
      .from('customers')
      .select('id, name, email, created_at')
      .limit(5);

    if (customersError) {
      console.error('❌ Customers table error:', customersError.message);
    } else {
      console.log(`✅ Found ${customersData?.length || 0} customers in customers table`);
      if (customersData?.length > 0) {
        console.log('   Sample customers:', 
          customersData.slice(0, 2).map(c => `${c.name} (${c.email || 'no email'})`));
      }
    }

    // Test 4: Check memory tables
    console.log('\n4. Checking memory system tables...');
    
    const tables = [
      'long_term_memories',
      'conversation_summaries',
      'messages',
      'conversations'
    ];

    for (const table of tables) {
      const { data, error } = await supabase.from(table).select('count', { count: 'exact', head: true });
      if (error) {
        console.log(`❌ Table '${table}': ${error.message}`);
      } else {
        console.log(`✅ Table '${table}': ${data?.length || 0} records`);
      }
    }

    // Summary
    console.log('\n📊 Summary:');
    const totalUsers = (usersData?.length || 0) + (customersData?.length || 0);
    if (totalUsers === 0) {
      console.log('⚠️  No users found in database. You need to:');
      console.log('   1. Run the CRM migration with sample data');
      console.log('   2. Or add test users manually');
    } else {
      console.log(`✅ Found ${totalUsers} total users available for memory debugging`);
    }

    return true;

  } catch (error) {
    console.error('❌ Unexpected error:', error.message);
    return false;
  }
}

// Run the test
testConnection().then(success => {
  if (success) {
    console.log('\n🎉 Database connection test completed!');
    console.log('You can now use the Memory Debug Interface at: http://localhost:3000/memorycheck');
  } else {
    console.log('\n💡 Fix the issues above and try again.');
  }
}); 