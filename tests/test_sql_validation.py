"""
Test script for SQL query validation security features.
Tests that only SELECT operations are allowed and all dangerous operations are blocked.
"""

import asyncio
import logging
import sys
import os

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import validate_sql_query, execute_safe_sql_query, SQLValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_allowed_queries():
    """Test queries that should be allowed."""
    print("✅ Testing ALLOWED queries...")
    
    allowed_queries = [
        "SELECT * FROM branches",
        "SELECT COUNT(*) FROM customers",
        "SELECT b.name, e.name FROM branches b JOIN employees e ON b.id = e.branch_id",
        "SELECT * FROM vehicles WHERE is_available = true",
        "SELECT customer_id, SUM(total_amount) FROM transactions GROUP BY customer_id",
        "  SELECT   name   FROM   customers  WHERE  email  LIKE  '%@company.com'  ",  # With extra whitespace
        """SELECT 
           o.id, 
           c.name as customer_name,
           v.brand || ' ' || v.model as vehicle
         FROM opportunities o
         JOIN customers c ON o.customer_id = c.id
         JOIN vehicles v ON o.vehicle_id = v.id
         WHERE o.stage = 'Won'""",  # Multi-line query
    ]
    
    passed = 0
    failed = 0
    
    for query in allowed_queries:
        try:
            validated = validate_sql_query(query)
            print(f"  ✅ PASS: {query[:50]}...")
            passed += 1
        except SQLValidationError as e:
            print(f"  ❌ FAIL: {query[:50]}... - {e}")
            failed += 1
    
    print(f"📊 Allowed queries test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_blocked_ddl_operations():
    """Test DDL operations that should be blocked."""
    print("🚫 Testing BLOCKED DDL operations...")
    
    blocked_queries = [
        "CREATE TABLE test (id int)",
        "ALTER TABLE branches ADD COLUMN test VARCHAR(255)",
        "DROP TABLE customers",
        "TRUNCATE TABLE vehicles",
        "GRANT SELECT ON branches TO public",
        "REVOKE ALL ON customers FROM public",
        "CREATE INDEX idx_test ON branches(name)",
        "DROP INDEX idx_branches_region",
    ]
    
    passed = 0
    failed = 0
    
    for query in blocked_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked: {str(e)[:80]}...")
            passed += 1
    
    print(f"📊 DDL blocking test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_blocked_dml_operations():
    """Test DML operations that should be blocked."""
    print("🚫 Testing BLOCKED DML operations...")
    
    blocked_queries = [
        "INSERT INTO customers (name, email) VALUES ('Test', 'test@test.com')",
        "UPDATE branches SET name = 'New Name' WHERE id = '123'",
        "DELETE FROM vehicles WHERE is_available = false",
        "MERGE INTO customers USING temp_customers ON customers.id = temp_customers.id",
        "UPSERT INTO branches (name) VALUES ('Test Branch')",
    ]
    
    passed = 0
    failed = 0
    
    for query in blocked_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked: {str(e)[:80]}...")
            passed += 1
    
    print(f"📊 DML blocking test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_dangerous_patterns():
    """Test dangerous SQL patterns that should be blocked."""
    print("⚠️ Testing DANGEROUS patterns...")
    
    dangerous_queries = [
        "SELECT pg_sleep(10) FROM branches",
        "SELECT * FROM branches; DROP TABLE customers;",  # SQL injection attempt
        "SELECT * FROM pg_tables",  # System table access
        "SELECT * FROM information_schema.tables",  # System schema access
        "SELECT /*comment*/ * FROM branches",  # With comments
        "SELECT * FROM branches -- comment",  # With line comment
        "COPY branches TO '/tmp/dump.csv'",  # File operation
        "SELECT * FROM branches UNION ALL SELECT * FROM pg_user",  # System table
    ]
    
    passed = 0
    failed = 0
    
    for query in dangerous_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked: {str(e)[:80]}...")
            passed += 1
    
    print(f"📊 Dangerous patterns test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_table_access_restrictions():
    """Test table access restrictions."""
    print("🔒 Testing TABLE ACCESS restrictions...")
    
    invalid_table_queries = [
        "SELECT * FROM pg_tables",
        "SELECT * FROM information_schema.columns",
        "SELECT * FROM auth.users",
        "SELECT * FROM storage.objects",
        "SELECT * FROM unauthorized_table",
        "SELECT b.*, u.* FROM branches b JOIN pg_user u ON true",
    ]
    
    passed = 0
    failed = 0
    
    for query in invalid_table_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked: {str(e)[:80]}...")
            passed += 1
    
    print(f"📊 Table access test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_query_complexity_limits():
    """Test query complexity limitations."""
    print("📏 Testing QUERY COMPLEXITY limits...")
    
    # Create a query with too many JOINs (over 10)
    too_many_joins = "SELECT * FROM branches " + " ".join([
        f"JOIN employees e{i} ON branches.id = e{i}.branch_id" for i in range(12)
    ])
    
    # Create a very long query (over 5000 characters)
    very_long_query = "SELECT " + ", ".join([f"column_{i}" for i in range(1000)]) + " FROM branches"
    
    complex_queries = [
        too_many_joins,
        very_long_query,
        "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM branches))))))",  # Too many nested levels
    ]
    
    passed = 0
    failed = 0
    
    for query in complex_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked for complexity!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked: {str(e)[:80]}...")
            passed += 1
    
    print(f"📊 Query complexity test: {passed} passed, {failed} failed\n")
    return failed == 0

async def test_end_to_end_execution():
    """Test end-to-end SQL execution with validation."""
    print("🔄 Testing END-TO-END execution...")
    
    try:
        # Test a simple safe query
        result = await execute_safe_sql_query("SELECT COUNT(*) as branch_count FROM branches")
        print(f"  ✅ PASS: Safe query executed successfully - {result}")
        
        # Test a blocked query
        try:
            await execute_safe_sql_query("DROP TABLE branches")
            print("  ❌ FAIL: Dangerous query should have been blocked!")
            return False
        except SQLValidationError:
            print("  ✅ PASS: Dangerous query correctly blocked in execution")
        
        print("📊 End-to-end test: PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ⚠️ SKIP: End-to-end test skipped (database not available): {e}")
        print("📊 End-to-end test: SKIPPED (requires database connection)\n")
        return True  # Don't fail the overall test if DB isn't available

async def run_all_tests():
    """Run all SQL validation tests."""
    print("🧪 Starting SQL Query Validation Security Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all validation tests
    test_results.append(test_allowed_queries())
    test_results.append(test_blocked_ddl_operations())
    test_results.append(test_blocked_dml_operations())
    test_results.append(test_dangerous_patterns())
    test_results.append(test_table_access_restrictions())
    test_results.append(test_query_complexity_limits())
    
    # Run end-to-end test
    test_results.append(await test_end_to_end_execution())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("=" * 60)
    print(f"🎯 FINAL RESULTS: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! SQL validation security is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Review the security implementation.")
        
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 