"""
Simple test script for SQL query validation functions.
Tests SQL validation security without requiring full LangChain environment.
"""

import re
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, DML

# Copy the validation logic here for testing without dependencies
ALLOWED_SQL_OPERATIONS = {"SELECT"}
BLOCKED_DDL_OPERATIONS = {
    "CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME", "COMMENT",
    "GRANT", "REVOKE", "ANALYZE", "VACUUM", "REINDEX"
}
BLOCKED_DML_OPERATIONS = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE"
}
BLOCKED_ADMIN_OPERATIONS = {
    "CALL", "EXECUTE", "EXEC", "DO", "COPY", "LOAD", "IMPORT", "EXPORT",
    "BACKUP", "RESTORE", "SET", "RESET", "SHOW", "DESCRIBE", "EXPLAIN"
}
BLOCKED_OPERATIONS = BLOCKED_DDL_OPERATIONS | BLOCKED_DML_OPERATIONS | BLOCKED_ADMIN_OPERATIONS

CRM_TABLES = [
    "branches", "employees", "customers", "vehicles",
    "opportunities", "transactions", "pricing", "activities"
]

class SQLValidationError(Exception):
    """Custom exception for SQL validation failures."""
    pass

def validate_sql_query(query: str) -> str:
    """Validate SQL query to ensure it only contains allowed operations."""
    if not query or not query.strip():
        raise SQLValidationError("Empty or whitespace-only query not allowed")
    
    cleaned_query = query.strip()
    
    # Use sqlparse to parse and validate the SQL
    try:
        parsed = sqlparse.parse(cleaned_query)
    except Exception as e:
        raise SQLValidationError(f"Invalid SQL syntax: {e}")
    
    if len(parsed) == 0:
        raise SQLValidationError("No valid SQL statements found")
    
    if len(parsed) > 1:
        raise SQLValidationError("Multiple statements not allowed. Only single SELECT queries permitted.")
    
    statement = parsed[0]
    
    # Extract the first meaningful token (should be SELECT)
    first_keyword = None
    for token in statement.flatten():
        if token.ttype is Keyword and token.value.upper() in (ALLOWED_SQL_OPERATIONS | BLOCKED_OPERATIONS):
            first_keyword = token.value.upper()
            break
    
    if not first_keyword:
        raise SQLValidationError("No valid SQL operation found in query")
    
    # Check if the operation is allowed
    if first_keyword not in ALLOWED_SQL_OPERATIONS:
        if first_keyword in BLOCKED_OPERATIONS:
            raise SQLValidationError(f"Operation '{first_keyword}' is not allowed. Only SELECT queries are permitted.")
        else:
            raise SQLValidationError(f"Unknown or invalid SQL operation: '{first_keyword}'. Only SELECT queries are permitted.")
    
    # Get normalized query for additional checks
    normalized_query = ' '.join(cleaned_query.split()).upper()
    
    # Additional security checks
    _check_dangerous_patterns(normalized_query)
    _validate_table_access(normalized_query)
    _validate_parsed_statement(statement)
    
    return cleaned_query

def _validate_parsed_statement(statement: Statement) -> None:
    """Additional validation using parsed SQL statement."""
    for token in statement.flatten():
        if token.ttype is Keyword and token.value.upper() in BLOCKED_OPERATIONS:
            raise SQLValidationError(f"Blocked operation '{token.value.upper()}' found in query")
        
        # Check for suspicious function calls
        if hasattr(token, 'value') and isinstance(token.value, str):
            token_upper = token.value.upper()
            dangerous_functions = ['PG_SLEEP', 'SLEEP', 'WAITFOR', 'DELAY']
            for func in dangerous_functions:
                if func in token_upper:
                    raise SQLValidationError(f"Dangerous function '{func}' detected in query")

def _check_dangerous_patterns(normalized_query: str) -> None:
    """Check for dangerous SQL patterns that should be blocked."""
    dangerous_patterns = [
        r'\bPG_SLEEP\b', r'\bSLEEP\b', r'\bWAIT\b',
        r'\bLO_IMPORT\b', r'\bLO_EXPORT\b', r'\bCOPY\b',
        r'\bPG_READ_FILE\b', r'\bPG_WRITE_FILE\b',
        r'\bALTER\s+SYSTEM\b', r'\bSET\s+ROLE\b',
        r'\bEXECUTE\s+IMMEDIATE\b', r'\bDYNAMIC\s+SQL\b',
        r'\$\$', r'\bSHELL\b', r'\bSYSTEM\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            raise SQLValidationError(f"Dangerous SQL pattern detected: {pattern}")

def _validate_table_access(normalized_query: str) -> None:
    """Validate that query only accesses allowed CRM tables."""
    table_patterns = [
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    referenced_tables = set()
    for pattern in table_patterns:
        matches = re.findall(pattern, normalized_query, re.IGNORECASE)
        referenced_tables.update([match.lower() for match in matches])
    
    # Check if all referenced tables are in the allowed CRM tables
    allowed_tables_lower = {table.lower() for table in CRM_TABLES}
    
    for table in referenced_tables:
        if table not in allowed_tables_lower:
            raise SQLValidationError(f"Access to table '{table}' is not allowed. Only CRM tables are accessible: {', '.join(CRM_TABLES)}")

# Test functions below...

def test_allowed_queries():
    """Test queries that should be allowed."""
    print("✅ Testing ALLOWED queries...")
    
    allowed_queries = [
        "SELECT * FROM branches",
        "SELECT COUNT(*) FROM customers",
        "SELECT b.name, e.name FROM branches b JOIN employees e ON b.id = e.branch_id",
        "SELECT * FROM vehicles WHERE is_available = true",
        "SELECT customer_id, SUM(total_amount) FROM transactions GROUP BY customer_id",
        "  SELECT   name   FROM   customers  WHERE  email  LIKE  '%@company.com'  ",
        """SELECT 
           o.id, 
           c.name as customer_name,
           v.brand || ' ' || v.model as vehicle
         FROM opportunities o
         JOIN customers c ON o.customer_id = c.id
         JOIN vehicles v ON o.vehicle_id = v.id
         WHERE o.stage = 'Won'""",
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

def test_blocked_operations():
    """Test operations that should be blocked."""
    print("🚫 Testing BLOCKED operations...")
    
    blocked_queries = [
        "CREATE TABLE test (id int)",
        "ALTER TABLE branches ADD COLUMN test VARCHAR(255)",
        "DROP TABLE customers",
        "INSERT INTO customers (name, email) VALUES ('Test', 'test@test.com')",
        "UPDATE branches SET name = 'New Name' WHERE id = '123'",
        "DELETE FROM vehicles WHERE is_available = false",
        "SELECT pg_sleep(10) FROM branches",
        "SELECT * FROM branches; DROP TABLE customers;",
        "SELECT * FROM pg_tables",
        "COPY branches TO '/tmp/dump.csv'",
    ]
    
    passed = 0
    failed = 0
    
    for query in blocked_queries:
        try:
            validate_sql_query(query)
            print(f"  ❌ FAIL: {query[:50]}... - Should have been blocked!")
            failed += 1
        except SQLValidationError as e:
            print(f"  ✅ PASS: {query[:50]}... - Correctly blocked")
            passed += 1
    
    print(f"📊 Blocked operations test: {passed} passed, {failed} failed\n")
    return failed == 0

def run_tests():
    """Run all validation tests."""
    print("🧪 SQL Query Validation Security Tests (Simple)")
    print("=" * 50)
    
    test_results = [
        test_allowed_queries(),
        test_blocked_operations()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("=" * 50)
    print(f"🎯 RESULTS: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! SQL validation is working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Review the implementation.")
        return False

if __name__ == "__main__":
    run_tests() 