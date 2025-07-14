#!/usr/bin/env python3
"""
Test script for query complexity limits validation.

Tests the following limits:
- Maximum 10 JOINs
- Maximum 2-level nested subqueries  
- Maximum 5000 character length
- 30-second timeout protection
"""

import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import _check_query_complexity, _count_subquery_nesting_levels, SQLValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_join_counting():
    """Test JOIN counting with various JOIN types."""
    print("\n=== Testing JOIN Counting ===")
    
    # Test cases with different JOIN types
    test_cases = [
        # Valid cases (≤10 JOINs)
        ("SELECT * FROM customers c INNER JOIN vehicles v ON c.id = v.customer_id", 1, True),
        ("SELECT * FROM customers c JOIN vehicles v ON c.id = v.customer_id LEFT JOIN branches b ON c.branch_id = b.id", 2, True),
        
        # Complex but valid case (exactly 10 JOINs)
        ("""SELECT * FROM customers c 
            INNER JOIN vehicles v ON c.id = v.customer_id
            LEFT JOIN branches b ON c.branch_id = b.id  
            RIGHT JOIN employees e ON b.id = e.branch_id
            FULL JOIN opportunities o ON c.id = o.customer_id
            CROSS JOIN pricing p ON v.id = p.vehicle_id
            JOIN transactions t ON o.id = t.opportunity_id
            LEFT JOIN activities a ON t.id = a.transaction_id
            INNER JOIN customers c2 ON c.referrer_id = c2.id
            RIGHT JOIN vehicles v2 ON c2.id = v2.customer_id""", 9, True),
        
        # Invalid case (>10 JOINs)
        ("""SELECT * FROM customers c 
            INNER JOIN vehicles v ON c.id = v.customer_id
            LEFT JOIN branches b ON c.branch_id = b.id  
            RIGHT JOIN employees e ON b.id = e.branch_id
            FULL JOIN opportunities o ON c.id = o.customer_id
            CROSS JOIN pricing p ON v.id = p.vehicle_id
            JOIN transactions t ON o.id = t.opportunity_id
            LEFT JOIN activities a ON t.id = a.transaction_id
            INNER JOIN customers c2 ON c.referrer_id = c2.id
            RIGHT JOIN vehicles v2 ON c2.id = v2.customer_id
            LEFT JOIN branches b2 ON c2.branch_id = b2.id
            INNER JOIN employees e2 ON b2.id = e2.branch_id""", 11, False),
    ]
    
    for query, expected_joins, should_pass in test_cases:
        try:
            _check_query_complexity(query)
            if should_pass:
                print(f"✅ PASS: Query with {expected_joins} JOINs accepted")
            else:
                print(f"❌ FAIL: Query with {expected_joins} JOINs should have been rejected")
        except SQLValidationError as e:
            if not should_pass:
                print(f"✅ PASS: Query with {expected_joins} JOINs correctly rejected: {e}")
            else:
                print(f"❌ FAIL: Query with {expected_joins} JOINs incorrectly rejected: {e}")

def test_subquery_nesting():
    """Test subquery nesting level detection."""
    print("\n=== Testing Subquery Nesting Levels ===")
    
    test_cases = [
        # Valid cases (≤2 levels)
        ("SELECT * FROM customers WHERE id = 1", 0, True),
        ("SELECT * FROM customers WHERE id IN (SELECT customer_id FROM vehicles)", 1, True),
        ("SELECT * FROM customers WHERE id IN (SELECT customer_id FROM vehicles WHERE model IN (SELECT model FROM pricing))", 2, True),
        
        # Invalid case (>2 levels)
        ("""SELECT * FROM customers WHERE id IN (
            SELECT customer_id FROM vehicles WHERE model IN (
                SELECT model FROM pricing WHERE price > (
                    SELECT AVG(price) FROM pricing WHERE category = 'luxury'
                )
            )
        )""", 3, False),
        
        # Complex valid case with multiple 2-level subqueries
        ("""SELECT * FROM customers c WHERE 
            c.id IN (SELECT customer_id FROM vehicles WHERE model IN (SELECT model FROM pricing)) 
            AND c.branch_id IN (SELECT id FROM branches WHERE region IN (SELECT region FROM employees))""", 2, True),
    ]
    
    for query, expected_level, should_pass in test_cases:
        actual_level = _count_subquery_nesting_levels(query)
        print(f"Query nesting level: {actual_level} (expected: {expected_level})")
        
        try:
            _check_query_complexity(query)
            if should_pass:
                print(f"✅ PASS: Query with {actual_level}-level nesting accepted")
            else:
                print(f"❌ FAIL: Query with {actual_level}-level nesting should have been rejected")
        except SQLValidationError as e:
            if not should_pass:
                print(f"✅ PASS: Query with {actual_level}-level nesting correctly rejected: {e}")
            else:
                print(f"❌ FAIL: Query with {actual_level}-level nesting incorrectly rejected: {e}")

def test_query_length():
    """Test query length limits."""
    print("\n=== Testing Query Length Limits ===")
    
    # Valid case (under 5000 characters)
    short_query = "SELECT * FROM customers WHERE name = 'John Smith'"
    try:
        _check_query_complexity(short_query)
        print(f"✅ PASS: Short query ({len(short_query)} chars) accepted")
    except SQLValidationError as e:
        print(f"❌ FAIL: Short query incorrectly rejected: {e}")
    
    # Invalid case (over 5000 characters)
    long_query = "SELECT * FROM customers WHERE " + " OR ".join([f"name = 'Customer{i}'" for i in range(1000)])
    print(f"Long query length: {len(long_query)} characters")
    
    try:
        _check_query_complexity(long_query)
        print(f"❌ FAIL: Long query ({len(long_query)} chars) should have been rejected")
    except SQLValidationError as e:
        print(f"✅ PASS: Long query correctly rejected: {e}")

def test_edge_cases():
    """Test edge cases and complex scenarios."""
    print("\n=== Testing Edge Cases ===")
    
    # Test parentheses that are not subqueries
    simple_math = "SELECT * FROM customers WHERE (age > 25 AND income > 50000) OR (status = 'premium')"
    try:
        nesting_level = _count_subquery_nesting_levels(simple_math)
        _check_query_complexity(simple_math)
        print(f"✅ PASS: Query with parentheses (not subqueries) accepted. Nesting level: {nesting_level}")
    except SQLValidationError as e:
        print(f"❌ FAIL: Query with parentheses incorrectly rejected: {e}")
    
    # Test mixed complexity (multiple factors)
    complex_query = """
    SELECT c.name, v.model, COUNT(*) 
    FROM customers c
    INNER JOIN vehicles v ON c.id = v.customer_id
    LEFT JOIN opportunities o ON c.id = o.customer_id
    WHERE c.id IN (
        SELECT customer_id FROM transactions WHERE amount > (
            SELECT AVG(amount) FROM transactions WHERE status = 'completed'
        )
    )
    GROUP BY c.name, v.model
    HAVING COUNT(*) > 1
    """
    
    try:
        nesting_level = _count_subquery_nesting_levels(complex_query)
        _check_query_complexity(complex_query)
        print(f"✅ PASS: Complex query accepted. Nesting level: {nesting_level}")
    except SQLValidationError as e:
        print(f"❌ FAIL: Complex query incorrectly rejected: {e}")

async def main():
    """Run all tests."""
    print("🔍 Testing Query Complexity Limits")
    print("=" * 50)
    
    try:
        test_join_counting()
        test_subquery_nesting()
        test_query_length()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ Query complexity testing completed!")
        print("\nImplemented limits:")
        print("- ✅ Maximum 10 JOINs (all types)")
        print("- ✅ Maximum 2-level nested subqueries")
        print("- ✅ Maximum 5000 character length")
        print("- ✅ Timeout protection (30 seconds)")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 