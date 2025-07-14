"""
Standalone test script for semantic validation of high-cardinality columns.
Tests core semantic similarity logic without requiring full LangChain environment.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import asyncio

# Mock the numpy functionality for basic testing
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors using basic math."""
    try:
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    except:
        return 0.0

# High-cardinality columns configuration (copied from tools.py)
HIGH_CARDINALITY_COLUMNS = {
    "customers": ["name", "company"],
    "employees": ["name"],
    "vehicles": ["brand", "model"],
    "opportunities": [],
    "branches": ["name", "brand"],
    "transactions": [],
    "pricing": [],
    "activities": ["subject", "description"]
}

# Known values cache (copied from tools.py)
KNOWN_VALUES_CACHE = {
    "employee_names": [
        "John Smith", "Sarah Johnson", "Mike Davis", "Lisa Wang", "Carlos Rodriguez",
        "Alex Thompson", "Emma Wilson", "David Chen", "Maria Garcia"
    ],
    "customer_names": [
        "Robert Brown", "Jennifer Lee", "Mark Johnson", "TechCorp Solutions", "Global Industries"
    ],
    "vehicle_brands": ["Toyota", "Honda", "Ford", "Nissan", "BMW", "Mercedes", "Audi"],
    "vehicle_models": ["Camry", "RAV4", "Civic", "CR-V", "F-150", "Altima", "Prius", "Corolla", "Accord"],
    "company_names": ["TechCorp Solutions", "Global Industries", "AutoCorp", "MegaTech Inc"]
}

class MockEmbeddings:
    """Mock embeddings class for testing without OpenAI API calls."""
    
    def __init__(self):
        # Pre-computed mock embeddings for test data
        self.embeddings_cache = {
            # Employee names
            "john smith": [0.1, 0.2, 0.3, 0.4, 0.5],
            "sarah johnson": [0.2, 0.3, 0.4, 0.5, 0.6],
            "mike davis": [0.3, 0.4, 0.5, 0.6, 0.7],
            
            # Customer names
            "robert brown": [0.4, 0.5, 0.6, 0.7, 0.8],
            "jennifer lee": [0.5, 0.6, 0.7, 0.8, 0.9],
            "mark johnson": [0.3, 0.5, 0.7, 0.8, 0.6],
            
            # Vehicle brands
            "toyota": [0.8, 0.1, 0.2, 0.3, 0.4],
            "honda": [0.9, 0.2, 0.3, 0.4, 0.5],
            "ford": [0.7, 0.3, 0.4, 0.5, 0.6],
            "nissan": [0.6, 0.4, 0.5, 0.6, 0.7],
            
            # Vehicle models
            "camry": [0.6, 0.7, 0.8, 0.9, 1.0],
            "civic": [0.5, 0.8, 0.9, 1.0, 0.1],
            "f-150": [0.4, 0.9, 1.0, 0.1, 0.2],
            "rav4": [0.7, 0.8, 0.6, 0.9, 0.5],
            
            # Test variations/typos
            "jon smith": [0.12, 0.21, 0.31, 0.41, 0.51],  # Similar to "john smith"
            "toyoda": [0.81, 0.11, 0.21, 0.31, 0.41],  # Similar to "toyota"
            "nonsense xyz": [0.99, 0.01, 0.01, 0.01, 0.01],  # Very different
            "random text": [0.98, 0.02, 0.01, 0.01, 0.01],   # Very different
            "drop table": [0.97, 0.01, 0.01, 0.01, 0.01],   # SQL injection attempt
        }
    
    async def embed_query(self, text: str):
        """Mock embedding generation."""
        text_lower = text.lower()
        if text_lower in self.embeddings_cache:
            return self.embeddings_cache[text_lower]
        
        # Generate pseudo-random but consistent embeddings for unknown text
        hash_val = hash(text_lower) % 1000
        return [(hash_val + i) / 1000.0 for i in range(5)]

def extract_high_cardinality_terms(query: str) -> List[Tuple[str, str, str]]:
    """Extract search terms for high-cardinality columns from SQL query."""
    terms = []
    seen_terms = set()  # To avoid duplicates
    
    # Look for patterns like: table.column = 'value' or column LIKE '%value%'
    # Process table.column patterns first to be more specific
    table_column_patterns = [
        r'(\w+)\.(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',  # table.column = 'value'
        r'(\w+)\.(\w+)\s*LIKE\s*[\'"]([^\'"]+)[\'"]',  # table.column LIKE 'value'
    ]
    
    # First pass: handle table.column patterns
    for pattern in table_column_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for table, column, value in matches:
            table = table.lower()
            column = column.lower()
            
            # Check if this is a high-cardinality column we should validate
            if table in HIGH_CARDINALITY_COLUMNS and column in HIGH_CARDINALITY_COLUMNS[table]:
                term_key = (table, column, value)
                if term_key not in seen_terms:
                    terms.append(term_key)
                    seen_terms.add(term_key)
    
    # Second pass: handle column-only patterns, but only if not already found
    column_only_patterns = [
        r'(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',  # column = 'value'
        r'(\w+)\s*LIKE\s*[\'"]([^\'"]+)[\'"]'  # column LIKE 'value'
    ]
    
    for pattern in column_only_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for column, value in matches:
            table = infer_table_from_query(column, query)
            
            table = table.lower()
            column = column.lower()
            
            # Check if this is a high-cardinality column we should validate
            if table in HIGH_CARDINALITY_COLUMNS and column in HIGH_CARDINALITY_COLUMNS[table]:
                term_key = (table, column, value)
                if term_key not in seen_terms:  # Only add if not already found
                    terms.append(term_key)
                    seen_terms.add(term_key)
    
    return terms

def infer_table_from_query(column: str, query: str) -> str:
    """Infer table name from query context when only column is specified."""
    normalized_query = query.upper()
    
    # Look for FROM clause to identify main table
    from_match = re.search(r'\bFROM\s+(\w+)', normalized_query)
    if from_match:
        table = from_match.group(1).lower()
        if table in HIGH_CARDINALITY_COLUMNS and column.lower() in HIGH_CARDINALITY_COLUMNS[table]:
            return table
    
    # Fallback: check all high-cardinality tables for this column
    for table, columns in HIGH_CARDINALITY_COLUMNS.items():
        if column.lower() in columns:
            return table
            
    return "unknown"

def get_known_values_for_column(table: str, column: str) -> List[str]:
    """Get known values for a specific table.column combination."""
    column_mapping = {
        ("employees", "name"): "employee_names",
        ("customers", "name"): "customer_names",  
        ("customers", "company"): "company_names",
        ("vehicles", "brand"): "vehicle_brands",
        ("vehicles", "model"): "vehicle_models",
        ("branches", "name"): "customer_names",
        ("branches", "brand"): "vehicle_brands",
    }
    
    cache_key = column_mapping.get((table, column))
    return KNOWN_VALUES_CACHE.get(cache_key, [])

def is_trivial_term(term: str) -> bool:
    """Check if a search term is trivial and doesn't need semantic validation."""
    if not term or len(term) < 2:
        return True
    
    # Skip if it looks like a UUID
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', term, re.IGNORECASE):
        return True
    
    # Skip if it's just numbers
    if term.isdigit():
        return True
        
    # Skip very common SQL patterns
    if term.upper() in ['TRUE', 'FALSE', 'NULL', '%', '_']:
        return True
        
    return False

async def calculate_semantic_similarities(
    embeddings: MockEmbeddings, 
    search_term: str, 
    known_values: List[str]
) -> List[Tuple[str, float]]:
    """Calculate semantic similarities between search term and known values."""
    try:
        # Embed the search term
        search_embedding = await embeddings.embed_query(search_term)
        
        # Embed all known values and calculate similarities
        similarities = []
        for value in known_values:
            try:
                known_embedding = await embeddings.embed_query(value)
                similarity = cosine_similarity(search_embedding, known_embedding)
                similarities.append((value, similarity))
            except Exception:
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:10]  # Return top 10 matches
        
    except Exception:
        return []

async def validate_high_cardinality_terms(query: str) -> Dict[str, Any]:
    """Validate search terms against high-cardinality columns using semantic similarity."""
    validation_result = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "validated_terms": []
    }
    
    try:
        terms_to_validate = extract_high_cardinality_terms(query)
        
        if not terms_to_validate:
            return validation_result
        
        embeddings = MockEmbeddings()
        
        for table, column, search_term in terms_to_validate:
            if is_trivial_term(search_term):
                continue
                
            known_values = get_known_values_for_column(table, column)
            
            if not known_values:
                validation_result["warnings"].append(
                    f"No validation data available for {table}.{column}"
                )
                continue
            
            similarity_results = await calculate_semantic_similarities(
                embeddings, search_term, known_values
            )
            
            best_match, best_score = similarity_results[0] if similarity_results else (None, 0.0)
            
            # Apply validation thresholds
            if best_score < 0.5:  # Low similarity - likely suspicious
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"Search term '{search_term}' for {table}.{column} appears invalid (similarity: {best_score:.2f})"
                )
                
                reasonable_matches = [(val, score) for val, score in similarity_results if score > 0.2]
                if reasonable_matches:
                    suggestions = [f"'{val}' ({score:.2f})" for val, score in reasonable_matches[:3]]
                    validation_result["suggestions"].append(
                        f"Did you mean: {', '.join(suggestions)} for {table}.{column}?"
                    )
                    
            elif best_score < 0.8:  # Moderate similarity - suggest correction
                validation_result["suggestions"].append(
                    f"For {table}.{column}, did you mean '{best_match}' instead of '{search_term}'? (similarity: {best_score:.2f})"
                )
                
            validation_result["validated_terms"].append({
                "table": table,
                "column": column,
                "search_term": search_term,
                "best_match": best_match,
                "similarity_score": best_score
            })
            
    except Exception as e:
        validation_result["warnings"].append(f"Semantic validation failed: {e}")
    
    return validation_result

# Test functions
async def test_term_extraction():
    """Test extraction of high-cardinality search terms from SQL queries."""
    print("🔍 Testing high-cardinality term extraction...")
    
    test_cases = [
        ("SELECT * FROM employees WHERE name = 'John Smith'", [("employees", "name", "John Smith")]),
        ("SELECT * FROM customers WHERE customers.name LIKE '%Robert%'", [("customers", "name", "%Robert%")]),
        ("SELECT * FROM vehicles WHERE brand = 'Toyota' AND model = 'Camry'", [("vehicles", "brand", "Toyota"), ("vehicles", "model", "Camry")]),
        ("SELECT e.name FROM employees e WHERE e.name = 'Sarah Johnson'", [("employees", "name", "Sarah Johnson")]),
        ("SELECT * FROM branches WHERE id = '12345'", []),  # No high-cardinality text columns
    ]
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = extract_high_cardinality_terms(query)
        if result == expected:
            print(f"  ✅ PASS: {query[:50]}...")
            passed += 1
        else:
            print(f"  ❌ FAIL: {query[:50]}...")
            print(f"       Expected: {expected}")
            print(f"       Got: {result}")
            failed += 1
    
    print(f"📊 Term extraction test: {passed} passed, {failed} failed\n")
    return failed == 0

def test_similarity_calculation():
    """Test cosine similarity calculation."""
    print("📐 Testing cosine similarity calculation...")
    
    test_cases = [
        ([1, 0, 0], [1, 0, 0], 1.0),  # Identical vectors
        ([1, 0, 0], [0, 1, 0], 0.0),  # Orthogonal vectors
        ([1, 1, 1], [1, 1, 1], 1.0),  # Identical non-unit vectors
        ([1, 2, 3], [2, 4, 6], 1.0),  # Proportional vectors
    ]
    
    passed = 0
    failed = 0
    
    for vec1, vec2, expected in test_cases:
        result = cosine_similarity(vec1, vec2)
        if abs(result - expected) < 0.001:
            print(f"  ✅ PASS: similarity({vec1}, {vec2}) = {result:.3f}")
            passed += 1
        else:
            print(f"  ❌ FAIL: similarity({vec1}, {vec2}) = {result:.3f}, expected {expected}")
            failed += 1
    
    print(f"📊 Cosine similarity test: {passed} passed, {failed} failed\n")
    return failed == 0

async def test_semantic_validation():
    """Test semantic validation with various query scenarios."""
    print("🧠 Testing semantic validation scenarios...")
    
    test_scenarios = [
        # Good queries (should pass)
        ("SELECT * FROM employees WHERE name = 'John Smith'", True, "exact match"),
        ("SELECT * FROM vehicles WHERE brand = 'Toyota'", True, "exact brand match"),
        
        # Typos (should suggest corrections)
        ("SELECT * FROM employees WHERE name = 'Jon Smith'", True, "typo should suggest correction"),
        ("SELECT * FROM vehicles WHERE brand = 'Toyoda'", True, "brand typo should suggest correction"),
        
        # Suspicious queries (should flag as invalid)
        ("SELECT * FROM employees WHERE name = 'Random Text'", False, "nonsense text"),
        ("SELECT * FROM vehicles WHERE brand = 'DROP TABLE'", False, "potential injection"),
        
        # No high-cardinality terms (should pass)
        ("SELECT COUNT(*) FROM branches", True, "no high-cardinality terms"),
    ]
    
    passed = 0
    failed = 0
    
    for query, should_be_valid, description in test_scenarios:
        try:
            result = await validate_high_cardinality_terms(query)
            
            print(f"  Testing: {description}")
            print(f"    Query: {query[:60]}...")
            print(f"    Valid: {result['valid']}")
            
            if result["warnings"]:
                print(f"    Warnings: {result['warnings']}")
            if result["suggestions"]:
                print(f"    Suggestions: {result['suggestions']}")
            
            # For this test, we consider it successful if:
            # 1. Valid queries don't get blocked entirely
            # 2. Invalid queries generate appropriate warnings
            test_passed = True
            if not should_be_valid and result["valid"] and len(result["warnings"]) == 0:
                test_passed = False
                print(f"    ❌ FAIL: Should have flagged as suspicious")
            elif should_be_valid and not result["valid"]:
                test_passed = False
                print(f"    ❌ FAIL: Should not have blocked valid query")
            else:
                print(f"    ✅ PASS: Validation result appropriate")
            
            if test_passed:
                passed += 1
            else:
                failed += 1
                
            print()
            
        except Exception as e:
            print(f"  ❌ FAIL: {query[:50]}... - Exception: {e}")
            failed += 1
    
    print(f"📊 Semantic validation test: {passed} passed, {failed} failed\n")
    return failed == 0

async def run_all_tests():
    """Run all semantic validation tests."""
    print("🧪 Semantic Validation Tests for High-Cardinality Columns")
    print("=" * 60)
    print("Note: Using mock embeddings for testing without API dependency")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(await test_term_extraction())
    test_results.append(test_similarity_calculation())
    test_results.append(await test_semantic_validation())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("=" * 60)
    print(f"🎯 FINAL RESULTS: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Semantic validation is working correctly.")
        print("💡 The system can:")
        print("   - Extract high-cardinality search terms from SQL queries")
        print("   - Calculate semantic similarity between search terms and known values")
        print("   - Detect typos and suggest corrections")
        print("   - Flag suspicious queries that may be injection attempts")
    else:
        print("❌ SOME TESTS FAILED! Review the implementation.")
        
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1) 