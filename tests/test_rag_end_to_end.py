#!/usr/bin/env python3
"""
End-to-End RAG Testing Suite
Tests the complete RAG pipeline with Ford Bronco document questions
"""

import asyncio
import sys
import os
import json
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, '.')

# Test questions based on the Ford Bronco document
TEST_QUESTIONS = [
    # Basic factual questions
    {
        "id": "engine_specs",
        "question": "What engine does the Ford Bronco have and what is its power output?",
        "expected_keywords": ["2.7L", "EcoBoost", "V6", "335 PS", "555 Nm"],
        "category": "basic_facts"
    },
    {
        "id": "transmission",
        "question": "What type of transmission does the Ford Bronco use?",
        "expected_keywords": ["10-Speed", "Automatic"],
        "category": "basic_facts"
    },
    {
        "id": "dimensions",
        "question": "What are the dimensions and ground clearance of the Ford Bronco?",
        "expected_keywords": ["4,811", "2,190", "1,846", "292 mm", "ground clearance"],
        "category": "basic_facts"
    },
    
    # Feature-based questions
    {
        "id": "safety_features",
        "question": "What safety features does the Ford Bronco include?",
        "expected_keywords": ["Co-Pilot360", "Pre-Collision", "BLIS", "Lane-Keeping", "360-Degree Camera"],
        "category": "features"
    },
    {
        "id": "entertainment_system",
        "question": "Describe the entertainment system in the Ford Bronco.",
        "expected_keywords": ["SYNC", "12-inch", "Apple CarPlay", "Android Auto", "B&O"],
        "category": "features"
    },
    {
        "id": "off_road_capabilities",
        "question": "What off-road capabilities does the Ford Bronco have?",
        "expected_keywords": ["Sasquatch", "4x4", "G.O.A.T.", "Mud-Terrain", "850 mm", "wading"],
        "category": "features"
    },
    
    # Complex/comparative questions
    {
        "id": "drive_modes",
        "question": "What are all the available drive modes in the Ford Bronco?",
        "expected_keywords": ["Normal", "Eco", "Sport", "Slippery", "Mud/Ruts", "Sand", "G.O.A.T."],
        "category": "complex"
    },
    {
        "id": "warranty_service",
        "question": "What warranty and service options are available for the Ford Bronco?",
        "expected_keywords": ["5 years", "150,000 km", "Extended Warranty", "Roadside Assistance", "Express Service"],
        "category": "complex"
    },
    
    # Edge cases
    {
        "id": "color_options",
        "question": "What colors is the Ford Bronco available in?",
        "expected_keywords": ["Eruption Green", "Race Red", "Cactus Gray", "Oxford White", "Azure Gray", "Shadow Black"],
        "category": "edge_case"
    },
    {
        "id": "irrelevant_question",
        "question": "What is the price of Tesla Model 3?",
        "expected_keywords": ["Tesla Model 3", "price", "not mentioned", "not contain"],  # Should acknowledge lack of info
        "category": "irrelevant"
    }
]

class RAGTestSuite:
    def __init__(self):
        self.results = []
        self.start_time = None
        
    async def setup(self):
        """Initialize RAG components"""
        try:
            print("🔧 Setting up RAG components...")
            
            # Import RAG components
            from agents.toolbox.rag_tools import simple_rag, _ensure_retriever
            
            # Test retriever initialization
            retriever = await _ensure_retriever()
            print(f"✅ Retriever initialized: {type(retriever).__name__}")
            
            # Test basic retrieval
            test_results = await retriever.retrieve("test", top_k=1, threshold=0.0)
            print(f"✅ Basic retrieval works: {len(test_results)} results available")
            
            self.simple_rag = simple_rag
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case["id"]
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        category = test_case["category"]
        
        print(f"\n🧪 Testing [{test_id}]: {question[:60]}...")
        
        start_time = datetime.now()
        
        try:
            # Call RAG tool
            response = await self.simple_rag.ainvoke({
                'question': question,
                'top_k': 5
            })
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Analyze response
            analysis = self.analyze_response(response, expected_keywords, category)
            
            result = {
                "test_id": test_id,
                "question": question,
                "category": category,
                "response": response,
                "response_time": response_time,
                "expected_keywords": expected_keywords,
                "analysis": analysis,
                "status": "PASS" if analysis["score"] >= 0.7 else "FAIL",
                "timestamp": datetime.now().isoformat()
            }
            
            # Print result
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {test_id}: {result['status']} (Score: {analysis['score']:.2f}, Time: {response_time:.2f}s)")
            
            if result["status"] == "FAIL":
                print(f"   Missing keywords: {analysis['missing_keywords']}")
                print(f"   Response preview: {response[:100]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ {test_id}: ERROR - {e}")
            return {
                "test_id": test_id,
                "question": question,
                "category": category,
                "error": str(e),
                "status": "ERROR",
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_response(self, response: str, expected_keywords: List[str], category: str) -> Dict[str, Any]:
        """Analyze response quality"""
        response_lower = response.lower()
        
        # For irrelevant questions, success means acknowledging lack of relevant info
        if category == "irrelevant":
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
            missing_keywords = [kw for kw in expected_keywords if kw.lower() not in response_lower]
            
            # Success if it acknowledges the lack of relevant information
            acknowledges_lack = any(phrase in response_lower for phrase in [
                "not mentioned", "not contain", "do not contain", "not explicitly mentioned",
                "not available", "no information", "recommend visiting", "contact"
            ])
            
            # Calculate score based on acknowledgment and keyword coverage
            keyword_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.5
            acknowledgment_bonus = 0.5 if acknowledges_lack else 0.0
            
            score = min(1.0, keyword_score + acknowledgment_bonus)
            
            return {
                "score": score,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "acknowledges_lack": acknowledges_lack,
                "response_length": len(response)
            }
        
        # For regular questions, check keyword coverage
        found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
        missing_keywords = [kw for kw in expected_keywords if kw.lower() not in response_lower]
        
        # Calculate score based on keyword coverage
        if len(expected_keywords) == 0:
            keyword_score = 1.0
        else:
            keyword_score = len(found_keywords) / len(expected_keywords)
        
        # Bonus points for comprehensive response
        length_bonus = min(0.2, len(response) / 1000)  # Up to 0.2 bonus for longer responses
        
        # Penalty for "No documents found" when we expect to find something
        no_docs_penalty = 0.5 if "no relevant documents found" in response_lower else 0.0
        
        final_score = max(0.0, keyword_score + length_bonus - no_docs_penalty)
        
        return {
            "score": final_score,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "keyword_coverage": keyword_score,
            "response_length": len(response),
            "has_no_docs_message": no_docs_penalty > 0
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        self.start_time = datetime.now()
        print(f"🚀 Starting RAG End-to-End Test Suite at {self.start_time}")
        print(f"📋 Running {len(TEST_QUESTIONS)} test cases...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            return {"status": "SETUP_FAILED", "results": []}
        
        # Run tests
        self.results = []
        for test_case in TEST_QUESTIONS:
            result = await self.run_single_test(test_case)
            self.results.append(result)
        
        # Generate summary
        summary = self.generate_summary()
        
        print(f"\n📊 Test Suite Complete!")
        print(f"⏱️  Total time: {summary['total_time']:.2f}s")
        print(f"✅ Passed: {summary['passed']}/{summary['total']}")
        print(f"❌ Failed: {summary['failed']}/{summary['total']}")
        print(f"🔥 Errors: {summary['errors']}/{summary['total']}")
        print(f"📈 Success rate: {summary['success_rate']:.1%}")
        
        return {
            "status": "COMPLETED",
            "summary": summary,
            "results": self.results
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total = len(self.results)
        passed = len([r for r in self.results if r.get("status") == "PASS"])
        failed = len([r for r in self.results if r.get("status") == "FAIL"])
        errors = len([r for r in self.results if r.get("status") == "ERROR"])
        
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if result.get("status") == "PASS":
                categories[cat]["passed"] += 1
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": passed / total if total > 0 else 0,
            "total_time": total_time,
            "avg_response_time": sum(r.get("response_time", 0) for r in self.results) / total if total > 0 else 0,
            "categories": categories
        }
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_test_results_{timestamp}.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.generate_summary(),
            "results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename

async def main():
    """Main test runner"""
    suite = RAGTestSuite()
    results = await suite.run_all_tests()
    
    # Save results
    suite.save_results()
    
    # Return exit code based on success
    if results["status"] == "COMPLETED":
        summary = results["summary"]
        if summary["success_rate"] >= 0.8:  # 80% success rate required
            print("\n🎉 Test suite PASSED! (≥80% success rate)")
            return 0
        else:
            print(f"\n💥 Test suite FAILED! ({summary['success_rate']:.1%} success rate < 80%)")
            return 1
    else:
        print("\n💥 Test suite FAILED! (Setup error)")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
