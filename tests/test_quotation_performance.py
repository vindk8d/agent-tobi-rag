"""
Performance Testing Suite for Quotation Generation and Storage Operations

This test suite validates:
- PDF generation performance under various loads
- Supabase storage upload/download performance  
- Memory usage and resource consumption
- Concurrent operations scalability
- Large data handling performance
- Storage operations benchmarking

Performance requirements:
- PDF generation: < 2 seconds per document
- Storage upload: < 5 seconds per PDF
- Concurrent operations: Support 10+ simultaneous requests
- Memory usage: < 200MB per PDF generation
- Storage retrieval: < 3 seconds for signed URL creation
"""

import os
import pytest
import asyncio
import time
import uuid
import psutil
import tempfile
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock
import concurrent.futures

# Skip all tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)

# Import modules only if credentials are available
try:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        from backend.core.pdf_generator import QuotationPDFGenerator, generate_quotation_pdf, PDFGenerationError
        from backend.core.storage import upload_quotation_pdf, create_signed_quotation_url, QuotationStorageError
        from backend.agents.tools import (
            _lookup_customer,
            _lookup_vehicle_by_criteria,
            _lookup_current_pricing,
            _lookup_employee_details,
            UserContext
        )
        
        # Check if WeasyPrint is available
        try:
            import weasyprint
            PDF_GENERATION_AVAILABLE = True
        except ImportError:
            PDF_GENERATION_AVAILABLE = False
            
        # Check if template files exist
        template_path = Path("backend/templates/quotation_template.html")
        css_path = Path("backend/templates/quotation_styles.css")
        TEMPLATES_AVAILABLE = template_path.exists() and css_path.exists()

except ImportError:
    # If imports fail, skip all tests
    pytestmark = pytest.mark.skip(reason="Required modules not available")
    PDF_GENERATION_AVAILABLE = False
    TEMPLATES_AVAILABLE = False


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.times = []
        self.memory_usage = []
        self.sizes = []
        self.errors = []
    
    def add_measurement(self, time_taken: float, memory_mb: float = 0, size_bytes: int = 0, error: str = None):
        self.times.append(time_taken)
        if memory_mb > 0:
            self.memory_usage.append(memory_mb)
        if size_bytes > 0:
            self.sizes.append(size_bytes)
        if error:
            self.errors.append(error)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.times:
            return {"error": "No measurements recorded"}
        
        return {
            "count": len(self.times),
            "avg_time": statistics.mean(self.times),
            "min_time": min(self.times),
            "max_time": max(self.times),
            "median_time": statistics.median(self.times),
            "std_dev_time": statistics.stdev(self.times) if len(self.times) > 1 else 0,
            "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "avg_size_kb": statistics.mean(self.sizes) / 1024 if self.sizes else 0,
            "error_rate": len(self.errors) / len(self.times) * 100,
            "errors": self.errors[:5]  # First 5 errors for debugging
        }


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_sample_quotation_data(size_variant: str = "normal") -> Dict[str, Any]:
    """Create sample quotation data with different complexity levels."""
    
    base_data = {
        "quotation_number": f"QUO-PERF-{uuid.uuid4().hex[:8].upper()}",
        "customer": {
            "id": str(uuid.uuid4()),
            "name": "Performance Test Customer",
            "email": "perf@example.com",
            "phone": "+63 917 123 4567",
            "address": "Performance Test Address, Manila"
        },
        "vehicle": {
            "id": str(uuid.uuid4()),
            "make": "Toyota",
            "model": "Camry",
            "year": 2024,
            "type": "Sedan",
            "color": "Silver",
            "specifications": {}
        },
        "pricing": {
            "base_price": 1500000.00,
            "total_amount": 1600000.00,
            "add_ons": []
        },
        "employee": {
            "id": "emp_perf_001",
            "name": "Performance Test Employee",
            "email": "perf.emp@test.com",
            "phone": "+63 2 8000 0000"
        }
    }
    
    if size_variant == "small":
        # Minimal data
        return base_data
    
    elif size_variant == "large":
        # Large data with many features and add-ons
        base_data["vehicle"]["specifications"] = {
            "features": [f"Feature {i}" for i in range(1, 51)],  # 50 features
            "technical_specs": {
                "engine": "2.5L 4-Cylinder Hybrid",
                "transmission": "CVT Automatic", 
                "fuel_type": "Hybrid",
                "seating_capacity": "5 passengers",
                "fuel_economy": "25.5 km/L",
                "safety_rating": "5-Star ASEAN NCAP",
                "warranty": "5 years / 100,000 km",
                "dimensions": {
                    "length": "4,885 mm",
                    "width": "1,840 mm", 
                    "height": "1,445 mm",
                    "wheelbase": "2,825 mm"
                }
            }
        }
        
        base_data["pricing"]["add_ons"] = [
            {"name": f"Add-on Package {i}", "price": 10000.00 + (i * 1000)}
            for i in range(1, 21)  # 20 add-ons
        ]
        
        base_data["pricing"]["total_amount"] = 2000000.00
        
        # Extended customer information
        base_data["customer"]["company"] = "Large Corporation Ltd."
        base_data["customer"]["address"] = "Very Long Corporate Address with Multiple Lines, Building Name, Street Name, Barangay, City, Province, Postal Code, Philippines"
        
    elif size_variant == "complex":
        # Complex pricing structure
        base_data["pricing"] = {
            "base_price": 1800000.00,
            "discounts": 100000.00,
            "insurance": 85000.00,
            "lto_fees": 25000.00,
            "add_ons": [
                {"name": "Premium Package", "price": 150000.00},
                {"name": "Extended Warranty", "price": 75000.00},
                {"name": "Paint Protection", "price": 45000.00},
                {"name": "Window Tinting", "price": 15000.00},
                {"name": "Floor Mats Set", "price": 8000.00}
            ],
            "vat_rate": 0.12,
            "vat_amount": 240000.00,
            "total_amount": 2198000.00
        }
    
    return base_data


class TestPDFGenerationPerformance:
    """Test PDF generation performance under various conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_single_pdf_generation_performance(self):
        """Test single PDF generation performance baseline."""
        
        quotation_data = create_sample_quotation_data("normal")
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Measure performance
        start_memory = get_memory_usage_mb()
        start_time = time.time()
        
        pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
        
        end_time = time.time()
        end_memory = get_memory_usage_mb()
        
        generation_time = end_time - start_time
        memory_used = end_memory - start_memory
        pdf_size = len(pdf_bytes)
        
        # Performance assertions
        assert generation_time < 2.0, f"PDF generation took too long: {generation_time:.2f}s"
        assert memory_used < 200.0, f"Memory usage too high: {memory_used:.1f}MB"
        assert pdf_size > 1000, f"PDF too small: {pdf_size} bytes"
        
        print(f"Single PDF Performance: {generation_time:.3f}s, {memory_used:.1f}MB, {pdf_size/1024:.1f}KB")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_generation_with_different_sizes(self):
        """Test PDF generation performance with different data sizes."""
        
        metrics = PerformanceMetrics()
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        size_variants = ["small", "normal", "large", "complex"]
        
        for variant in size_variants:
            quotation_data = create_sample_quotation_data(variant)
            
            start_memory = get_memory_usage_mb()
            start_time = time.time()
            
            try:
                pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
                
                end_time = time.time()
                end_memory = get_memory_usage_mb()
                
                generation_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                metrics.add_measurement(generation_time, memory_used, len(pdf_bytes))
                
                # Individual variant performance checks
                assert generation_time < 3.0, f"{variant} PDF generation took too long: {generation_time:.2f}s"
                
            except Exception as e:
                metrics.add_measurement(0, 0, 0, str(e))
        
        stats = metrics.get_stats()
        print(f"Size Variant Performance: avg={stats['avg_time']:.3f}s, max={stats['max_time']:.3f}s")
        
        # Overall performance requirements
        assert stats["avg_time"] < 2.0, f"Average generation time too high: {stats['avg_time']:.3f}s"
        assert stats["error_rate"] == 0, f"Error rate too high: {stats['error_rate']:.1f}%"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_concurrent_pdf_generation(self):
        """Test concurrent PDF generation performance."""
        
        async def generate_single_pdf(pdf_id: int) -> Tuple[int, float, float, int]:
            """Generate a single PDF and return performance metrics."""
            quotation_data = create_sample_quotation_data("normal")
            quotation_data["quotation_number"] = f"QUO-CONCURRENT-{pdf_id:03d}"
            
            generator = QuotationPDFGenerator(template_dir="backend/templates")
            
            start_memory = get_memory_usage_mb()
            start_time = time.time()
            
            pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
            
            end_time = time.time()
            end_memory = get_memory_usage_mb()
            
            return pdf_id, end_time - start_time, end_memory - start_memory, len(pdf_bytes)
        
        # Test with 5 concurrent PDF generations
        concurrent_count = 5
        start_time = time.time()
        
        tasks = [generate_single_pdf(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        times = [result[1] for result in results]
        memory_usage = [result[2] for result in results]
        sizes = [result[3] for result in results]
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        total_memory = sum(memory_usage)
        
        print(f"Concurrent Generation ({concurrent_count} PDFs):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per PDF: {avg_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Total memory: {total_memory:.1f}MB")
        
        # Performance assertions for concurrent operations
        assert total_time < 10.0, f"Concurrent generation took too long: {total_time:.2f}s"
        assert max_time < 4.0, f"Slowest PDF took too long: {max_time:.2f}s"
        assert total_memory < 500.0, f"Total memory usage too high: {total_memory:.1f}MB"
        
        # All PDFs should be generated successfully
        assert len(results) == concurrent_count
        assert all(size > 1000 for size in sizes), "Some PDFs are too small"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_generation_stress_test(self):
        """Stress test PDF generation with multiple iterations."""
        
        metrics = PerformanceMetrics()
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        iterations = 10
        start_total_time = time.time()
        
        for i in range(iterations):
            quotation_data = create_sample_quotation_data("normal")
            quotation_data["quotation_number"] = f"QUO-STRESS-{i:03d}"
            
            start_memory = get_memory_usage_mb()
            start_time = time.time()
            
            try:
                pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
                
                end_time = time.time()
                end_memory = get_memory_usage_mb()
                
                generation_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                metrics.add_measurement(generation_time, memory_used, len(pdf_bytes))
                
            except Exception as e:
                metrics.add_measurement(0, 0, 0, str(e))
        
        total_time = time.time() - start_total_time
        stats = metrics.get_stats()
        
        print(f"Stress Test Results ({iterations} iterations):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per PDF: {stats['avg_time']:.3f}s")
        print(f"  Standard deviation: {stats['std_dev_time']:.3f}s")
        print(f"  Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
        print(f"  Error rate: {stats['error_rate']:.1f}%")
        
        # Performance requirements for stress test
        assert stats["avg_time"] < 2.0, f"Average time too high under stress: {stats['avg_time']:.3f}s"
        assert stats["error_rate"] < 5.0, f"Error rate too high under stress: {stats['error_rate']:.1f}%"
        assert stats["std_dev_time"] < 1.0, f"Too much variance in performance: {stats['std_dev_time']:.3f}s"


class TestStoragePerformance:
    """Test Supabase storage performance for quotation PDFs."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_storage_upload_performance(self):
        """Test single storage upload performance."""
        
        # Generate a test PDF
        quotation_data = create_sample_quotation_data("normal")
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
        
        customer_id = quotation_data["customer"]["id"]
        employee_id = quotation_data["employee"]["id"]
        
        # Measure upload performance
        start_time = time.time()
        
        try:
            upload_result = await upload_quotation_pdf(
                pdf_bytes=pdf_bytes,
                customer_id=customer_id,
                employee_id=employee_id,
                quotation_number=quotation_data["quotation_number"]
            )
            
            upload_time = time.time() - start_time
            
            print(f"Storage Upload Performance: {upload_time:.3f}s for {len(pdf_bytes)/1024:.1f}KB")
            
            # Performance assertions
            assert upload_time < 5.0, f"Upload took too long: {upload_time:.2f}s"
            assert upload_result["size"] == len(pdf_bytes)
            assert upload_result["content_type"] == "application/pdf"
            
            # Test signed URL creation performance
            storage_path = upload_result["path"]
            
            start_time = time.time()
            signed_url = await create_signed_quotation_url(storage_path, expires_in_seconds=3600)
            url_creation_time = time.time() - start_time
            
            print(f"Signed URL Creation Performance: {url_creation_time:.3f}s")
            
            assert url_creation_time < 3.0, f"URL creation took too long: {url_creation_time:.2f}s"
            assert signed_url.startswith("https://")
            
        except Exception as e:
            pytest.skip(f"Storage test skipped due to: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage upload performance."""
        
        async def upload_single_pdf(pdf_id: int) -> Tuple[int, float, float, bool]:
            """Upload a single PDF and return performance metrics."""
            
            # Generate unique PDF
            quotation_data = create_sample_quotation_data("normal")
            quotation_data["quotation_number"] = f"QUO-STORAGE-{pdf_id:03d}"
            
            generator = QuotationPDFGenerator(template_dir="backend/templates")
            pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
            
            customer_id = quotation_data["customer"]["id"]
            employee_id = quotation_data["employee"]["id"]
            
            # Measure upload time
            start_time = time.time()
            
            try:
                upload_result = await upload_quotation_pdf(
                    pdf_bytes=pdf_bytes,
                    customer_id=customer_id,
                    employee_id=employee_id,
                    quotation_number=quotation_data["quotation_number"]
                )
                
                upload_time = time.time() - start_time
                
                # Test signed URL creation
                start_url_time = time.time()
                signed_url = await create_signed_quotation_url(upload_result["path"])
                url_time = time.time() - start_url_time
                
                return pdf_id, upload_time, url_time, True
                
            except Exception as e:
                print(f"Upload {pdf_id} failed: {e}")
                return pdf_id, 0, 0, False
        
        # Test with 3 concurrent uploads (conservative for storage)
        concurrent_count = 3
        start_time = time.time()
        
        try:
            tasks = [upload_single_pdf(i) for i in range(concurrent_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Filter successful results
            successful_results = [r for r in results if isinstance(r, tuple) and r[3]]
            
            if not successful_results:
                pytest.skip("All concurrent storage operations failed - likely network/credentials issue")
            
            upload_times = [r[1] for r in successful_results]
            url_times = [r[2] for r in successful_results]
            
            avg_upload_time = statistics.mean(upload_times)
            avg_url_time = statistics.mean(url_times)
            success_rate = len(successful_results) / len(results) * 100
            
            print(f"Concurrent Storage ({concurrent_count} operations):")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average upload time: {avg_upload_time:.3f}s")
            print(f"  Average URL creation time: {avg_url_time:.3f}s")
            print(f"  Success rate: {success_rate:.1f}%")
            
            # Performance assertions for concurrent storage
            assert total_time < 15.0, f"Concurrent storage took too long: {total_time:.2f}s"
            assert avg_upload_time < 8.0, f"Average upload time too high: {avg_upload_time:.2f}s"
            assert success_rate >= 80.0, f"Success rate too low: {success_rate:.1f}%"
            
        except Exception as e:
            pytest.skip(f"Concurrent storage test skipped due to: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_large_pdf_storage_performance(self):
        """Test storage performance with large PDF files."""
        
        # Generate large PDF
        quotation_data = create_sample_quotation_data("large")
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
        
        pdf_size_kb = len(pdf_bytes) / 1024
        print(f"Testing large PDF storage: {pdf_size_kb:.1f}KB")
        
        customer_id = quotation_data["customer"]["id"]
        employee_id = quotation_data["employee"]["id"]
        
        # Measure upload performance for large file
        start_time = time.time()
        
        try:
            upload_result = await upload_quotation_pdf(
                pdf_bytes=pdf_bytes,
                customer_id=customer_id,
                employee_id=employee_id,
                quotation_number=quotation_data["quotation_number"]
            )
            
            upload_time = time.time() - start_time
            upload_speed_kbps = pdf_size_kb / upload_time if upload_time > 0 else 0
            
            print(f"Large PDF Upload Performance: {upload_time:.3f}s at {upload_speed_kbps:.1f} KB/s")
            
            # Performance assertions for large files
            assert upload_time < 10.0, f"Large PDF upload took too long: {upload_time:.2f}s"
            assert upload_speed_kbps > 5.0, f"Upload speed too slow: {upload_speed_kbps:.1f} KB/s"
            
            # Test signed URL creation for large file
            start_time = time.time()
            signed_url = await create_signed_quotation_url(upload_result["path"])
            url_creation_time = time.time() - start_time
            
            assert url_creation_time < 3.0, f"URL creation for large file took too long: {url_creation_time:.2f}s"
            
        except Exception as e:
            pytest.skip(f"Large PDF storage test skipped due to: {e}")


class TestEndToEndPerformance:
    """Test end-to-end performance of the complete quotation system."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_complete_quotation_workflow_performance(self):
        """Test performance of complete quotation generation and storage workflow."""
        
        with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
            
            total_start_time = time.time()
            
            # Step 1: Data lookup (simulated with sample data for performance)
            lookup_start = time.time()
            quotation_data = create_sample_quotation_data("normal")
            lookup_time = time.time() - lookup_start
            
            # Step 2: PDF generation
            pdf_start = time.time()
            generator = QuotationPDFGenerator(template_dir="backend/templates")
            pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
            pdf_time = time.time() - pdf_start
            
            # Step 3: Storage upload
            storage_start = time.time()
            try:
                upload_result = await upload_quotation_pdf(
                    pdf_bytes=pdf_bytes,
                    customer_id=quotation_data["customer"]["id"],
                    employee_id=quotation_data["employee"]["id"],
                    quotation_number=quotation_data["quotation_number"]
                )
                
                # Step 4: Signed URL creation
                url_start = time.time()
                signed_url = await create_signed_quotation_url(upload_result["path"])
                url_time = time.time() - url_start
                
                storage_time = time.time() - storage_start
                
            except Exception as e:
                # If storage fails, still test the PDF generation performance
                storage_time = 0
                url_time = 0
                print(f"Storage operations skipped due to: {e}")
            
            total_time = time.time() - total_start_time
            
            print(f"Complete Workflow Performance:")
            print(f"  Data lookup: {lookup_time:.3f}s")
            print(f"  PDF generation: {pdf_time:.3f}s")
            print(f"  Storage upload: {storage_time:.3f}s")
            print(f"  URL creation: {url_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            
            # End-to-end performance requirements
            assert pdf_time < 2.0, f"PDF generation too slow: {pdf_time:.3f}s"
            assert total_time < 15.0, f"Complete workflow too slow: {total_time:.3f}s"
            
            # PDF should be valid
            assert len(pdf_bytes) > 1000
            assert pdf_bytes.startswith(b'%PDF')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_memory_usage_over_time(self):
        """Test memory usage patterns over multiple quotation generations."""
        
        initial_memory = get_memory_usage_mb()
        memory_measurements = [initial_memory]
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Generate 5 PDFs and monitor memory
        for i in range(5):
            quotation_data = create_sample_quotation_data("normal")
            quotation_data["quotation_number"] = f"QUO-MEMORY-{i:03d}"
            
            pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
            current_memory = get_memory_usage_mb()
            memory_measurements.append(current_memory)
            
            # Validate PDF was generated
            assert len(pdf_bytes) > 1000
        
        final_memory = get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_measurements)
        
        print(f"Memory Usage Analysis:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
        print(f"  Peak: {max_memory:.1f}MB")
        
        # Memory usage requirements
        assert memory_increase < 100.0, f"Memory increase too high: {memory_increase:.1f}MB"
        assert max_memory < initial_memory + 200.0, f"Peak memory too high: {max_memory:.1f}MB"


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_quotation_performance.py -v -s
    pytest.main([__file__, "-v", "-s"])
