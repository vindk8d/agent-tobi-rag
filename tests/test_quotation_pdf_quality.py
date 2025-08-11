"""
PDF Quality Validation Tests for Quotation Generation

Tests the PDF generation system with real CRM data to ensure:
- Professional appearance and layout
- Accurate data rendering
- Proper formatting and styling
- Edge case handling
- Performance with various data scenarios

This test suite validates that generated PDFs meet business requirements
for professional vehicle quotations.
"""

import os
import pytest
import asyncio
import uuid
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# Skip all tests if Supabase credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"),
    reason="Supabase credentials not available - set SUPABASE_URL and SUPABASE_SERVICE_KEY"
)

# Import modules only if credentials are available
try:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        from backend.core.pdf_generator import QuotationPDFGenerator, generate_quotation_pdf, PDFGenerationError
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


class TestPDFQualityWithRealData:
    """Test PDF generation quality with real CRM data."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_generation_with_complete_real_data(self):
        """Test PDF generation with complete real CRM data."""
        
        # Get real data from CRM system
        with UserContext(user_id="emp_001", user_type="employee", employee_id="emp_001"):
            # Look up real customer data
            customer_data = await _lookup_customer("john@example.com")
            if not customer_data:
                # Use realistic sample data if no real customer found
                customer_data = {
                    "id": str(uuid.uuid4()),
                    "name": "Juan Carlos Santos",
                    "email": "juan.santos@gmail.com",
                    "phone": "+63 917 123 4567",
                    "address": "123 Makati Avenue, Makati City, Metro Manila 1200",
                    "company": "Santos Construction Corp."
                }
            
            # Look up real vehicle data
            vehicles = await _lookup_vehicle_by_criteria({"make": "Toyota", "model": "Camry"}, limit=1)
            if vehicles:
                vehicle_data = vehicles[0]
                # Convert database fields to expected PDF format
                if "brand" in vehicle_data and "make" not in vehicle_data:
                    vehicle_data["make"] = vehicle_data["brand"]
                # Ensure required fields exist
                vehicle_data.setdefault("specifications", {})
            else:
                # Use realistic sample vehicle data
                vehicle_data = {
                    "id": str(uuid.uuid4()),
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2024,
                    "type": "Sedan",
                    "color": "Pearl White",
                    "engine": "2.5L 4-Cylinder Hybrid",
                    "transmission": "CVT Automatic",
                    "fuel_type": "Hybrid",
                    "specifications": {
                        "seating_capacity": "5 passengers",
                        "fuel_economy": "25.5 km/L",
                        "safety_rating": "5-Star ASEAN NCAP",
                        "warranty": "5 years / 100,000 km",
                        "features": [
                            "Toyota Safety Sense 2.0",
                            "Leather-appointed seats",
                            "9-inch touchscreen display",
                            "Wireless phone charging",
                            "Dual-zone automatic climate control",
                            "LED headlights and taillights"
                        ]
                    }
                }
            
            # Look up real employee data
            employee_data = await _lookup_employee_details("emp_001")
            if not employee_data:
                # Use realistic sample employee data
                employee_data = {
                    "id": "emp_001",
                    "name": "Maria Isabella Rodriguez",
                    "email": "maria.rodriguez@premiummotors.ph",
                    "phone": "+63 2 8123 4567",
                    "position": "Senior Sales Consultant",
                    "branch": "Makati Showroom",
                    "branch_address": "456 Ayala Avenue, Makati City"
                }
            
            # Look up real pricing data
            vehicle_id = vehicle_data.get("id") if vehicle_data else str(uuid.uuid4())
            pricing_data = await _lookup_current_pricing(vehicle_id)
            if not pricing_data:
                # Use realistic pricing structure
                pricing_data = {
                    "base_price": 1580000.00,
                    "discounts": 50000.00,
                    "insurance": 45000.00,
                    "lto_fees": 15000.00,
                    "add_ons": [
                        {"name": "Extended Warranty (2 years)", "price": 25000.00},
                        {"name": "Paint Protection Film", "price": 35000.00},
                        {"name": "Premium Floor Mats", "price": 8000.00}
                    ],
                    "total_amount": 1613000.00
                }
            else:
                # Ensure required fields exist in real pricing data
                if "total_amount" not in pricing_data and "total_price" in pricing_data:
                    pricing_data["total_amount"] = pricing_data["total_price"]
                elif "total_amount" not in pricing_data:
                    # Calculate total if missing
                    base = pricing_data.get("base_price", 0)
                    insurance = pricing_data.get("insurance", 0)
                    lto = pricing_data.get("lto_fees", 0)
                    discounts = pricing_data.get("discounts", 0)
                    add_on_total = sum(item.get("price", 0) for item in pricing_data.get("add_ons", []))
                    pricing_data["total_amount"] = base + insurance + lto + add_on_total - discounts
        
        # Create complete quotation data
        quotation_data = {
            "quotation_number": f"QUO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            "customer": customer_data,
            "vehicle": vehicle_data,
            "pricing": pricing_data,
            "employee": employee_data,
            "validity_days": 30,
            "created_date": datetime.now(),
            "notes": "Special promotional pricing valid until end of month. Includes complimentary 3-year service package."
        }
        
        # Generate PDF
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
        
        # Validate PDF generation
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b'%PDF')  # PDF signature
        
        # Validate PDF size (should be reasonable for a quotation)
        pdf_size_kb = len(pdf_bytes) / 1024
        assert 10 < pdf_size_kb < 2000, f"PDF size {pdf_size_kb:.1f}KB seems unreasonable"
        
        # Save PDF for manual inspection if needed
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
            
        print(f"Generated PDF saved to: {temp_path}")
        print(f"PDF size: {pdf_size_kb:.1f} KB")
        
        # Clean up
        os.unlink(temp_path)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_with_minimal_data(self):
        """Test PDF generation with minimal required data."""
        
        minimal_data = {
            "quotation_number": "QUO-TEST-001",
            "customer": {
                "name": "Test Customer",
                "email": "test@example.com",
                "phone": "+63 912 345 6789",
                "address": "Test Address"
            },
            "vehicle": {
                "make": "Toyota",
                "model": "Vios",
                "year": 2024,
                "type": "Sedan",
                "color": "White"
            },
            "pricing": {
                "base_price": 800000.00,
                "total_amount": 850000.00
            },
            "employee": {
                "name": "Test Employee",
                "email": "employee@test.com",
                "phone": "+63 2 8000 0000"
            }
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(minimal_data)
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b'%PDF')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_with_complex_pricing(self):
        """Test PDF generation with complex pricing scenarios."""
        
        complex_pricing_data = {
            "quotation_number": "QUO-COMPLEX-001",
            "customer": {
                "name": "Corporate Client Ltd.",
                "email": "procurement@corporate.com",
                "phone": "+63 2 8888 9999",
                "address": "Corporate Plaza, BGC, Taguig City",
                "company": "Corporate Client Ltd."
            },
            "vehicle": {
                "make": "Toyota",
                "model": "Hilux",
                "year": 2024,
                "type": "Pickup Truck",
                "color": "Attitude Black Metallic",
                "engine": "2.4L Turbo Diesel",
                "transmission": "6-Speed Automatic",
                "specifications": {
                    "seating_capacity": "5 passengers",
                    "payload_capacity": "1,080 kg",
                    "towing_capacity": "3,500 kg",
                    "ground_clearance": "279 mm",
                    "features": [
                        "4WD with Active Traction Control",
                        "Hill Start Assist",
                        "Vehicle Stability Control",
                        "7-inch touchscreen with Apple CarPlay/Android Auto",
                        "Reverse camera with guidelines"
                    ]
                }
            },
            "pricing": {
                "base_price": 1890000.00,
                "discounts": 100000.00,
                "insurance": 85000.00,
                "lto_fees": 25000.00,
                "add_ons": [
                    {"name": "Tonneau Cover", "price": 45000.00},
                    {"name": "Side Steps", "price": 25000.00},
                    {"name": "Mudguards Set", "price": 8000.00},
                    {"name": "Tinted Windows", "price": 15000.00},
                    {"name": "Extended Warranty (3 years)", "price": 40000.00}
                ],
                "total_amount": 2033000.00,
                "vat_rate": 0.12,
                "vat_amount": 218036.00
            },
            "employee": {
                "name": "Roberto Martinez",
                "email": "roberto.martinez@premiummotors.ph",
                "phone": "+63 2 8123 4568",
                "position": "Fleet Sales Manager",
                "branch": "BGC Showroom"
            },
            "validity_days": 45,
            "delivery_time": "6-8 weeks",
            "reservation_fee": 100000.00,
            "notes": "Fleet discount applied. Bulk purchase terms available for orders of 5+ units."
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(complex_pricing_data)
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        # Save for inspection
        with tempfile.NamedTemporaryFile(suffix='_complex.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
            
        print(f"Complex pricing PDF saved to: {temp_path}")
        
        # Clean up
        os.unlink(temp_path)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_with_special_characters(self):
        """Test PDF generation with special characters and Unicode."""
        
        unicode_data = {
            "quotation_number": "QUO-UNICODE-001",
            "customer": {
                "name": "José María Rizal-Sánchez",
                "email": "josé@example.com",
                "phone": "+63 917 555 0123",
                "address": "123 Magsaysay Ave., Quezon City, Metro Manila",
                "company": "Rizal & Associates Co."
            },
            "vehicle": {
                "make": "Toyota",
                "model": "RAV4",
                "year": 2024,
                "type": "SUV",
                "color": "Magnetic Gray Metallic",
                "specifications": {
                    "features": [
                        "Multi-terrain Select (MTS)",
                        "Dynamic Torque Control 4WD",
                        "Pre-collision System (PCS)",
                        "Lane Departure Alert (LDA)"
                    ]
                }
            },
            "pricing": {
                "base_price": 2150000.00,
                "total_amount": 2280000.00
            },
            "employee": {
                "name": "María Esperanza Santos-Cruz",
                "email": "maria.santos@premiummotors.ph",
                "phone": "+63 2 8123 4569"
            },
            "notes": "Pricing includes: ₱50,000 cash discount, comprehensive insurance, and 5-year extended warranty."
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        pdf_bytes = await generator.generate_quotation_pdf(unicode_data)
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_html_preview_generation(self):
        """Test HTML preview generation for debugging."""
        
        sample_data = {
            "quotation_number": "QUO-PREVIEW-001",
            "customer": {
                "name": "Preview Customer",
                "email": "preview@example.com",
                "phone": "+63 912 000 0000",
                "address": "Preview Address"
            },
            "vehicle": {
                "make": "Toyota",
                "model": "Corolla Cross",
                "year": 2024,
                "type": "Crossover SUV",
                "color": "Celestite Gray Metallic"
            },
            "pricing": {
                "base_price": 1235000.00,
                "total_amount": 1320000.00
            },
            "employee": {
                "name": "Preview Employee",
                "email": "preview.emp@test.com",
                "phone": "+63 2 8000 1111"
            }
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        html_content = await generator.preview_html(sample_data)
        
        assert html_content is not None
        assert len(html_content) > 0
        assert "QUO-PREVIEW-001" in html_content
        assert "Preview Customer" in html_content
        assert "Toyota" in html_content
        assert "Corolla Cross" in html_content

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_error_handling(self):
        """Test PDF generation error handling."""
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Test missing required fields
        with pytest.raises(PDFGenerationError) as exc_info:
            await generator.generate_quotation_pdf({})
        assert "Missing required field" in str(exc_info.value)
        
        # Test invalid customer data
        with pytest.raises(PDFGenerationError) as exc_info:
            await generator.generate_quotation_pdf({
                "quotation_number": "TEST",
                "customer": "invalid",  # Should be dict
                "vehicle": {},
                "pricing": {},
                "employee": {}
            })
        assert "must be of type dict" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_save_functionality(self):
        """Test PDF save to file functionality."""
        
        sample_data = {
            "quotation_number": "QUO-SAVE-001",
            "customer": {
                "name": "Save Test Customer",
                "email": "save@example.com",
                "phone": "+63 912 000 0001",
                "address": "Save Test Address"
            },
            "vehicle": {
                "make": "Toyota",
                "model": "Innova",
                "year": 2024,
                "type": "MPV",
                "color": "Silver Metallic"
            },
            "pricing": {
                "base_price": 1450000.00,
                "total_amount": 1550000.00
            },
            "employee": {
                "name": "Save Test Employee",
                "email": "save.emp@test.com",
                "phone": "+63 2 8000 2222"
            }
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            file_path, file_size = await generator.generate_and_save_pdf(sample_data, temp_path)
            
            assert file_path == temp_path
            assert file_size > 0
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size == file_size
            
        finally:
            # Clean up
            if Path(temp_path).exists():
                os.unlink(temp_path)


class TestPDFDataFormatting:
    """Test PDF data formatting and display."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_currency_formatting(self):
        """Test currency formatting in PDFs."""
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Test currency formatting
        assert generator._format_currency(1234567.89) == "₱ 1,234,567.89"
        assert generator._format_currency(0) == "₱ 0.00"
        assert generator._format_currency(1000) == "₱ 1,000.00"
        
        # Test with different currency
        assert generator._format_currency(1000, "$") == "$ 1,000.00"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_phone_formatting(self):
        """Test phone number formatting in PDFs."""
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        # Test Philippine mobile number formatting
        assert generator._format_phone("09171234567") == "+63 917 123 4567"
        assert generator._format_phone("9171234567") == "+63 917 123 4567"
        assert generator._format_phone("+63 917 123 4567") == "+63 917 123 4567"
        
        # Test invalid/empty numbers
        assert generator._format_phone("") == "N/A"
        assert generator._format_phone(None) == "N/A"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_date_formatting(self):
        """Test date formatting in PDFs."""
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        test_date = datetime(2024, 12, 25, 10, 30, 0)
        
        # Test default formatting
        assert generator._format_date(test_date) == "December 25, 2024"
        
        # Test custom formatting
        assert generator._format_date(test_date, "%Y-%m-%d") == "2024-12-25"
        assert generator._format_date(test_date, "%d/%m/%Y") == "25/12/2024"
        
        # Test string input
        iso_string = "2024-12-25T10:30:00Z"
        formatted = generator._format_date(iso_string)
        assert "December 25, 2024" == formatted


class TestPDFPerformance:
    """Test PDF generation performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_pdf_generation_performance(self):
        """Test PDF generation performance with timing."""
        
        sample_data = {
            "quotation_number": "QUO-PERF-001",
            "customer": {
                "name": "Performance Test Customer",
                "email": "perf@example.com",
                "phone": "+63 912 000 0003",
                "address": "Performance Test Address"
            },
            "vehicle": {
                "make": "Toyota",
                "model": "Land Cruiser",
                "year": 2024,
                "type": "SUV",
                "color": "White Pearl Crystal Shine",
                "specifications": {
                    "features": [f"Feature {i}" for i in range(1, 21)]  # Many features
                }
            },
            "pricing": {
                "base_price": 4500000.00,
                "add_ons": [
                    {"name": f"Add-on {i}", "price": 10000.00}
                    for i in range(1, 11)  # Many add-ons
                ],
                "total_amount": 4700000.00
            },
            "employee": {
                "name": "Performance Test Employee",
                "email": "perf.emp@test.com",
                "phone": "+63 2 8000 3333"
            }
        }
        
        generator = QuotationPDFGenerator(template_dir="backend/templates")
        
        import time
        start_time = time.time()
        
        pdf_bytes = await generator.generate_quotation_pdf(sample_data)
        
        generation_time = time.time() - start_time
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        # Performance expectations (should generate within reasonable time)
        assert generation_time < 10.0, f"PDF generation took too long: {generation_time:.2f}s"
        
        print(f"PDF generation completed in {generation_time:.2f} seconds")
        print(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PDF_GENERATION_AVAILABLE, reason="WeasyPrint not available")
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="PDF templates not available")
    async def test_concurrent_pdf_generation(self):
        """Test concurrent PDF generation."""
        
        async def generate_pdf(quotation_id: str) -> bytes:
            data = {
                "quotation_number": f"QUO-CONCURRENT-{quotation_id}",
                "customer": {
                    "name": f"Customer {quotation_id}",
                    "email": f"customer{quotation_id}@example.com",
                    "phone": f"+63 912 000 {quotation_id.zfill(4)}",
                    "address": f"Address {quotation_id}"
                },
                "vehicle": {
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2024,
                    "type": "Sedan",
                    "color": "Silver"
                },
                "pricing": {
                    "base_price": 1500000.00,
                    "total_amount": 1600000.00
                },
                "employee": {
                    "name": f"Employee {quotation_id}",
                    "email": f"emp{quotation_id}@test.com",
                    "phone": "+63 2 8000 0000"
                }
            }
            
            generator = QuotationPDFGenerator(template_dir="backend/templates")
            return await generator.generate_quotation_pdf(data)
        
        # Generate 3 PDFs concurrently
        import time
        start_time = time.time()
        
        tasks = [generate_pdf(str(i)) for i in range(1, 4)]
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        # Validate all PDFs were generated
        assert len(results) == 3
        for pdf_bytes in results:
            assert pdf_bytes is not None
            assert len(pdf_bytes) > 0
            assert pdf_bytes.startswith(b'%PDF')
        
        print(f"Concurrent generation of 3 PDFs completed in {concurrent_time:.2f} seconds")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_quotation_pdf_quality.py -v
    pytest.main([__file__, "-v"])
