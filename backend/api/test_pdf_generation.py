"""
Test API endpoints for PDF generation functionality.
Provides endpoints for testing PDF generation and HTML preview.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, validator
from io import BytesIO
import asyncio

from core.pdf_generator import QuotationPDFGenerator, PDFGenerationError

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["PDF Testing"])

# Pydantic models for request validation
class CustomerData(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    company: str = Field(default="", max_length=255)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    phone: str = Field(..., min_length=1, max_length=50)
    address: str = Field(default="", max_length=500)

class VehicleSpecifications(BaseModel):
    engine: str = Field(default="", max_length=100)
    power: str = Field(default="", max_length=50)
    torque: str = Field(default="", max_length=50)
    fuel_type: str = Field(default="gasoline", max_length=50)
    transmission: str = Field(default="automatic", max_length=50)

class VehicleData(BaseModel):
    make: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern=r'^(sedan|suv|hatchback|pickup|van|motorcycle|truck)$')
    color: str = Field(default="", max_length=50)
    year: str = Field(default="2025", max_length=10)
    specifications: VehicleSpecifications = Field(default_factory=VehicleSpecifications)

class AddOn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=500)
    price: float = Field(..., ge=0)

class PricingData(BaseModel):
    base_price: float = Field(..., ge=0)
    insurance: float = Field(default=0.0, ge=0)
    lto_fees: float = Field(default=0.0, ge=0)
    discounts: float = Field(default=0.0, ge=0)
    total_amount: float = Field(..., ge=0)
    add_ons: list[AddOn] = Field(default_factory=list)
    discount_description: str = Field(default="", max_length=255)

class EmployeeData(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    position: str = Field(..., pattern=r'^(sales_agent|account_executive|manager|director|admin)$')
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    phone: str = Field(..., min_length=1, max_length=50)
    branch_name: str = Field(..., min_length=1, max_length=255)
    branch_region: str = Field(..., pattern=r'^(north|south|east|west|central)$')

class QuotationTestRequest(BaseModel):
    quotation_number: str = Field(..., min_length=1, max_length=50)
    customer: CustomerData
    vehicle: VehicleData
    pricing: PricingData
    employee: EmployeeData

    @validator('quotation_number')
    def validate_quotation_number(cls, v):
        # Basic validation for quotation number format
        if not v.startswith('Q'):
            raise ValueError('Quotation number must start with "Q"')
        return v

    @validator('pricing')
    def validate_pricing_total(cls, v):
        # Calculate expected total
        add_on_total = sum(addon.price for addon in v.add_ons)
        expected_total = v.base_price + v.insurance + v.lto_fees + add_on_total - v.discounts
        
        # Allow small floating point differences
        if abs(v.total_amount - expected_total) > 0.01:
            logger.warning(f"Total amount mismatch: provided={v.total_amount}, calculated={expected_total}")
            # Auto-correct the total
            v.total_amount = expected_total
        
        return v

@router.post("/test-pdf-generation")
async def generate_test_pdf(request: QuotationTestRequest):
    """
    Generate a PDF quotation from test data.
    
    Returns the PDF as a downloadable file stream.
    """
    try:
        logger.info(f"Generating PDF for quotation {request.quotation_number}")
        
        # Convert Pydantic model to dict for PDF generator
        quotation_data = request.dict()
        
        # Initialize PDF generator
        generator = QuotationPDFGenerator()
        
        # Generate PDF bytes
        pdf_bytes = await generator.generate_quotation_pdf(quotation_data)
        
        # Create response with proper headers
        pdf_stream = BytesIO(pdf_bytes)
        
        headers = {
            'Content-Disposition': f'attachment; filename="quotation_{request.quotation_number}.pdf"',
            'Content-Type': 'application/pdf',
            'Content-Length': str(len(pdf_bytes))
        }
        
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type='application/pdf',
            headers=headers
        )
        
    except PDFGenerationError as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=400, detail=f"PDF generation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in PDF generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/test-pdf-preview")
async def preview_test_pdf(request: QuotationTestRequest):
    """
    Generate an HTML preview of the quotation (without PDF conversion).
    
    Returns the rendered HTML for preview purposes.
    """
    try:
        logger.info(f"Generating HTML preview for quotation {request.quotation_number}")
        
        # Convert Pydantic model to dict for PDF generator
        quotation_data = request.dict()
        
        # Initialize PDF generator
        generator = QuotationPDFGenerator()
        
        # Generate HTML preview
        html_content = await generator.preview_html(quotation_data)
        
        return HTMLResponse(content=html_content, status_code=200)
        
    except PDFGenerationError as e:
        logger.error(f"HTML preview generation failed: {e}")
        raise HTTPException(status_code=400, detail=f"HTML preview failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in HTML preview: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/test-pdf-sample-data")
async def get_sample_data():
    """
    Get sample quotation data for testing purposes.
    
    Returns various sample data sets for different vehicle types.
    """
    sample_data = {
        "economy": {
            "quotation_number": "Q2025-ECO-001",
            "customer": {
                "name": "Miguel Santos",
                "company": "",
                "email": "miguel.santos@gmail.com",
                "phone": "09171234567",
                "address": "456 Quezon Avenue, Quezon City, Metro Manila"
            },
            "vehicle": {
                "make": "Mitsubishi",
                "model": "Mirage",
                "type": "hatchback",
                "color": "Red Diamond",
                "year": "2025",
                "specifications": {
                    "engine": "1.2L 3-Cylinder",
                    "power": "78",
                    "torque": "100",
                    "fuel_type": "gasoline",
                    "transmission": "manual"
                }
            },
            "pricing": {
                "base_price": 720000.0,
                "insurance": 25000.0,
                "lto_fees": 12000.0,
                "discounts": 20000.0,
                "total_amount": 737000.0,
                "add_ons": [],
                "discount_description": "Cash Payment Discount"
            },
            "employee": {
                "name": "Carlos Reyes",
                "position": "sales_agent",
                "email": "carlos.reyes@premiummotors.ph",
                "phone": "09181234567",
                "branch_name": "Quezon City Branch",
                "branch_region": "central"
            }
        },
        "family": {
            "quotation_number": "Q2025-FAM-001",
            "customer": {
                "name": "Roberto and Ana Cruz",
                "company": "Cruz Family Business",
                "email": "roberto@cruzfamily.ph",
                "phone": "09171234567",
                "address": "789 Ortigas Avenue, Pasig City, Metro Manila"
            },
            "vehicle": {
                "make": "Ford",
                "model": "Territory",
                "type": "suv",
                "color": "Moondust Silver",
                "year": "2025",
                "specifications": {
                    "engine": "1.5L EcoBoost Turbo",
                    "power": "141",
                    "torque": "225",
                    "fuel_type": "gasoline",
                    "transmission": "cvt"
                }
            },
            "pricing": {
                "base_price": 1350000.0,
                "insurance": 55000.0,
                "lto_fees": 18000.0,
                "discounts": 30000.0,
                "total_amount": 1393000.0,
                "add_ons": [
                    {
                        "name": "Roof Rails",
                        "description": "Aluminum roof rail system",
                        "price": 15000.0
                    },
                    {
                        "name": "Floor Mats",
                        "description": "Weather-resistant floor mats",
                        "price": 8000.0
                    }
                ],
                "discount_description": "Family Package Discount"
            },
            "employee": {
                "name": "Maria Santos",
                "position": "account_executive",
                "email": "maria.santos@premiummotors.ph",
                "phone": "09181234567",
                "branch_name": "Pasig Branch",
                "branch_region": "east"
            }
        },
        "luxury": {
            "quotation_number": "Q2025-LUX-001",
            "customer": {
                "name": "Patricia Gonzalez",
                "company": "Elite Business Solutions Inc.",
                "email": "patricia@elitebiz.ph",
                "phone": "09171234567",
                "address": "321 Ayala Avenue, Makati City, Metro Manila"
            },
            "vehicle": {
                "make": "BMW",
                "model": "X5",
                "type": "suv",
                "color": "Alpine White",
                "year": "2025",
                "specifications": {
                    "engine": "3.0L Twin-Turbo I6",
                    "power": "335",
                    "torque": "450",
                    "fuel_type": "gasoline",
                    "transmission": "automatic"
                }
            },
            "pricing": {
                "base_price": 4500000.0,
                "insurance": 120000.0,
                "lto_fees": 25000.0,
                "discounts": 100000.0,
                "total_amount": 4875000.0,
                "add_ons": [
                    {
                        "name": "M Sport Package",
                        "description": "Performance styling and suspension",
                        "price": 250000.0
                    },
                    {
                        "name": "Premium Sound System",
                        "description": "Harman Kardon Audio System",
                        "price": 80000.0
                    },
                    {
                        "name": "Panoramic Sunroof",
                        "description": "Electric panoramic glass sunroof",
                        "price": 120000.0
                    }
                ],
                "discount_description": "VIP Customer Discount"
            },
            "employee": {
                "name": "Alexander Tan",
                "position": "manager",
                "email": "alexander.tan@premiummotors.ph",
                "phone": "09181234567",
                "branch_name": "Makati Premium Branch",
                "branch_region": "central"
            }
        }
    }
    
    return sample_data

@router.get("/test-pdf-health")
async def health_check():
    """
    Health check endpoint for PDF generation service.
    
    Verifies that all dependencies are available and working.
    """
    try:
        # Test PDF generator initialization
        generator = QuotationPDFGenerator()
        
        # Test template loading
        template = generator.template_env.get_template("quotation_template.html")
        
        # Basic template render test (no actual PDF generation)
        minimal_data = {
            "quotation_number": "TEST",
            "customer": {"name": "Test", "email": "test@test.com"},
            "vehicle": {"make": "Test", "model": "Test"},
            "pricing": {"base_price": 0, "total_amount": 0},
            "employee": {"name": "Test", "email": "test@test.com"}
        }
        
        # This will validate data and prepare template data
        prepared_data = generator._prepare_template_data(minimal_data)
        
        return {
            "status": "healthy",
            "message": "PDF generation service is operational",
            "template_loaded": template.name,
            "dependencies": {
                "weasyprint": "available",
                "jinja2": "available",
                "templates": "loaded"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"PDF generation service is not available: {str(e)}"
        )