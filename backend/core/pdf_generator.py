"""
PDF Generation Module for Quotations
Uses WeasyPrint to generate professional PDF documents from HTML templates.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
import weasyprint
from jinja2 import Environment, FileSystemLoader, Template
import aiofiles

# Configure logging
logger = logging.getLogger(__name__)

class PDFGenerationError(Exception):
    """Custom exception for PDF generation errors"""
    pass

class QuotationPDFGenerator:
    """
    Professional PDF generator for vehicle quotations using WeasyPrint.
    Handles template rendering, styling, and PDF generation with comprehensive error handling.
    """
    
    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the PDF generator.
        
        Args:
            template_dir: Directory containing HTML templates and CSS files
        """
        self.template_dir = Path(template_dir)
        self.template_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters for template rendering
        self.template_env.filters['currency'] = self._format_currency
        self.template_env.filters['date'] = self._format_date
        self.template_env.filters['phone'] = self._format_phone
        
        # PDF generation settings
        self.pdf_options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '15mm',
            'margin-bottom': '20mm',
            'margin-left': '15mm',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None,
        }
    
    @staticmethod
    def _format_currency(value: float, currency: str = "₱") -> str:
        """Format currency values with proper formatting"""
        try:
            return f"{currency} {value:,.2f}"
        except (ValueError, TypeError):
            return f"{currency} 0.00"
    
    @staticmethod
    def _format_date(value: datetime, format_string: str = "%B %d, %Y") -> str:
        """Format datetime objects for display"""
        try:
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return value.strftime(format_string)
        except (ValueError, AttributeError):
            return datetime.now().strftime(format_string)
    
    @staticmethod
    def _format_phone(value: str) -> str:
        """Format phone numbers for display"""
        if not value:
            return "N/A"
        
        # Remove non-numeric characters
        digits = ''.join(filter(str.isdigit, value))
        
        # Format Philippine mobile numbers
        if len(digits) == 11 and digits.startswith('09'):
            return f"+63 {digits[1:4]} {digits[4:7]} {digits[7:]}"
        elif len(digits) == 10 and digits.startswith('9'):
            return f"+63 {digits[:3]} {digits[3:6]} {digits[6:]}"
        
        return value
    
    def _validate_quotation_data(self, data: Dict[str, Any]) -> None:
        """
        Validate required fields in quotation data.
        
        Args:
            data: Quotation data dictionary
            
        Raises:
            PDFGenerationError: If required fields are missing
        """
        required_fields = {
            'quotation_number': str,
            'customer': dict,
            'vehicle': dict,
            'pricing': dict,
            'employee': dict,
        }
        
        for field, expected_type in required_fields.items():
            if field not in data:
                raise PDFGenerationError(f"Missing required field: {field}")
            
            if not isinstance(data[field], expected_type):
                raise PDFGenerationError(f"Field '{field}' must be of type {expected_type.__name__}")
        
        # Validate nested required fields
        customer_fields = ['name', 'email']
        for field in customer_fields:
            if field not in data['customer'] or not data['customer'][field]:
                raise PDFGenerationError(f"Missing required customer field: {field}")
        
        vehicle_fields = ['make', 'model']
        for field in vehicle_fields:
            if field not in data['vehicle'] or not data['vehicle'][field]:
                raise PDFGenerationError(f"Missing required vehicle field: {field}")
        
        pricing_fields = ['base_price', 'total_amount']
        for field in pricing_fields:
            if field not in data['pricing']:
                raise PDFGenerationError(f"Missing required pricing field: {field}")
        
        employee_fields = ['name', 'email']
        for field in employee_fields:
            if field not in data['employee'] or not data['employee'][field]:
                raise PDFGenerationError(f"Missing required employee field: {field}")
    
    def _prepare_template_data(self, quotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and enrich data for template rendering.
        
        Args:
            quotation_data: Raw quotation data
            
        Returns:
            Enhanced data dictionary for template rendering
        """
        # Make a copy to avoid modifying original data
        data = quotation_data.copy()
        
        # Add default values and computed fields
        now = datetime.now()
        
        # Company information (should come from config/environment)
        data.setdefault('company_name', 'Premium Motors Philippines')
        data.setdefault('company_address', '123 EDSA, Makati City, Metro Manila, Philippines')
        data.setdefault('company_phone', '+63 2 8123 4567')
        data.setdefault('company_email', 'sales@premiummotors.ph')
        data.setdefault('company_website', 'www.premiummotors.ph')
        
        # Quotation metadata
        data.setdefault('quotation_date', now.strftime("%B %d, %Y"))
        data.setdefault('validity_days', 30)
        
        # Calculate expiry date
        expiry_date = now + timedelta(days=data['validity_days'])
        data.setdefault('expiry_date', expiry_date.strftime("%B %d, %Y"))
        
        # Default pricing values
        pricing = data['pricing']
        pricing.setdefault('insurance', 0.0)
        pricing.setdefault('lto_fees', 0.0)
        pricing.setdefault('discounts', 0.0)
        pricing.setdefault('add_ons', [])
        
        # Default delivery and payment terms
        data.setdefault('delivery_time', '4-6 weeks')
        data.setdefault('reservation_fee', 50000.0)
        
        # Ensure vehicle specifications exist
        if 'specifications' not in data['vehicle']:
            data['vehicle']['specifications'] = {}
        
        return data
    
    async def _render_html_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Render HTML template with provided data.
        
        Args:
            template_name: Name of the template file
            data: Data to render in the template
            
        Returns:
            Rendered HTML string
            
        Raises:
            PDFGenerationError: If template rendering fails
        """
        try:
            template = self.template_env.get_template(template_name)
            html_content = template.render(**data)
            return html_content
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise PDFGenerationError(f"Template rendering failed: {str(e)}")
    
    async def _generate_pdf_bytes(self, html_content: str, css_path: Optional[str] = None) -> bytes:
        """
        Generate PDF bytes from HTML content using WeasyPrint.
        
        Args:
            html_content: Rendered HTML content
            css_path: Optional path to additional CSS file
            
        Returns:
            PDF content as bytes
            
        Raises:
            PDFGenerationError: If PDF generation fails
        """
        try:
            # Run PDF generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate_pdf():
                # Create HTML document
                html_doc = weasyprint.HTML(string=html_content, base_url=str(self.template_dir))
                
                # Load CSS stylesheets
                css_stylesheets = []
                if css_path and Path(css_path).exists():
                    css_stylesheets.append(weasyprint.CSS(filename=css_path))
                
                # Generate PDF
                pdf_bytes = html_doc.write_pdf(stylesheets=css_stylesheets)
                return pdf_bytes
            
            pdf_bytes = await loop.run_in_executor(None, _generate_pdf)
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise PDFGenerationError(f"PDF generation failed: {str(e)}")
    
    async def generate_quotation_pdf(
        self, 
        quotation_data: Dict[str, Any],
        template_name: str = "quotation_template.html",
        css_file: str = "quotation_styles.css"
    ) -> bytes:
        """
        Generate a complete quotation PDF from data.
        
        Args:
            quotation_data: Complete quotation data
            template_name: HTML template file name
            css_file: CSS stylesheet file name
            
        Returns:
            PDF content as bytes
            
        Raises:
            PDFGenerationError: If generation fails at any step
        """
        try:
            logger.info(f"Starting PDF generation for quotation {quotation_data.get('quotation_number', 'unknown')}")
            
            # Validate input data
            self._validate_quotation_data(quotation_data)
            
            # Prepare template data
            template_data = self._prepare_template_data(quotation_data)
            
            # Render HTML template
            html_content = await self._render_html_template(template_name, template_data)
            
            # Prepare CSS path
            css_path = self.template_dir / css_file if css_file else None
            
            # Generate PDF
            pdf_bytes = await self._generate_pdf_bytes(html_content, css_path)
            
            logger.info(f"Successfully generated PDF ({len(pdf_bytes)} bytes) for quotation {quotation_data['quotation_number']}")
            return pdf_bytes
            
        except PDFGenerationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PDF generation: {e}")
            raise PDFGenerationError(f"Unexpected error: {str(e)}")
    
    async def generate_and_save_pdf(
        self, 
        quotation_data: Dict[str, Any],
        output_path: str,
        template_name: str = "quotation_template.html",
        css_file: str = "quotation_styles.css"
    ) -> Tuple[str, int]:
        """
        Generate PDF and save to file system.
        
        Args:
            quotation_data: Complete quotation data
            output_path: Path where PDF should be saved
            template_name: HTML template file name
            css_file: CSS stylesheet file name
            
        Returns:
            Tuple of (file_path, file_size_bytes)
            
        Raises:
            PDFGenerationError: If generation or saving fails
        """
        try:
            # Generate PDF bytes
            pdf_bytes = await self.generate_quotation_pdf(quotation_data, template_name, css_file)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save PDF to file
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(pdf_bytes)
            
            file_size = len(pdf_bytes)
            logger.info(f"PDF saved to {output_path} ({file_size} bytes)")
            
            return str(output_path), file_size
            
        except Exception as e:
            logger.error(f"Failed to save PDF to {output_path}: {e}")
            raise PDFGenerationError(f"Failed to save PDF: {str(e)}")
    
    async def preview_html(self, quotation_data: Dict[str, Any]) -> str:
        """
        Generate HTML preview of quotation (for testing/debugging).
        
        Args:
            quotation_data: Complete quotation data
            
        Returns:
            Rendered HTML content
        """
        try:
            self._validate_quotation_data(quotation_data)
            template_data = self._prepare_template_data(quotation_data)
            html_content = await self._render_html_template("quotation_template.html", template_data)
            return html_content
        except Exception as e:
            logger.error(f"HTML preview generation failed: {e}")
            raise PDFGenerationError(f"HTML preview failed: {str(e)}")

# Convenience functions for common use cases
async def generate_quotation_pdf(quotation_data: Dict[str, Any]) -> bytes:
    """
    Convenience function to generate a quotation PDF.
    
    Args:
        quotation_data: Complete quotation data
        
    Returns:
        PDF content as bytes
    """
    generator = QuotationPDFGenerator()
    return await generator.generate_quotation_pdf(quotation_data)

async def save_quotation_pdf(quotation_data: Dict[str, Any], output_path: str) -> Tuple[str, int]:
    """
    Convenience function to generate and save a quotation PDF.
    
    Args:
        quotation_data: Complete quotation data
        output_path: Path where PDF should be saved
        
    Returns:
        Tuple of (file_path, file_size_bytes)
    """
    generator = QuotationPDFGenerator()
    return await generator.generate_and_save_pdf(quotation_data, output_path)