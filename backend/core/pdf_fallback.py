"""
Fallback PDF generation using reportlab if WeasyPrint fails.
This ensures PDF functionality works even without system dependencies.
"""

import logging
from typing import Optional, Dict, Any
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

# Try to import WeasyPrint, but don't fail if it's not available
WEASYPRINT_AVAILABLE = False
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
    logger.info("WeasyPrint successfully imported")
except (ImportError, OSError) as e:
    logger.warning(f"WeasyPrint not available: {e}")
    logger.info("Using fallback PDF generation with reportlab")

# Import reportlab as fallback
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("Reportlab not available either - PDF generation will fail")


class FallbackPDFGenerator:
    """Fallback PDF generator using reportlab."""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("Neither WeasyPrint nor reportlab are available for PDF generation")
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4a4a4a'),
            spaceAfter=20,
            alignment=TA_LEFT
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12
        ))
    
    def generate_quotation_pdf(self, data: Dict[str, Any]) -> bytes:
        """Generate a quotation PDF using reportlab."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Add title
        story.append(Paragraph("QUOTATION", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Add customer info
        if 'customer' in data:
            story.append(Paragraph("Customer Information", self.styles['CustomSubtitle']))
            customer = data['customer']
            customer_info = f"""
            <b>Name:</b> {customer.get('name', 'N/A')}<br/>
            <b>Email:</b> {customer.get('email', 'N/A')}<br/>
            <b>Phone:</b> {customer.get('phone', 'N/A')}<br/>
            <b>Address:</b> {customer.get('address', 'N/A')}
            """
            story.append(Paragraph(customer_info, self.styles['CustomNormal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add items table
        if 'items' in data and data['items']:
            story.append(Paragraph("Quotation Items", self.styles['CustomSubtitle']))
            
            # Prepare table data
            table_data = [['Item', 'Description', 'Quantity', 'Price', 'Total']]
            total_amount = 0
            
            for item in data['items']:
                quantity = item.get('quantity', 1)
                price = item.get('price', 0)
                total = quantity * price
                total_amount += total
                
                table_data.append([
                    item.get('name', 'N/A'),
                    item.get('description', ''),
                    str(quantity),
                    f"${price:,.2f}",
                    f"${total:,.2f}"
                ])
            
            # Add total row
            table_data.append(['', '', '', 'TOTAL:', f"${total_amount:,.2f}"])
            
            # Create table
            table = Table(table_data, colWidths=[1.5*inch, 2.5*inch, 0.8*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (-2, -1), (-1, -1), 'Helvetica-Bold'),
                ('BACKGROUND', (-2, -1), (-1, -1), colors.lightgrey),
            ]))
            story.append(table)
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def generate_pdf_from_html(self, html_content: str) -> bytes:
        """Generate PDF from HTML content using reportlab (simplified)."""
        # For complex HTML, we would need to parse it
        # This is a simplified version that extracts text
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Simple HTML stripping (in production, use proper HTML parser)
        import re
        text = re.sub('<[^<]+?>', '', html_content)
        
        # Add content
        for line in text.split('\n'):
            if line.strip():
                story.append(Paragraph(line, self.styles['CustomNormal']))
                story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


def generate_pdf(html_content: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Universal PDF generation function that uses WeasyPrint if available,
    otherwise falls back to reportlab.
    """
    if WEASYPRINT_AVAILABLE:
        try:
            if html_content:
                return weasyprint.HTML(string=html_content).write_pdf()
            else:
                # Generate HTML from data and use WeasyPrint
                # This would need proper HTML template generation
                raise NotImplementedError("Data to HTML conversion needed")
        except Exception as e:
            logger.error(f"WeasyPrint failed, trying fallback: {e}")
    
    # Use fallback
    if not REPORTLAB_AVAILABLE:
        raise ImportError("No PDF generation library available")
    
    generator = FallbackPDFGenerator()
    
    if html_content:
        return generator.generate_pdf_from_html(html_content)
    elif data:
        return generator.generate_quotation_pdf(data)
    else:
        raise ValueError("Either html_content or data must be provided")
