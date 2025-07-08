"""
Web scraping utilities for static (BeautifulSoup4) and dynamic (Playwright) content.
"""

from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Web scraper supporting both static and dynamic content.
    Uses requests + BeautifulSoup4 for static pages, Playwright for dynamic pages.
    """
    @staticmethod
    def scrape_static(url: str, timeout: int = 10) -> Optional[str]:
        """Scrape static HTML content using requests and BeautifulSoup4."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.warning(f"Static scrape failed for {url}: {e}")
            return None

    @staticmethod
    def scrape_dynamic(url: str, timeout: int = 20) -> Optional[str]:
        """Scrape dynamic content using Playwright (Chromium)."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=timeout * 1000)
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.warning(f"Dynamic scrape failed for {url}: {e}")
            return None

    @staticmethod
    def scrape(url: str, timeout: int = 10, dynamic_fallback: bool = True) -> Dict[str, Any]:
        """
        Scrape a URL, using static method first, then Playwright if needed.
        Returns a dict with 'text', 'method', and 'success'.
        """
        text = WebScraper.scrape_static(url, timeout=timeout)
        if text:
            return {"text": text, "method": "static", "success": True}
        if dynamic_fallback:
            text = WebScraper.scrape_dynamic(url, timeout=timeout * 2)
            if text:
                return {"text": text, "method": "dynamic", "success": True}
        return {"text": None, "method": None, "success": False} 