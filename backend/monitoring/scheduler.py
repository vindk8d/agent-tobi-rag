"""
Daily refresh scheduler for data sources using APScheduler.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from database import db_client
from rag.pipeline import DocumentProcessingPipeline
from scrapers.web_scraper import WebScraper
from models.datasource import DataSourceStatus, ScrapingFrequency
import logging
import asyncio

logger = logging.getLogger(__name__)

class DataSourceScheduler:
    """
    Scheduler for daily refresh of data sources with error handling.
    """
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.pipeline = DocumentProcessingPipeline()

    def start(self):
        self.scheduler.add_job(self.refresh_all_sources, 'interval', days=1, id='daily_refresh')
        self.scheduler.start()
        logger.info("Data source daily refresh scheduler started.")

    def stop(self):
        self.scheduler.shutdown()
        logger.info("Data source scheduler stopped.")

    def refresh_all_sources(self):
        logger.info("Starting daily refresh of data sources...")
        try:
            sources = db_client.client.table("data_sources").select("id, name, url, status, scraping_frequency").eq("status", DataSourceStatus.ACTIVE).eq("scraping_frequency", ScrapingFrequency.DAILY).execute()
            if not sources.data:
                logger.info("No active daily data sources found.")
                return
            for src in sources.data:
                url = src.get("url")
                data_source_id = src.get("id")
                data_source_name = src.get("name", "Unknown Source")
                if not url or not data_source_id:
                    continue
                try:
                    # Process using the fixed pipeline interface
                    asyncio.run(self.pipeline.process_url(url, data_source_name=data_source_name))
                    logger.info(f"Refreshed data source {data_source_id} ({data_source_name}: {url})")
                except Exception as e:
                    logger.error(f"Error refreshing data source {data_source_id}: {e}")
        except Exception as e:
            logger.error(f"Scheduler failed: {e}") 