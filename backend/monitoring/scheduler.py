"""
Daily refresh scheduler and routine maintenance jobs using APScheduler.
Includes data source refresh, quotation PDF cleanup, and vehicle specification cleanup.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from core.database import db_client
from rag.pipeline import DocumentProcessingPipeline
from scrapers.web_scraper import WebScraper
from models.datasource import DataSourceStatus, ScrapingFrequency
import logging
import asyncio
from .quotation_cleanup import cleanup_expired_quotations
from .vehicle_cleanup import run_vehicle_cleanup_sync, run_vehicle_orphan_cleanup_sync

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
        # Run quotation cleanup twice daily by default
        self.scheduler.add_job(self.run_quotation_cleanup, 'interval', hours=12, id='quotation_cleanup')
        # Run vehicle backup cleanup daily (following quotation patterns)
        self.scheduler.add_job(self.run_vehicle_cleanup, 'interval', days=1, id='vehicle_cleanup')
        # Run vehicle orphan cleanup weekly (less frequent, more thorough)
        self.scheduler.add_job(self.run_vehicle_orphan_cleanup, 'interval', days=7, id='vehicle_orphan_cleanup')
        self.scheduler.start()
        logger.info("Data source daily refresh and cleanup schedulers started.")

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

    def run_quotation_cleanup(self):
        logger.info("Running quotation PDF cleanup job...")
        try:
            summary = cleanup_expired_quotations()
            logger.info(
                "Quotation cleanup summary: selected=%s, deleted=%s, failed=%s",
                summary.get("selected"), summary.get("deleted"), summary.get("failed")
            )
        except Exception as e:
            logger.error(f"Quotation cleanup job failed: {e}")

    def run_vehicle_cleanup(self):
        """Run vehicle specification backup cleanup job following quotation patterns."""
        logger.info("Running vehicle specification backup cleanup job...")
        try:
            # Clean up backups older than 30 days (configurable)
            summary = run_vehicle_cleanup_sync(days_old=30, max_delete=200)
            logger.info(
                "Vehicle cleanup summary: selected=%s, deleted=%s, failed=%s, status=%s",
                summary.get("selected"), summary.get("deleted"), summary.get("failed"), summary.get("status")
            )
            
            # Log any errors
            if summary.get("errors"):
                for error in summary["errors"]:
                    logger.warning(f"Vehicle cleanup error: {error}")
                    
        except Exception as e:
            logger.error(f"Vehicle cleanup job failed: {e}")

    def run_vehicle_orphan_cleanup(self):
        """Run vehicle orphaned document cleanup job following quotation patterns."""
        logger.info("Running vehicle orphaned document cleanup job...")
        try:
            summary = run_vehicle_orphan_cleanup_sync(max_delete=100)
            logger.info(
                "Vehicle orphan cleanup summary: selected=%s, deleted=%s, failed=%s, status=%s",
                summary.get("selected"), summary.get("deleted"), summary.get("failed"), summary.get("status")
            )
            
            # Log any errors
            if summary.get("errors"):
                for error in summary["errors"]:
                    logger.warning(f"Vehicle orphan cleanup error: {error}")
                    
        except Exception as e:
            logger.error(f"Vehicle orphan cleanup job failed: {e}")
