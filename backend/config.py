"""
Configuration settings for the Salesperson Copilot RAG system.
Loads environment variables and provides configuration classes.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
import asyncio
from pathlib import Path

# Manual .env file loading to completely avoid os.getcwd() calls
try:
    # Use pathlib to get the .env file path without triggering os.getcwd()
    # Using absolute() instead of resolve() to avoid blocking calls
    CURRENT_DIR = Path(__file__).absolute().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    ENV_FILE_PATH = PROJECT_ROOT / ".env"
    
    # Load .env file manually line by line to avoid any dotenv os.getcwd() calls
    if ENV_FILE_PATH.exists():
        with open(ENV_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")  # Remove quotes
                    if key and not key in os.environ:  # Don't override existing env vars
                        os.environ[key] = value
                        
except Exception as e:
    # If .env loading fails, continue with just os.environ
    pass

# Async settings cache to prevent repeated blocking calls
_settings_cache = None
_settings_lock = asyncio.Lock()


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    max_tokens: int = Field(default=1500, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")

    # Context Window Management
    max_result_size_simple_model: int = Field(default=50000, description="Max result size for simple models (bytes)")
    force_complex_model_size: int = Field(default=20000, description="Size threshold to force complex model (bytes)")
    max_display_length: int = Field(default=10000, description="Max length for truncated display (bytes)")


class LangSmithConfig(BaseSettings):
    """LangSmith tracing configuration"""
    model_config = SettingsConfigDict(
        env_prefix="LANGCHAIN_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    tracing_enabled: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    project: str = Field(default="salesperson-copilot-rag", env="LANGCHAIN_PROJECT")


class RAGConfig(BaseSettings):
    """RAG system configuration"""
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    similarity_threshold: float = Field(default=0.5, env="RAG_SIMILARITY_THRESHOLD")
    max_retrieved_documents: int = Field(default=10, env="RAG_MAX_RETRIEVED_DOCUMENTS")
    embedding_batch_size: int = Field(default=100, env="RAG_EMBEDDING_BATCH_SIZE")


class TelegramConfig(BaseSettings):
    """Telegram bot configuration"""
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    webhook_url: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_URL")


class RedisConfig(BaseSettings):
    """Redis configuration for caching and queuing"""
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")


class FastAPIConfig(BaseSettings):
    """FastAPI server configuration"""
    model_config = SettingsConfigDict(
        env_prefix="FASTAPI_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    host: str = Field(default="0.0.0.0", env="FASTAPI_HOST")
    port: int = Field(default=8000, env="FASTAPI_PORT")
    debug: bool = Field(default=False, env="FASTAPI_DEBUG")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"], 
        env="FASTAPI_CORS_ORIGINS"
    )


class NextJSConfig(BaseSettings):
    """Next.js frontend configuration"""
    model_config = SettingsConfigDict(
        env_prefix="NEXT_PUBLIC_",
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    api_url: str = Field(default="http://localhost:8000", env="NEXT_PUBLIC_API_URL")
    supabase_url: str = Field(..., env="NEXT_PUBLIC_SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="NEXT_PUBLIC_SUPABASE_ANON_KEY")
    nextauth_secret: Optional[str] = Field(default=None, env="NEXTAUTH_SECRET")
    nextauth_url: str = Field(default="http://localhost:3000", env="NEXTAUTH_URL")


class SystemConfig(BaseSettings):
    """General system configuration"""
    model_config = SettingsConfigDict(
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    response_timeout_seconds: int = Field(default=30, env="RESPONSE_TIMEOUT_SECONDS")


class Settings(BaseSettings):
    """Main settings class that loads all configurations directly"""
    
    # Use new pydantic v2 configuration - REMOVED env_file completely
    model_config = SettingsConfigDict(
        # REMOVED: env_file to prevent automatic .env discovery and os.getcwd() calls
        case_sensitive=False,
        extra='allow'
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field(default="gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    openai_simple_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_SIMPLE_MODEL")
    openai_complex_model: str = Field(default="gpt-4", env="OPENAI_COMPLEX_MODEL")
    openai_max_tokens: int = Field(default=1500, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")
    supabase_db_password: Optional[str] = Field(default=None, env="SUPABASE_DB_PASSWORD")
    
    # LangSmith Configuration
    langsmith_tracing_enabled: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="salesperson-copilot-rag", env="LANGCHAIN_PROJECT")
    
    # RAG Configuration
    rag_chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    rag_similarity_threshold: float = Field(default=0.5, env="RAG_SIMILARITY_THRESHOLD")
    rag_max_retrieved_documents: int = Field(default=10, env="RAG_MAX_RETRIEVED_DOCUMENTS")
    rag_embedding_batch_size: int = Field(default=100, env="RAG_EMBEDDING_BATCH_SIZE")
    
    # FastAPI Configuration
    fastapi_host: str = Field(default="0.0.0.0", env="FASTAPI_HOST")
    fastapi_port: int = Field(default=8000, env="FASTAPI_PORT")
    fastapi_debug: bool = Field(default=False, env="FASTAPI_DEBUG")
    
    # System Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Memory management configuration
    memory_max_messages: int = Field(default=12, env="MEMORY_MAX_MESSAGES")
    memory_summary_interval: int = Field(default=10, env="MEMORY_SUMMARY_INTERVAL")
    memory_auto_summarize: bool = Field(default=True, env="MEMORY_AUTO_SUMMARIZE")
    # Removed master_summary_conversation_limit - system simplified to use conversation summaries only

    # Telegram Configuration
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    
    # Convenience properties to maintain compatibility
    @property
    def openai(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            api_key=self.openai_api_key,
            embedding_model=self.openai_embedding_model,
            chat_model=self.openai_chat_model,
            simple_model=self.openai_simple_model,
            complex_model=self.openai_complex_model,
            max_tokens=self.openai_max_tokens,
            temperature=self.openai_temperature
        )
    
    @property 
    def supabase(self):
        from types import SimpleNamespace
        
        # Generate PostgreSQL connection string for LangGraph persistence
        def get_postgresql_connection_string():
            if not self.supabase_db_password:
                raise ValueError("SUPABASE_DB_PASSWORD is required for PostgreSQL connection")
            
            # Extract project ID from Supabase URL
            # URL format: https://your-project-id.supabase.co
            import re
            match = re.match(r'https://([^.]+)\.supabase\.co', self.supabase_url)
            if not match:
                raise ValueError("Invalid Supabase URL format")
            
            project_id = match.group(1)
            
            # Use pooler connection format for better compatibility (supports both IPv4 and IPv6)
            # Format: postgres://postgres.{project_id}:{password}@aws-0-{region}.pooler.supabase.com:5432/postgres
            # Using ap-southeast-1 as the correct region for this project
            return f"postgres://postgres.{project_id}:{self.supabase_db_password}@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
        
        return SimpleNamespace(
            url=self.supabase_url,
            anon_key=self.supabase_anon_key,
            service_key=self.supabase_service_key,
            postgresql_connection_string=get_postgresql_connection_string()
        )
    
    @property
    def rag(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            chunk_size=self.rag_chunk_size,
            chunk_overlap=self.rag_chunk_overlap,
            similarity_threshold=self.rag_similarity_threshold,
            max_retrieved_documents=self.rag_max_retrieved_documents,
            embedding_batch_size=self.rag_embedding_batch_size
        )
    
    @property
    def langsmith(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            tracing_enabled=self.langsmith_tracing_enabled,
            endpoint=self.langsmith_endpoint,
            api_key=self.langsmith_api_key,
            project=self.langsmith_project
        )

async def get_settings() -> Settings:
    """
    Get cached settings instance asynchronously.
    This function prevents blocking calls by using async caching.
    """
    global _settings_cache
    
    if _settings_cache is not None:
        return _settings_cache
    
    async with _settings_lock:
        if _settings_cache is not None:
            return _settings_cache
        
        # Initialize settings in a way that doesn't block by using asyncio.to_thread
        try:
            _settings_cache = await asyncio.to_thread(Settings)
            return _settings_cache
        except Exception as e:
            # If initialization fails, create a minimal settings object
            # This prevents blocking calls from causing complete failure
            raise RuntimeError(f"Failed to initialize settings: {e}")


@lru_cache()
def get_settings_sync() -> Settings:
    """
    Get cached settings instance synchronously (for non-async contexts).
    This function uses lru_cache to ensure settings are loaded only once.
    WARNING: This may cause blocking calls in ASGI environments.
    """
    return Settings()


async def validate_openai_config() -> bool:
    """
    Validate OpenAI configuration and test API connection.
    Returns True if configuration is valid and API is accessible.
    """
    try:
        import openai
        settings = await get_settings()
        
        # Set the API key
        openai.api_key = settings.openai.api_key
        
        # Test API connection with a simple call
        client = openai.OpenAI(api_key=settings.openai.api_key)
        
        # Test embedding model
        response = client.embeddings.create(
            model=settings.openai.embedding_model,
            input="Test embedding"
        )
        
        print(f"✅ OpenAI API connection successful")
        print(f"✅ Embedding model '{settings.openai.embedding_model}' is accessible")
        print(f"✅ Embedding dimensions: {len(response.data[0].embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI configuration error: {str(e)}")
        return False


async def validate_supabase_config() -> bool:
    """
    Validate Supabase configuration and test connection.
    Returns True if configuration is valid and database is accessible.
    """
    try:
        from supabase import create_client, Client
        settings = await get_settings()
        
        # Create Supabase client
        supabase: Client = create_client(
            settings.supabase.url,
            settings.supabase.service_key
        )
        
        # Test connection with a simple query
        result = supabase.table("data_sources").select("count", count="exact").execute()
        
        print(f"✅ Supabase connection successful")
        print(f"✅ Database accessible at {settings.supabase.url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Supabase configuration error: {str(e)}")
        return False


async def setup_langsmith_tracing():
    """
    Setup LangSmith tracing if configured.
    """
    settings = await get_settings()
    
    if settings.langsmith.tracing_enabled and settings.langsmith.api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith.tracing_enabled).lower()
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        
        print(f"✅ LangSmith tracing enabled for project: {settings.langsmith.project}")
    else:
        print("⚠️ LangSmith tracing not configured (API key missing)")


async def validate_all_configs() -> bool:
    """
    Validate all configurations and return overall status.
    """
    print("🔧 Validating system configurations...")
    
    openai_ok = await validate_openai_config()
    supabase_ok = await validate_supabase_config()
    
    await setup_langsmith_tracing()
    
    if openai_ok and supabase_ok:
        print("✅ All core configurations validated successfully!")
        return True
    else:
        print("❌ Some configurations failed validation")
        return False


if __name__ == "__main__":
    # Run validation when script is executed directly
    import asyncio
    asyncio.run(validate_all_configs())

# ============================================================================
# DEPRIORITIZED CODE - Website scraping functionality has been deprioritized
# ============================================================================

# DEPRIORITIZED: Website scraping functionality has been deprioritized
# class ScrapingConfig(BaseSettings):
#     """Web scraping configuration"""
#     delay_seconds: float = Field(default=1.0, env="SCRAPING_DELAY_SECONDS")
#     timeout_seconds: int = Field(default=30, env="SCRAPING_TIMEOUT_SECONDS")
#     max_retries: int = Field(default=3, env="SCRAPING_MAX_RETRIES")
#     user_agent: str = Field(default="SalespersonCopilot/1.0", env="SCRAPING_USER_AGENT")
    
#     class Config:
#         env_prefix = "SCRAPING_" 