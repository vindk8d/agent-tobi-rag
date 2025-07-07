"""
Configuration settings for the Salesperson Copilot RAG system.
Loads environment variables and provides configuration classes.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    
    class Config:
        env_prefix = "OPENAI_"


class SupabaseConfig(BaseSettings):
    """Supabase configuration"""
    url: str = Field(..., env="SUPABASE_URL")
    anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")
    
    class Config:
        env_prefix = "SUPABASE_"


class LangSmithConfig(BaseSettings):
    """LangSmith tracing configuration"""
    tracing_enabled: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    project: str = Field(default="salesperson-copilot-rag", env="LANGCHAIN_PROJECT")
    
    class Config:
        env_prefix = "LANGCHAIN_"


class RAGConfig(BaseSettings):
    """RAG system configuration"""
    chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    similarity_threshold: float = Field(default=0.8, env="RAG_SIMILARITY_THRESHOLD")
    max_retrieved_documents: int = Field(default=10, env="RAG_MAX_RETRIEVED_DOCUMENTS")
    embedding_batch_size: int = Field(default=100, env="RAG_EMBEDDING_BATCH_SIZE")
    
    class Config:
        env_prefix = "RAG_"


class ScrapingConfig(BaseSettings):
    """Web scraping configuration"""
    delay_seconds: float = Field(default=1.0, env="SCRAPING_DELAY_SECONDS")
    timeout_seconds: int = Field(default=30, env="SCRAPING_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, env="SCRAPING_MAX_RETRIES")
    user_agent: str = Field(default="SalespersonCopilot/1.0", env="SCRAPING_USER_AGENT")
    
    class Config:
        env_prefix = "SCRAPING_"


class TelegramConfig(BaseSettings):
    """Telegram bot configuration"""
    bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    webhook_url: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_URL")
    
    class Config:
        env_prefix = "TELEGRAM_"


class RedisConfig(BaseSettings):
    """Redis configuration for caching and queuing"""
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    class Config:
        env_prefix = "REDIS_"


class FastAPIConfig(BaseSettings):
    """FastAPI server configuration"""
    host: str = Field(default="0.0.0.0", env="FASTAPI_HOST")
    port: int = Field(default=8000, env="FASTAPI_PORT")
    debug: bool = Field(default=False, env="FASTAPI_DEBUG")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"], 
        env="FASTAPI_CORS_ORIGINS"
    )
    
    class Config:
        env_prefix = "FASTAPI_"


class NextJSConfig(BaseSettings):
    """Next.js frontend configuration"""
    api_url: str = Field(default="http://localhost:8000", env="NEXT_PUBLIC_API_URL")
    supabase_url: str = Field(..., env="NEXT_PUBLIC_SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="NEXT_PUBLIC_SUPABASE_ANON_KEY")
    nextauth_secret: Optional[str] = Field(default=None, env="NEXTAUTH_SECRET")
    nextauth_url: str = Field(default="http://localhost:3000", env="NEXTAUTH_URL")
    
    class Config:
        env_prefix = "NEXT_PUBLIC_"


class SystemConfig(BaseSettings):
    """General system configuration"""
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    response_timeout_seconds: int = Field(default=30, env="RESPONSE_TIMEOUT_SECONDS")


class Settings(BaseSettings):
    """Main settings class that combines all configurations"""
    
    # Individual config sections
    openai: OpenAIConfig = OpenAIConfig()
    supabase: SupabaseConfig = SupabaseConfig()
    langsmith: LangSmithConfig = LangSmithConfig()
    rag: RAGConfig = RAGConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    telegram: TelegramConfig = TelegramConfig()
    redis: RedisConfig = RedisConfig()
    fastapi: FastAPIConfig = FastAPIConfig()
    nextjs: NextJSConfig = NextJSConfig()
    system: SystemConfig = SystemConfig()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This function uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


def validate_openai_config() -> bool:
    """
    Validate OpenAI configuration and test API connection.
    Returns True if configuration is valid and API is accessible.
    """
    try:
        import openai
        settings = get_settings()
        
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


def validate_supabase_config() -> bool:
    """
    Validate Supabase configuration and test connection.
    Returns True if configuration is valid and database is accessible.
    """
    try:
        from supabase import create_client, Client
        settings = get_settings()
        
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


def setup_langsmith_tracing():
    """
    Setup LangSmith tracing if configured.
    """
    settings = get_settings()
    
    if settings.langsmith.tracing_enabled and settings.langsmith.api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith.tracing_enabled).lower()
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        
        print(f"✅ LangSmith tracing enabled for project: {settings.langsmith.project}")
    else:
        print("⚠️ LangSmith tracing not configured (API key missing)")


def validate_all_configs() -> bool:
    """
    Validate all configurations and return overall status.
    """
    print("🔧 Validating system configurations...")
    
    openai_ok = validate_openai_config()
    supabase_ok = validate_supabase_config()
    
    setup_langsmith_tracing()
    
    if openai_ok and supabase_ok:
        print("✅ All core configurations validated successfully!")
        return True
    else:
        print("❌ Some configurations failed validation")
        return False


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_all_configs() 