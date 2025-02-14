from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from MAX.utils.logger import Logger

logger = Logger.get_logger()

# Get the project root directory (python directory)
ROOT_DIR = Path(__file__).parent.parent.parent.parent


class Settings(BaseSettings):
    # X API Settings
    X_BEARER_TOKEN: str
    X_API_KEY: str
    X_API_SECRET: str
    X_ACCESS_TOKEN: Optional[str] = None
    X_ACCESS_TOKEN_SECRET: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set environment variables after loading
        os.environ["X_BEARER_TOKEN"] = self.X_BEARER_TOKEN
        os.environ["X_API_KEY"] = self.X_API_KEY
        os.environ["X_API_SECRET"] = self.X_API_SECRET
        if self.X_ACCESS_TOKEN:
            os.environ["X_ACCESS_TOKEN"] = self.X_ACCESS_TOKEN
        if self.X_ACCESS_TOKEN_SECRET:
            os.environ["X_ACCESS_TOKEN_SECRET"] = self.X_ACCESS_TOKEN_SECRET

        # Debug logging
        logger.info("Settings initialized")
        logger.info(f"ENV file path: {ROOT_DIR / '.env'}")
        logger.info(f"ENV file exists: {(ROOT_DIR / '.env').exists()}")

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Create a settings instance
try:
    settings = Settings(_env_file=str(ROOT_DIR / ".env"))
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load settings: {str(e)}")
    raise
