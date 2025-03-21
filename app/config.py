import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


load_dotenv(override=True)


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL: str = os.getenv("MODEL")
    DB_SCHEMA: str = os.getenv("DB_SCHEMA")

    class Config:
        env_file = "../.env"


settings = Settings()
