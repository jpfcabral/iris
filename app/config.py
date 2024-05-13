from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CELERY_BROKER_URL: str = ""
    CELERY_BACKEND_URL: str = ""


settings = Settings()
