from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    CELERY_BROKER_URL: str = ""
    CELERY_BACKEND_URL: str = ""
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_S3_ENDPOINT_URL: str = ""


settings = Settings()
