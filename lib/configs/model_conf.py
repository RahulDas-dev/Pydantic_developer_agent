from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TProviderName = Literal["azure", "openai", "google", "anthropic", "aws", "ollama"]
TModelName = Literal[
    "gpt-4.0",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-pro",
    "claude-3-opus20240229",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # This is BEDROCK MODEL
]


class ModelConfig(BaseSettings):
    PROVIDER_NAME: TProviderName = Field(description="Name of the provider", default="azure")
    MODEL_NAME: TModelName = Field(description="Name of the model", default="gpt-4.1")
    API_KEY: str = Field(description="API key for the model", default="")
    API_ENDPOINT: str = Field(description="API endpoint for the model", default="")
    API_VERSION: str = Field(description="API version for the model", default="")

    model_config = SettingsConfigDict(
        frozen=True,
        env_nested_delimiter="__",
        env_file=".config",
        env_file_encoding="utf-8",
        extra="ignore",
    )
