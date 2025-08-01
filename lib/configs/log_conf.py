from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

LogLevel = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"]


class LogConfig(BaseSettings):
    LOG_LEVEL: LogLevel = "INFO"
    LOG_DIRECTORY: str = Field(default="")
    LOG_FORMAT: str = Field(
        description="Format string for log messages",
        default="%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)d] - %(message)s",
    )

    LOG_DATEFORMAT: str | None = Field(description="Date format string for log timestamps", default=None)

    @field_validator("LOG_DIRECTORY")
    @classmethod
    def name_must_contain_space(cls, value: str) -> str:
        if value == "":
            return value
        if Path(value).is_dir():
            return value
        raise ValueError("Log directory is not exists")
