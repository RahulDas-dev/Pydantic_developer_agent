from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

Environments = Literal["sandbox", "dev"]


class ExtraConfig(BaseSettings):
    ENVIRONMENT: Environments = "sandbox"
    DEBUG: bool = False
    TIMEZONE: str = Field(description="Timezone", default="UTC")

    @property
    def is_sandbox(self) -> bool:
        return self.ENVIRONMENT == "sandbox"
