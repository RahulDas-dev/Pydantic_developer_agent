from typing import Literal

from pydantic_settings import BaseSettings

TPERMISSION = Literal["yolo", "default", "none"]


class ToolsConfig(BaseSettings):
    PERMISSION: TPERMISSION = "default"
