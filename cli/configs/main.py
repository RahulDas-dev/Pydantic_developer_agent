from pydantic_settings import SettingsConfigDict

from cli.configs.log_conf import LogConfig
from lib.configs.model_conf import ModelConfig
from lib.configs.tools_config import ToolsConfig

from .dev_config import DevConfig


class AppConfig(LogConfig, ModelConfig, DevConfig, ToolsConfig):
    model_config = SettingsConfigDict(
        frozen=True,
        env_nested_delimiter="__",
        env_file=".config",
        env_file_encoding="utf-8",
        extra="ignore",
    )
