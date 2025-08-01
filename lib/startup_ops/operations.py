# ruff:  noqa: PLC0415
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from lib.configs import AgentConfig


def _setup_envars() -> None:
    from dotenv import load_dotenv

    load_dotenv()


def _setup_timezone(config: AgentConfig) -> None:
    os.environ["TZ"] = config.TIMEZONE or "UTC"
    if hasattr(time, "tzset"):
        time.tzset()  # type: ignore  # noqa: PGH003


def _setup_warnings(config: AgentConfig) -> None:
    if not config.DEBUG:
        import warnings

        warnings.simplefilter("ignore", ResourceWarning)


def _setup_loggers(config: AgentConfig) -> None:
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log_handlers: list[logging.Handler] = []
    if config.LOG_DIRECTORY:
        if not Path(config.LOG_DIRECTORY).is_dir():
            Path.mkdir(Path(config.LOG_DIRECTORY), parents=True, exist_ok=True)
        log_file = Path(config.LOG_DIRECTORY) / f"Llm_Log_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
        log_handlers.append(logging.FileHandler(filename=log_file))
        log_handlers.append(logging.StreamHandler(sys.stdout))
    else:
        log_handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFORMAT,
        handlers=log_handlers,
        force=True,
    )
    if config.is_sandbox and config.TIMEZONE:
        import pytz

        timezone = pytz.timezone(config.TIMEZONE)

        def time_converter(seconds: float | None) -> time.struct_time:
            if seconds is None:
                return datetime.now(tz=timezone).timetuple()
            return datetime.fromtimestamp(seconds, tz=timezone).timetuple()

        for handler in logging.root.handlers:
            if handler.formatter:
                handler.formatter.converter = time_converter

        for handler in logging.root.handlers:
            if handler.formatter:
                handler.formatter.converter = time_converter


def startup_operations(config: AgentConfig) -> None:
    """
    Perform all startup operations.
    """
    _setup_envars()
    _setup_timezone(config)
    _setup_warnings(config)
    _setup_loggers(config)
