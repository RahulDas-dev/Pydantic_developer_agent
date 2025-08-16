import asyncio
import logging
import uuid
from pathlib import Path
from queue import Queue
from typing import Any

from dotenv import load_dotenv

from cli.configs import AppConfig
from cli.console import TerminalUI
from cli.startup_ops import setup_envars, setup_loggers, setup_timezone, setup_warnings
from lib.event_sys import UserInputEvent, get_event_bus, reset_event_bus

logger = logging.getLogger("app")


class Application:
    def __init__(
        self, session_id: str | None = None, target_dir: str | None = None, config_params: dict[str, Any] | None = None
    ):
        self.session_id = uuid.uuid4().hex if session_id is None else session_id
        self.target_dir = target_dir if target_dir is not None else str(Path.cwd())
        self.config = AppConfig() if config_params is None else AppConfig(**config_params)
        self.ui = TerminalUI(self.session_id)
        self._initialized = False
        self.event_bus = get_event_bus()
        self._terminate_app = False
        self._input_events: Queue[UserInputEvent] = Queue(maxsize=100)  # Store last 10 input messages

    async def initialize(self) -> None:
        """
        Initializes the configuration, client, and tool call manager.
        This method must be called before processing any input.
        """
        # setup_logger_for_cli(self.config)
        setup_envars()
        setup_timezone(self.config)
        setup_warnings(self.config)
        setup_loggers(self.config)
        self._terminate_app = False
        logger.info(f"Config Details: {self.config}")
        status = load_dotenv()
        if not status:
            logger.warning("No .env file found Make Sure Required Environment Variables are set")
        self.ui.initialize()
        self._subscribe_to_events()
        self._initialized = True

    def _subscribe_to_events(self) -> None:
        event_handlers = {
            "input | text": self._handel_input_events,
            "input | command": self._handel_input_events,
            "input | exit": self._handel_input_events,
        }
        for event_type_str, handler in event_handlers.items():
            self.event_bus.subscribe(event_type_str, handler)  # type: ignore
            logger.debug(f"Subscribed specialized handler for {event_type_str}")

    async def _handel_input_events(self, event: UserInputEvent) -> None:
        if not isinstance(event, UserInputEvent):
            return
        self._input_events.put(event)

    async def run(self) -> None:
        if self._initialized is False:
            await self.initialize()

        while not self._terminate_app:
            event = self._input_events.get() if not self._input_events.empty() else None
            if event is None:
                await asyncio.sleep(0.1)
                self.ui.refresh()
                continue
            if event.event_type == "input | exit":
                await self._handle_exit_gracefully()
                self._terminate_app = True
                continue
            if event.event_type == "input | command":
                await self.handle_command(event)
            if event.event_type == "input | text":
                await self.handle_text_input(event)
            self.ui.refresh()
            await asyncio.sleep(0.1)

    async def _handle_exit_gracefully(self) -> None:
        logger.info("Exiting application gracefully...")
        self.ui.stop_display()
        await reset_event_bus()
        await self._save_state()

    async def _save_state(self) -> None:
        pass

    async def handle_command(self, event: UserInputEvent) -> None:
        pass

    async def handle_text_input(self, event: UserInputEvent) -> None:
        task_str = event.data.strip()
