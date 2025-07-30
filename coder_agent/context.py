import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .prompts import (
    DIRECT_SYSTEM_ACCESS_MESSAGE,
    DOCKER_CONTAINER_MESSAGE,
    GIT_CONTEXT_MESSAGE,
    PYTHON_CONTEXT_MESSAGE,
    SANDBOX_CONTEXT_MESSAGE,
)
from .utils import has_python_files, is_git_repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentContext:
    workspace_path: str

    def retrieve_system_message(self) -> str | None:
        """
        Retrieve the system message from the file.
        """
        _system_md_var = os.environ.get("SYSTEM_MD", "").lower()

        if not _system_md_var:
            return None

        config_dir = Path(self.workspace_path) / ".config"
        _system_md_enabled = False

        if config_dir.exists():
            _system_message_file = config_dir / "system_messages.md"
            _system_md_enabled = bool(Path(_system_message_file).exists())

        if not _system_md_enabled:
            return None

        with open(_system_message_file, "r") as f:
            return f.read()

    def retrieve_sandbox_context(self) -> str:
        """
        Retrieve the sandbox context for the system message.
        """
        if os.environ.get("SANDBOX_CONTEXT"):
            return SANDBOX_CONTEXT_MESSAGE
        if os.environ.get("DOCKER_CONTAINER"):
            return DOCKER_CONTAINER_MESSAGE
        return DIRECT_SYSTEM_ACCESS_MESSAGE

    def retrieve_git_context(self) -> str | None:
        if is_git_repository(self.workspace_path):
            return GIT_CONTEXT_MESSAGE
        return None

    def retrieve_python_context(self) -> str | None:
        """
        Retrieve the core system message.
        """
        if has_python_files(self.workspace_path):
            return PYTHON_CONTEXT_MESSAGE
        return None
