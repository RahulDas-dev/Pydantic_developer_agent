import logging
import os
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .prompts import (
    DIRECT_SYSTEM_ACCESS_MESSAGE,
    DOCKER_CONTAINER_MESSAGE,
    GIT_CONTEXT_MESSAGE,
    PYTHON_CONTEXT_MESSAGE,
    SANDBOX_CONTEXT_MESSAGE,
)
from .utils import has_node_files, has_python_files, is_git_repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentContext:
    workspace_path: str

    @property
    def is_workspace_empty(self) -> bool:
        """
        Check if the workspace directory is empty.
        """
        return not any(Path(self.workspace_path).iterdir())

    @property
    def is_git_repository(self) -> bool:
        """
        Check if the workspace is a git repository.
        """
        return is_git_repository(self.workspace_path)

    @property
    def is_python_project(self) -> bool:
        """
        Check if the workspace contains Python files.
        """
        return has_python_files(self.workspace_path)

    @property
    def is_node_project(self) -> bool:
        """
        Check if the workspace contains Python files.
        """
        return has_node_files(self.workspace_path)

    @property
    def is_docker_container(self) -> bool:
        """
        Check if the agent is running inside a Docker container.
        """
        return bool(os.environ.get("DOCKER_CONTAINER"))

    @property
    def is_sandboxed(self) -> bool:
        """
        Check if the agent is running in a sandboxed environment.
        """
        return bool(os.environ.get("SANDBOX_CONTEXT")) or bool(os.environ.get("DOCKER_CONTAINER"))

    def retrieve_system_message(self) -> str | None:
        """
        Retrieve the system message from the file.
        """
        _system_md_var = os.environ.get("SYSTEM_MD", "").lower()

        if not _system_md_var:
            return None

        config_dir = Path(self.workspace_path) / ".config"
        if config_dir.exists():
            _system_message_file = config_dir / "system_messages.md"
            if Path(_system_message_file).exists():
                with open(_system_message_file, "r") as f:
                    return f.read()
            return None
        return None

    def retrieve_sandbox_context(self) -> str:
        """
        Retrieve the sandbox context for the system message.
        """
        if os.environ.get("SANDBOX_CONTEXT"):
            return SANDBOX_CONTEXT_MESSAGE
        if os.environ.get("DOCKER_CONTAINER"):
            return DOCKER_CONTAINER_MESSAGE
        return Template(DIRECT_SYSTEM_ACCESS_MESSAGE).safe_substitute(CURRENT_WORKING_DIRECTORY=self.workspace_path)

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
