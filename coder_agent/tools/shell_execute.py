import asyncio
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolExecutionContext, ToolExecutionResult, ToolParameter, ToolSchema


class ShellExecuteTool(BaseTool):
    """Tool for executing shell commands."""

    def __init__(self):
        super().__init__(name="shell_execute", description="Execute a shell command")

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="command", data_type="string", description="Shell command to execute", required=True
                ),
                ToolParameter(
                    name="working_dir",
                    data_type="string",
                    description="Working directory for command execution",
                    default=".",
                ),
                ToolParameter(
                    name="timeout", data_type="integer", description="Command timeout in seconds", default=30
                ),
                ToolParameter(
                    name="capture_output",
                    data_type="boolean",
                    description="Capture command output",
                    default=True,
                ),
            ],
            required=["command"],
        )

    async def should_confirm_execute(self, parameters: dict[str, Any], context: ToolExecutionContext) -> bool:
        """Always confirm shell command execution for security."""
        return True

    async def execute(self, parameters: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        """Execute the shell command tool."""
        command = parameters["command"]
        working_dir = parameters.get("working_dir", ".")
        timeout = parameters.get("timeout", 30)
        capture_output = parameters.get("capture_output", True)

        try:
            # Resolve working directory
            work_path = Path(working_dir).resolve()

            if not work_path.exists() or not work_path.is_dir():
                return ToolExecutionResult(
                    content=f"Invalid working directory: {working_dir}",
                    success=False,
                )

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=work_path,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                return ToolExecutionResult(
                    content=f"Command timed out after {timeout} seconds",
                    success=False,
                )

            # Format output
            output_parts = [f"Command: {command}"]
            output_parts.append(f"Working directory: {work_path}")
            output_parts.append(f"Exit code: {process.returncode}")

            if capture_output and stdout:
                output_parts.append(f"\\nSTDOUT:\\n{stdout.decode('utf-8', errors='replace')}")

            if capture_output and stderr:
                output_parts.append(f"\\nSTDERR:\\n{stderr.decode('utf-8', errors='replace')}")

            content = "\\n".join(output_parts)

            return ToolExecutionResult(
                content=content,
                success=process.returncode == 0,
                metadata={
                    "command": command,
                    "exit_code": process.returncode,
                    "working_dir": str(work_path),
                },
            )

        except Exception as e:
            return ToolExecutionResult(
                content=f"Failed to execute command: {e!s}",
                success=False,
            )
