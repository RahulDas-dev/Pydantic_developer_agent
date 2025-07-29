import logging
import subprocess
from pathlib import Path
from typing import Any

from lib.base_types import CancellationToken

from .base import BaseTool, ToolExecutionContext, ToolExecutionResult, ToolParameter, ToolSchema

logger = logging.getLogger("tools")


class GitTool(BaseTool):
    """Tool for Git repository operations and information."""

    def __init__(self, root_directory: str | None = None):
        super().__init__(
            name="git_info",
            description="Provides information about the Git repository, including status, branch, "
            "recent commits, and changes.",
        )
        self.root_directory = Path(root_directory) if root_directory else Path.cwd()
        self.root_directory = self.root_directory.resolve()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    data_type="string",
                    description="The Git operation to perform",
                    default="info",
                    required=True,
                    extra={
                        "enum_values": ["status", "branch", "log", "diff", "info"],
                    },
                ),
                ToolParameter(
                    name="limit",
                    data_type="integer",
                    description="For 'log' operation: limit number of commits to show (default: 10)",
                    default=10,
                    extra={
                        "minimum": 1,
                        "maximum": 100,
                    },
                ),
            ],
        )

    def _is_git_repository(self, path: Path) -> bool:
        """Check if path is within a Git repository."""
        current = path
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _find_git_root(self, path: Path) -> Path | None:
        """Find the root of the Git repository."""
        current = path
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    async def _run_git_command(self, args: list[str], cwd: Path) -> dict[str, Any]:
        """Run a Git command and return the result."""
        try:
            cmd = ["git", *args]
            result = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    async def _get_git_status(self, git_root: Path) -> dict[str, Any]:
        """Get Git status information."""
        result = await self._run_git_command(["status", "--porcelain"], git_root)
        if not result["success"]:
            return result

        # Parse status output
        files = {"modified": [], "added": [], "deleted": [], "untracked": []}

        for line in result["stdout"].split("\n"):
            if not line.strip():
                continue

            status = line[:2]
            filename = line[3:].strip()

            if status[0] == "M" or status[1] == "M":
                files["modified"].append(filename)
            elif status[0] == "A" or status[1] == "A":
                files["added"].append(filename)
            elif status[0] == "D" or status[1] == "D":
                files["deleted"].append(filename)
            elif status == "??":
                files["untracked"].append(filename)

        return {
            "success": True,
            "files": files,
            "total_changes": sum(len(v) for v in files.values()),
        }

    async def _get_branch_info(self, git_root: Path) -> dict[str, Any]:
        """Get Git branch information."""
        branch_result = await self._run_git_command(["branch", "--show-current"], git_root)
        if not branch_result["success"]:
            return branch_result

        branches_result = await self._run_git_command(["branch", "-a"], git_root)

        return {
            "success": True,
            "current_branch": branch_result["stdout"],
            "all_branches": branches_result["stdout"].split("\n") if branches_result["success"] else [],
        }

    async def _get_commit_log(self, git_root: Path, limit: int) -> dict[str, Any]:
        """Get Git commit log."""
        result = await self._run_git_command(["log", f"-{limit}", "--oneline", "--decorate"], git_root)

        if not result["success"]:
            return result

        commits = [line.strip() for line in result["stdout"].split("\n") if line.strip()]

        return {"success": True, "commits": commits, "count": len(commits)}

    async def _get_diff_info(self, git_root: Path) -> dict[str, Any]:
        """Get Git diff information."""
        # Get staged diff
        staged_result = await self._run_git_command(["diff", "--cached", "--stat"], git_root)

        # Get unstaged diff
        unstaged_result = await self._run_git_command(["diff", "--stat"], git_root)

        return {
            "success": True,
            "staged_changes": staged_result["stdout"] if staged_result["success"] else "Error getting staged changes",
            "unstaged_changes": unstaged_result["stdout"]
            if unstaged_result["success"]
            else "Error getting unstaged changes",
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolExecutionContext,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionResult:
        """Execute the Git tool."""
        try:
            operation = parameters.get("operation", "info")
            limit = parameters.get("limit", 10)

            # Check if we're in a Git repository
            if not self._is_git_repository(self.root_directory):
                return ToolExecutionResult(
                    success=False,
                    content="Error: Not in a Git repository.",
                    metadata={"error": "not_git_repository"},
                )

            git_root = self._find_git_root(self.root_directory)
            if not git_root:
                return ToolExecutionResult(
                    success=False,
                    content="Error: Could not find Git repository root.",
                    metadata={"error": "git_root_not_found"},
                )

            result_content = []
            metadata = {"operation": operation, "git_root": str(git_root)}

            if operation in ["info", "status"]:
                # Get status information
                status_info = await self._get_git_status(git_root)
                if status_info["success"]:
                    result_content.append("=== Git Status ===")
                    files = status_info["files"]

                    if status_info["total_changes"] == 0:
                        result_content.append("Working directory clean")
                    else:
                        if files["modified"]:
                            result_content.append(f"Modified files ({len(files['modified'])}):")
                            result_content.extend([f"  M {f}" for f in files["modified"]])

                        if files["added"]:
                            result_content.append(f"Added files ({len(files['added'])}):")
                            result_content.extend([f"  A {f}" for f in files["added"]])

                        if files["deleted"]:
                            result_content.append(f"Deleted files ({len(files['deleted'])}):")
                            result_content.extend([f"  D {f}" for f in files["deleted"]])

                        if files["untracked"]:
                            result_content.append(f"Untracked files ({len(files['untracked'])}):")
                            result_content.extend([f"  ? {f}" for f in files["untracked"]])

                    metadata["status"] = status_info

                result_content.append("")

            if operation in ["info", "branch"]:
                # Get branch information
                branch_info = await self._get_branch_info(git_root)
                if branch_info["success"]:
                    result_content.append("=== Branch Information ===")
                    result_content.append(f"Current branch: {branch_info['current_branch']}")
                    result_content.append("")
                    metadata["branch"] = branch_info

            if operation in ["info", "log"]:
                # Get commit log
                log_info = await self._get_commit_log(git_root, limit)
                if log_info["success"]:
                    result_content.append(f"=== Recent Commits (last {log_info['count']}) ===")
                    result_content.extend(list(log_info["commits"]))
                    result_content.append("")
                    metadata["log"] = log_info

            if operation == "diff":
                # Get diff information
                diff_info = await self._get_diff_info(git_root)
                if diff_info["success"]:
                    result_content.append("=== Staged Changes ===")
                    if diff_info["staged_changes"]:
                        result_content.append(diff_info["staged_changes"])
                    else:
                        result_content.append("No staged changes")

                    result_content.append("")
                    result_content.append("=== Unstaged Changes ===")
                    if diff_info["unstaged_changes"]:
                        result_content.append(diff_info["unstaged_changes"])
                    else:
                        result_content.append("No unstaged changes")

                    metadata["diff"] = diff_info

            if not result_content:
                return ToolExecutionResult(
                    success=False,
                    content=f"Error: No information available for operation '{operation}'",
                    metadata={"error": "no_info_available"},
                )

            final_content = "\n".join(result_content)

            return ToolExecutionResult(
                success=True,
                content=final_content,
                display_content=f"Git {operation} information retrieved",
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Git tool execution failed: {e}")
            return ToolExecutionResult(
                success=False,
                content=f"Error: Git operation failed: {e!s}",
                metadata={"error": "execution_failed", "exception": str(e)},
            )
