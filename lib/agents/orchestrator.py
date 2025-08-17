import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ToolReturn

from lib.tools import glob_search, list_directory, read_file

from .context import AgentContext
from .factories import llm_factory

logger = logging.getLogger(__name__)


# Enums for status tracking
class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ActionItemStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class CoderAgentResponse(BaseModel):
    success: bool = Field(description="Whether the delegated action item was completed successfully")
    result: str | None = Field(
        default=None,
        description="The execution results, output data, and any relevant metadata from the completed action",
    )
    error: str | None = Field(
        default=None, description="Error message or exception details if the action item failed during execution"
    )
    deliverables: list[str] = Field(
        default_factory=list,
        description="list of files, artifacts, or outputs created by the coder agent (e.g., ['main.py', 'config.json', 'README.md'])",
    )


# Data models
class ActionItem(BaseModel):
    id: str
    description: str
    assigned_agent: str | None = None
    status: ActionItemStatus = ActionItemStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: CoderAgentResponse | None = None
    retry_count: int = 0


class TaskState(BaseModel):
    task_id: str
    original_request: str
    status: TaskStatus = TaskStatus.NOT_STARTED
    action_items: list[ActionItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_updates: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    Summary: str | None = None


orchestrator_agent = Agent[AgentContext, str](
    name="OrchestratorAgent",
    model=llm_factory(),
    tools=[
        Tool(glob_search, takes_ctx=True),
        Tool(list_directory, takes_ctx=True),
        Tool(read_file, takes_ctx=True),
    ],
    deps_type=AgentContext,
    output_type=str,
)


@orchestrator_agent.system_prompt
async def get_system_prompt(ctx: RunContext[AgentContext]) -> str:
    return ORCHESTRATOR_SYSTEM_MESSAGE


@orchestrator_agent.tool
def delegate_to_coderagent(
    ctx: RunContext[AgentContext],
    action_item_description: str,
) -> ToolReturn:
    """Delegates an action item to the CoderAgent for execution.

    Args:
        action_item_description (str): The action item description to delegate along with necessary context.
    Returns:
        Summary(str): The response from the CoderAgent after executing the action item.
    """
    pass


@orchestrator_agent.tool
def update_task_state(
    ctx: RunContext[AgentContext],
    task_id: str,
    action_item: ActionItem,
) -> ToolReturn:
    """Updates the state of a task with the given action item.
    Args:
        task_id (str): The ID of the task to update.
        action_item (ActionItem): The action item to add to the task state.
    Returns:
        TaskState: The updated task state.
    """
    pass


@orchestrator_agent.tool
def check_dependencies(
    ctx: RunContext[AgentContext],
    task_id: str,
) -> ToolReturn:
    """Checks if all dependencies for the given task are met.
    Args:
        task_id (str): The ID of the task to check dependencies for.
    Returns:
        bool: True if all dependencies are met, False otherwise.
    """
    pass
