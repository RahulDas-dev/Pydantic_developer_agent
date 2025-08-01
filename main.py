# ruff: noqa: T201
import asyncio
from pathlib import Path

from lib import AgentContext, build_agent_from_config, config, startup_operations

startup_operations(config)

workspace_path = "D:/react/DataView"
tasks = ["can u read the project code and let me know ur understanding?"]


async def run_agent(workspace_path: str, task_str: str | None) -> None:
    if not Path(workspace_path).is_dir():
        raise ValueError("The provided workspace path should be a directory, not a file.")

    context = AgentContext(workspace_path=workspace_path)
    agent = build_agent_from_config(config)

    while True:
        while not task_str:
            task_str = input("Enter your task or type quit/exit to terminate: > ")
            if task_str.strip():
                break
            print("Task cannot be empty. Please enter a valid task or type quit/exit to terminate: > ")

        if task_str.lower() in ["quit", "exit"]:
            break
        await asyncio.sleep(0.1)
        results = await agent.run(user_prompt=task_str, deps=context)
        print(results.output)
        task_str = None


asyncio.run(run_agent(workspace_path, tasks[0]))
