# ruff: noqa: T201 PLC0415
import asyncio
import time
from pathlib import Path


async def run_agent(workspace_path: str, task_str: str | None) -> None:
    from lib import AgentContext, ModelConfig, build_primary_agent, get_event_bus

    if not Path(workspace_path).is_dir():
        raise ValueError("The provided workspace path should be a directory, not a file.")

    # Get the event bus
    event_bus = get_event_bus()

    # Create the context with the event bus
    context = AgentContext(workspace_path=workspace_path, event_bus=event_bus)

    agent = build_primary_agent(ModelConfig())

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


def run_ui() -> None:
    """Main entry point for the UI"""
    # Create an instance of the EventBus
    from cli.console import TerminalUI
    from lib import get_event_bus

    event_bus = get_event_bus()

    # Create the UI with the event bus
    session_id = "terminal-ui-session"
    ui = TerminalUI(session_id, event_bus)
    ui.initialize()
    strt_time = time.time()

    try:
        while True:
            ui.refresh()
            time.sleep(0.2)
            if time.time() - strt_time > 60:
                break
    except Exception as e:
        print(f"Error in UI loop: {e}")
    except KeyboardInterrupt:
        print("Received keyboard interrupt.")
    finally:
        # Ensure the display is stopped
        ui.stop_display()
        print("Code Agent UI stopped.")


async def run_app(workspace_path: str | None = None) -> None:
    """Run the agent and UI in parallel"""
    from cli.app import Application

    await Application(target_dir=workspace_path).run()


if __name__ == "__main__":
    workspace_path = "D:/react/DataView"
    tasks = ["can u read the project code and let me know ur understanding?"]
    # Run the UI in a separate thread
    # run_ui()
    # ui_thread.join()
    # asyncio.run(run_agent(workspace_path, tasks[0]))
    asyncio.run(run_app(workspace_path))
