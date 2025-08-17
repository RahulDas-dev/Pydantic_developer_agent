from __future__ import annotations

import asyncio
import logging
import os
import select
import sys
import time
from collections import deque
from datetime import datetime
from queue import Queue

from pydantic_ai.messages import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lib.event_sys import EventBus, StreamOutEvent, UserInputEvent, get_event_bus

# Import platform-specific modules for keyboard input
if os.name == "nt":  # Windows
    import msvcrt

# Constants
MAX_MESSAGES = 100
MAX_DISPLAYED_MESSAGES = 20
MAX_CONTENT_PREVIEW = 50  # Maximum length for content previews
CONTENT_PREVIEW_LENGTH = 50  # Preview length for content in tool results

logger = logging.getLogger(__name__)


class TerminalUI:
    """Main UI class for the Code Agent with pydantic-ai event integration"""

    def __init__(self, session_id: str, event_bus: EventBus | None = None):
        self.console = Console()
        self.event_bus = event_bus if event_bus is not None else get_event_bus()
        self.token_count = 0
        self.error_message = ""
        self.console_messages = deque(maxlen=MAX_MESSAGES)  # Queue-like structure with automatic length limiting
        self.running = True
        self.session_id = session_id
        self.processed_events = asyncio.Queue(maxsize=50)  # Store processed events as async queue
        self.layout_update_task = None  # For async layout update task

        # User input history
        self.user_input_history = Queue(maxsize=50)  # Store user input history
        self.current_input = ""  # Current user input being typed

        # Store active tool calls with their start times
        self.active_tool_calls: dict[str, float] = {}

        # Create the layout once
        self.layout = None
        self.live_display = None

    def initialize(self) -> None:
        """Start the UI display with event-driven updates and async layout updates"""
        if self.live_display is not None:
            return

        logger.info("Starting UI with event-driven updates...")
        self.running = True

        # Ensure subscriptions are set up
        self._setup_event_subscriptions()

        # Create a new layout if needed
        if not self.layout:
            self.layout = self._create_layout()

        # Start the live display with animations and no vertical spacing
        self.live_display = Live(self.layout, refresh_per_second=10, vertical_overflow="visible", transient=True)
        self.live_display.start()

        # Initial layout update
        self.update_layout(self.layout)

        # Start async layout updates if possible
        try:
            loop = asyncio.get_event_loop()
            self.layout_update_task = loop.create_task(self._async_layout_updates())
        except RuntimeError:
            logger.warning("Couldn't start async layout updates - falling back to event-driven updates")

    def _setup_event_subscriptions(self) -> None:
        """Subscribe to all event types with specialized handlers"""
        # Define event types and their handlers
        event_handlers = {
            "part_start | text": self._handle_part_start_text,
            "part_start | thinking": self._handle_part_start_thinking,
            "part_start | tool-call": self._null_handle,
            "part_start | builtin-tool-call": self._null_handle,
            "part_start | builtin-tool-return": self._null_handle,
            "part_delta | text": self._handle_part_delta_text,
            "part_delta | thinking": self._handle_part_delta_thinking,
            "part_delta | tool_call": self._handle_part_delta_tool_call,
            "final_result | ": self._handle_final_result_event,
            "function_tool_call | tool-call": self._handle_function_tool_call_event,
            "function_tool_result | tool-return": self._handle_function_tool_result_event,
            "function_tool_result | retry-prompt": self._handle_function_tool_result_event,
            "builtin_tool_call | builtin-tool-call": self._handle_builtin_tool_call_event,
            "builtin_tool_result | builtin-tool-return": self._handle_builtin_tool_result_event,
        }

        # Subscribe each event type to its specialized handler
        for event_type_str, handler in event_handlers.items():
            # Use a direct subscription without type annotation
            self.event_bus.subscribe(event_type_str, handler)  # type: ignore[arg-type]
            logger.debug(f"Subscribed specialized handler for {event_type_str}")

    async def _null_handle(self, event: StreamOutEvent) -> None:
        self._add_processed_event(event)

    async def _handle_part_start_text(self, event: StreamOutEvent) -> None:
        """Handle text-specific PartStartEvent with content extraction"""
        if not isinstance(event.data, PartStartEvent) or not isinstance(event.data.part, TextPart):
            return

        # Extract only the text content
        content = event.data.part.content
        formatted_message = f"\n\n{content}"

        # Store the formatted message
        self.console_messages.append(formatted_message)
        self._add_processed_event(event)

        # Update only the console panel with new content
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())

    async def _handle_part_start_thinking(self, event: StreamOutEvent) -> None:
        """Handle text-specific PartStartEvent with content extraction"""
        if not isinstance(event.data, PartStartEvent) or not isinstance(event.data.part, ThinkingPart):
            return

        # Extract only the text content
        content = event.data.part.content
        formatted_message = f"\n\n{content}"

        # Store the formatted message
        self.console_messages.append(formatted_message)
        self._add_processed_event(event)

        # Update only the console panel with new content
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())

    async def _handle_part_delta_text(self, event: StreamOutEvent) -> None:
        """Handle PartStartEvent with content extraction"""
        if not isinstance(event.data, PartDeltaEvent) or not isinstance(event.data.delta, TextPartDelta):
            return

        if len(event.data.delta.content_delta) > MAX_CONTENT_PREVIEW:
            content = event.data.delta.content_delta[:MAX_CONTENT_PREVIEW] + "..."
        else:
            content = event.data.delta.content_delta

        self.console_messages.append(content)
        self._add_processed_event(event)

        # Update only the console panel with new content
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())

    async def _handle_part_delta_thinking(self, event: StreamOutEvent) -> None:
        """Handle PartStartEvent with content extraction"""
        if not isinstance(event.data, PartDeltaEvent) or not isinstance(event.data.delta, ThinkingPartDelta):
            return

        if event.data.delta.content_delta is None:
            self._add_processed_event(event)
            return

        if len(event.data.delta.content_delta) > MAX_CONTENT_PREVIEW:
            content = event.data.delta.content_delta[:MAX_CONTENT_PREVIEW] + "..."
        else:
            content = event.data.delta.content_delta

        # Store the formatted message
        self.console_messages.append(content)
        self._add_processed_event(event)

        # Update only the console panel with new content
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())

    async def _handle_part_delta_tool_call(self, event: StreamOutEvent) -> None:
        """Handle PartDeltaEvent with content extraction"""
        if not isinstance(event.data, PartDeltaEvent) or not isinstance(event.data.delta, ToolCallPart):
            return

        # Extract the content from the delta
        tool_name = event.data.delta.tool_name
        content = f"{tool_name}"

        # Store incremental content for this part index (for UI display only)
        self.console_messages.append(content)
        self._add_processed_event(event)

        # Update only the console panel with new content
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())

    async def _handle_final_result_event(self, event: StreamOutEvent) -> None:
        """Handle FinalResultEvent with content extraction"""
        if not isinstance(event.data, FinalResultEvent):
            return
        self._add_processed_event(event)

    async def _handle_function_tool_call_event(self, event: StreamOutEvent) -> None:
        """Handle FunctionToolCallEvent with content extraction"""
        if not isinstance(event.data, FunctionToolCallEvent):
            return

        tool_name = event.data.part.tool_name

        # Store the formatted message
        self.console_messages.append(tool_name)
        self._add_processed_event(event)  # Store processed event for debugging

        # Update console and toolbar (for tool call timers)
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())
            self.layout["toolbar"].update(self._create_toolbar())

    async def _handle_function_tool_result_event(self, event: StreamOutEvent) -> None:
        """Handle FunctionToolResultEvent with content extraction"""
        if not isinstance(event.data, FunctionToolResultEvent):
            return

        # Create formatted message based on available properties
        tool_name = getattr(event.data.result, "tool_name", "")
        tool_info = f" via {tool_name}" if tool_name else ""
        formatted_message = f"Tool result received{tool_info}"

        # Check for content in case of retry prompt
        if hasattr(event.data.result, "content") and event.data.result.content:
            content_preview = str(event.data.result.content)[:CONTENT_PREVIEW_LENGTH]
            if len(str(event.data.result.content)) > CONTENT_PREVIEW_LENGTH:
                content_preview += "..."
            formatted_message = f"{formatted_message} - {content_preview}"

        # Store the formatted message
        self.console_messages.append(formatted_message)
        self._add_processed_event(event)

        # Update console and toolbar (for error message or tool call completion)
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())
            self.layout["toolbar"].update(self._create_toolbar())

    async def _handle_builtin_tool_call_event(self, event: StreamOutEvent) -> None:
        """Handle BuiltinToolCallEvent with content extraction"""
        if not isinstance(event.data, BuiltinToolCallEvent):
            return

        tool_name = event.data.part.tool_name
        self.console_messages.append(tool_name)
        self._add_processed_event(event)

        # Update console and toolbar (for tool call timers)
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())
            self.layout["toolbar"].update(self._create_toolbar())

    async def _handle_builtin_tool_result_event(self, event: StreamOutEvent) -> None:
        """Handle BuiltinToolResultEvent with content extraction"""
        if not isinstance(event.data, BuiltinToolResultEvent):
            return

        # Extract only the tool id
        formatted_message = str(event.data.result.metadata)

        # Store the formatted message
        self.console_messages.append(formatted_message)
        self._add_processed_event(event)

        # Update console and toolbar (for tool call completion)
        if self.live_display and self.layout:
            self.layout["console"].update(self._create_console_panel())
            self.layout["toolbar"].update(self._create_toolbar())

    def _add_processed_event(self, event: StreamOutEvent) -> None:
        """Add an event to the processed_events queue with error handling"""
        try:
            # Use put_nowait to avoid blocking in async methods
            self.processed_events.put_nowait(event)
        except asyncio.QueueFull:
            # If queue is full, make room by removing the oldest item
            try:
                # This is non-blocking but we need to handle if it's already empty
                self.processed_events.get_nowait()
                self.processed_events.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                logger.warning("Could not add event to processed_events queue")

    def _display_user_input(self, input_text: str) -> None:
        """Handle user input by logging it and updating UI"""
        try:
            # Create user input message for display
            input_message = f"\n> {input_text}"

            # Add to console messages for display
            self.console_messages.append(input_message)

            # Log the user input
            logger.info(f"User input: {input_text}")

            # Update the console display
            if self.live_display and self.layout:
                self.layout["console"].update(self._create_console_panel())
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            self.error_message = f"Error: {e!s}"
            if self.live_display and self.layout:
                self.layout["toolbar"].update(self._create_toolbar())

    def _create_toolbar(self) -> Panel:
        """Create the toolbar panel with active tool call information"""
        table = Table.grid(padding=1)
        table.add_column(style="dim", justify="left")
        table.add_column(style="dim", justify="center")
        table.add_column(style="dim", justify="right")

        # Token count
        token_display = f"Tokens: {self.token_count:,}"

        # Status with session info and active tool calls
        active_tools = len(self.active_tool_calls)
        if active_tools > 0:
            oldest_tool_time = min(self.active_tool_calls.values()) if self.active_tool_calls else 0
            elapsed = time.time() - oldest_tool_time if oldest_tool_time else 0
            status = f"Active ({active_tools} tools running, {elapsed:.1f}s)"
        else:
            status = f"Active (Session: {self.session_id[:8]}...)"

        # Error display
        error_display = f"Error: {self.error_message}" if self.error_message else "No Errors"

        table.add_row(token_display, f"Status: {status}", error_display)

        return Panel(table, title=None, border_style="dim blue", height=3, padding=0)

    def _truncate_content(self, content: str, max_length: int = MAX_CONTENT_PREVIEW, from_end: bool = False) -> str:
        """Helper to truncate content to a max length"""
        if not content:
            return ""

        if len(content) <= max_length:
            return content

        if from_end:
            # Get last max_length characters
            return f"...{content[-max_length:]}"

        # Get first max_length characters
        return f"{content[:max_length]}..."

    def _create_console_panel(self) -> Panel:
        """Create the console output panel with enhanced event formatting"""
        console_text = Text()

        # Get the last MAX_DISPLAYED_MESSAGES messages from deque
        displayed_messages = list(self.console_messages)
        if len(displayed_messages) > MAX_DISPLAYED_MESSAGES:
            displayed_messages = displayed_messages[-MAX_DISPLAYED_MESSAGES:]

        for message in displayed_messages:
            # Just add the pre-formatted message
            console_text.append(f"{message}")
        # Handle panel with minimal border and no padding
        return Panel(console_text, title=None, border_style="dim blue", padding=0)

    def _create_input_panel(self) -> Panel:
        """Create the input panel with current user input and history"""
        input_content = Text()

        # Show current input with cursor
        input_content.append("> ", style="bright_white bold")
        input_content.append(self.current_input, style="bright_white bold")
        input_content.append("█", style="bright_white bold blink")  # Blinking cursor

        # Add help text
        input_content.append("\n\nCommands:", style="bold")
        input_content.append("\n• /token <count> - Set token count")
        input_content.append("\n• /session <id> - Change session ID")
        input_content.append("\n• /clear - Clear console")
        input_content.append("\n• /quit - Exit application")

        return Panel(input_content, title=None, border_style="dim blue", padding=0)

    def _create_layout(self) -> Layout:
        """Create the main layout with toolbar at bottom and no gaps"""
        layout = Layout()

        # Create a single vertical layout with no gaps
        layout.split_column(
            Layout(name="console", ratio=16), Layout(name="input", ratio=7), Layout(name="toolbar", size=3)
        )

        return layout

    def update_layout(self, layout: Layout) -> None:
        """Update the layout with current content, ensuring no gap between panels"""
        layout["toolbar"].update(self._create_toolbar())
        layout["console"].update(self._create_console_panel())
        layout["input"].update(self._create_input_panel())

    async def _async_layout_updates(self) -> None:
        """Start an async task that continuously updates the layout"""
        # To store background tasks
        background_tasks = set()

        while self.running:
            # Check for keyboard input without blocking
            if os.name == "nt":  # Windows
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode("utf-8", errors="ignore")
                    task = asyncio.create_task(self.process_key_press(key))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
            elif select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                task = asyncio.create_task(self.process_key_press(key))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

            # Update the layout regardless of whether we processed an event
            if self.live_display and self.layout:
                self.update_layout(self.layout)

            # Short sleep to prevent CPU overuse
            await asyncio.sleep(0.1)  # Update at 10fps

    def stop_display(self) -> None:
        """Stop the UI display and cancel async tasks"""
        self.running = False

        # Cancel async layout update task if running
        if hasattr(self, "layout_update_task") and self.layout_update_task:
            try:
                self.layout_update_task.cancel()
            except Exception as e:
                logger.warning(f"Error cancelling layout update task: {e}")
            self.layout_update_task = None

        # Stop live display
        if self.live_display is not None:
            self.live_display.stop()
            self.live_display = None

    async def process_key_press(self, key: str) -> None:
        """Process a key press from the user and update the input panel"""
        if key in {"\n", "\r"}:  # Enter key
            # Process the current input
            await self._process_user_input()
        elif key in {"\b", "\x7f"}:  # Backspace
            # Remove the last character
            if self.current_input:
                self.current_input = self.current_input[:-1]
        else:
            # Add the key to the current input
            self.current_input += key

        # Update the input panel
        if self.live_display and self.layout:
            self.layout["input"].update(self._create_input_panel())

    async def _process_user_input(self) -> None:
        """Process the current user input and emit event if needed"""
        if not self.current_input:
            return

        # Add to history
        self.user_input_history.put(self.current_input)

        # Check for special commands
        if self.current_input.startswith("/"):
            self._handle_command()
        else:
            # Emit user input event
            await self._handle_user_input()

        # Clear current input
        self.current_input = ""

        # Update the input panel
        if self.live_display and self.layout:
            self.layout["input"].update(self._create_input_panel())

    def _handle_command(self) -> None:
        """Handle special commands that start with /"""
        parts = self.current_input.split()
        command = parts[0].lower()

        if command == "/token" and len(parts) > 1:
            try:
                self.token_count = int(parts[1])
                if self.live_display and self.layout:
                    self.layout["toolbar"].update(self._create_toolbar())
            except ValueError:
                self.error_message = f"Invalid token count: {parts[1]}"
        elif command == "/session" and len(parts) > 1:
            self.session_id = parts[1]
            if self.live_display and self.layout:
                self.layout["toolbar"].update(self._create_toolbar())
        elif command == "/clear":
            self.console_messages.clear()
            if self.live_display and self.layout:
                self.layout["console"].update(self._create_console_panel())
        elif command == "/quit":
            self.running = False

    async def _handle_user_input(self) -> None:
        """Process user input and add to display"""
        try:
            # Handle user input in our console
            await self.event_bus.emit(
                UserInputEvent(session_id=self.session_id, data=self.current_input, timestamp=datetime.now())
            )
            self._display_user_input(self.current_input)
            logger.debug(f"Processed user input: {self.current_input}")
        except Exception as e:
            logger.error(f"Error in _emit_user_input_event: {e}")
            self.error_message = f"Error: {e!s}"
            if self.live_display and self.layout:
                self.layout["toolbar"].update(self._create_toolbar())

    def refresh(self) -> None:
        """Manually refresh the display"""
        if self.live_display is not None and self.layout:
            self.update_layout(self.layout)
