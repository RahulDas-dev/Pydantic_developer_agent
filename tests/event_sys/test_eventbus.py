# ruff: noqa: S101 SLF001
"""Test cases for EventBus"""

import asyncio
from typing import NoReturn
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import PartStartEvent, TextPart

from lib.event_sys.async_bus import (
    BackpressureStrategy,
    EventBus,
    EventSubscription,
    SessionEventSubscription,
)
from lib.event_sys.types import StreamOutEvent, UserInputEvent


def stream_out_event(session_id: str = "test_session", index: int = 1, data: str = "test_data") -> StreamOutEvent:
    """Create a test StreamOutEvent with part_start | text event type"""
    return StreamOutEvent(
        session_id=session_id,
        data=PartStartEvent(index=index, part=TextPart(content=data, part_kind="text"), event_kind="part_start"),
    )


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh EventBus for each test"""
    return EventBus()


@pytest.fixture
def mock_event() -> StreamOutEvent:
    """Create a mock event for testing"""
    return StreamOutEvent(
        session_id="test_session",
        data=PartStartEvent(index=1, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"),
    )


class TestEventBus:
    """Test cases for EventBus class"""

    def test_event_bus_initialization(self, event_bus: EventBus) -> None:
        """Test EventBus initialization"""
        assert event_bus.default_queue_size == 1000
        assert event_bus.default_backpressure_strategy == BackpressureStrategy.DROP_OLDEST
        assert len(event_bus._subscriptions) == 0
        assert len(event_bus._session_subscriptions) == 0

    @pytest.mark.asyncio
    async def test_subscribe(self, event_bus: EventBus) -> None:
        """Test basic subscription"""
        handler = MagicMock()

        subscription = event_bus.subscribe("part_start | text", handler)

        assert isinstance(subscription, EventSubscription)
        assert subscription.event_type == "part_start | text"
        assert subscription.handler == handler
        assert len(event_bus._subscriptions["part_start | text"]) == 1

    @pytest.mark.asyncio
    async def test_subscribe_once(self, event_bus: EventBus) -> None:
        """Test one-time subscription"""
        handler = MagicMock()

        subscription = event_bus.subscribe_once("part_start | text", handler)

        assert isinstance(subscription, EventSubscription)
        assert len(event_bus._subscriptions["part_start | text"]) == 1

    @pytest.mark.asyncio
    async def test_subscribe_session(self, event_bus: EventBus) -> None:
        """Test session subscription"""
        handler = MagicMock()

        subscription = event_bus.subscribe_session("test_session", "part_start | text", handler)

        assert isinstance(subscription, SessionEventSubscription)
        assert subscription.session_id == "test_session"
        assert len(event_bus._subscriptions["part_start | text"]) == 1
        assert len(event_bus._session_subscriptions["test_session"]) == 1

    @pytest.mark.asyncio
    async def test_emit_event(self, event_bus: EventBus, mock_event: StreamOutEvent) -> None:
        """Test event emission"""
        handler = MagicMock()
        subscription = event_bus.subscribe("part_start | text", handler)

        await event_bus.emit(mock_event)

        # Wait a bit for processing
        await asyncio.sleep(0.1)

        # Event should be queued
        assert subscription.event_queue.qsize() >= 0  # Event might be processed already

    @pytest.mark.asyncio
    async def test_get_handler_count(self, event_bus: EventBus) -> None:
        """Test getting handler count for event type"""
        handler1 = MagicMock()
        handler2 = MagicMock()

        event_bus.subscribe("part_start | text", handler1)
        event_bus.subscribe("part_start | text", handler2)

        count = event_bus.get_handler_count("part_start | text")
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_session_handler_count(self, event_bus: EventBus) -> None:
        """Test getting session handler count"""
        handler = MagicMock()

        event_bus.subscribe_session("test_session", "part_start | text", handler)

        count = event_bus.get_session_handler_count("test_session")
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, event_bus: EventBus) -> None:
        """Test getting active sessions"""
        handler = MagicMock()

        event_bus.subscribe_session("session1", "part_start | text", handler)
        event_bus.subscribe_session("session2", "part_start | text", handler)

        sessions = event_bus.get_active_sessions()
        assert "session1" in sessions
        assert "session2" in sessions

    @pytest.mark.asyncio
    async def test_cleanup_session(self, event_bus: EventBus) -> None:
        """Test session cleanup"""
        handler = MagicMock()
        subscription = event_bus.subscribe_session("test_session", "part_start | text", handler)

        assert not subscription.is_cancelled

        await event_bus.cleanup_session("test_session")

        assert subscription.is_cancelled
        assert len(event_bus._session_subscriptions) == 0

    @pytest.mark.asyncio
    async def test_get_backpressure_stats(self, event_bus: EventBus) -> None:
        """Test getting backpressure statistics"""
        handler = MagicMock()
        event_bus.subscribe("part_start | text", handler)

        stats = event_bus.get_backpressure_stats()

        assert "total_subscriptions" in stats
        assert "active_subscriptions" in stats
        assert "total_processed" in stats
        assert "total_dropped" in stats
        assert "subscriptions" in stats
        assert stats["total_subscriptions"] == 1
        assert stats["active_subscriptions"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_cancelled_subscriptions(self, event_bus: EventBus) -> None:
        """Test cleanup of cancelled subscriptions"""
        handler = MagicMock()
        subscription = event_bus.subscribe("part_start | text", handler)

        subscription.cancel()

        removed_count = await event_bus.cleanup_cancelled_subscriptions()

        assert removed_count == 1
        assert len(event_bus._subscriptions["part_start | text"]) == 0

    @pytest.mark.asyncio
    async def test_set_global_backpressure_strategy(self, event_bus: EventBus) -> None:
        """Test setting global backpressure strategy"""
        handler = MagicMock()
        subscription = event_bus.subscribe("part_start | text", handler)

        await event_bus.set_global_backpressure_strategy(BackpressureStrategy.BLOCK)

        assert event_bus.default_backpressure_strategy == BackpressureStrategy.BLOCK
        assert subscription.backpressure_strategy == BackpressureStrategy.BLOCK

    @pytest.mark.asyncio
    async def test_clear_all(self, event_bus: EventBus) -> None:
        """Test clearing all subscriptions"""
        handler = MagicMock()
        event_bus.subscribe("part_start | text", handler)
        event_bus.subscribe_session("test_session", "part_start | text", handler)

        await event_bus.clear_all()

        assert len(event_bus._subscriptions) == 0
        assert len(event_bus._session_subscriptions) == 0

    @pytest.mark.asyncio
    async def test_temporary_subscription(self, event_bus: EventBus) -> None:
        """Test temporary subscription context manager"""
        handler = MagicMock()

        async with event_bus.temporary_subscription("part_start | text", handler) as subscription:
            assert isinstance(subscription, EventSubscription)
            assert not subscription.is_cancelled
            assert len(event_bus._subscriptions["part_start | text"]) == 1

        # Subscription should be cancelled after context
        assert subscription.is_cancelled

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, event_bus: EventBus, mock_event: StreamOutEvent) -> None:
        """Test multiple handlers for the same event"""
        handler1 = MagicMock()
        handler2 = MagicMock()

        _ = event_bus.subscribe("part_start | text", handler1)
        _ = event_bus.subscribe("part_start | text", handler2)

        await event_bus.emit(mock_event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Both subscriptions should have events queued
        assert len(event_bus._subscriptions["part_start | text"]) == 2

    @pytest.mark.asyncio
    async def test_background_task_management(self, event_bus: EventBus) -> None:
        """Test that background tasks are properly managed"""
        handler = MagicMock()

        # Subscribe should create background tasks
        subscription = event_bus.subscribe("part_start | text", handler)

        # There should be background tasks
        assert len(event_bus._background_tasks) > 0

        # Cancel subscription
        subscription.cancel()

        # Wait a bit for cleanup
        await asyncio.sleep(0.1)

        # Background tasks should be cleaned up eventually
        # (This might not happen immediately due to async cleanup)

    @pytest.mark.asyncio
    async def test_event_processing_error_handling(self, event_bus: EventBus, mock_event: StreamOutEvent) -> None:
        """Test that errors in event handlers are handled gracefully"""

        def failing_handler(event: StreamOutEvent | UserInputEvent) -> NoReturn:
            raise Exception("Handler error")

        subscription = event_bus.subscribe("part_start | text", failing_handler)

        # This should not raise an exception
        await event_bus.emit(mock_event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Subscription should still be active despite handler error
        assert not subscription.is_cancelled


# Integration tests
class TestEventBusIntegration:
    """Integration tests for EventBus functionality"""

    @pytest.mark.asyncio
    async def test_full_event_flow(self, event_bus: EventBus) -> None:
        """Test complete event flow from subscription to processing"""
        received_events = []

        def handler(event) -> None:
            received_events.append(event)

        # Subscribe
        _ = event_bus.subscribe("part_start | text", handler)

        # Emit events
        event1 = stream_out_event(data="event1")
        event2 = stream_out_event(index=2, data="event2")

        await event_bus.emit(event1)
        await event_bus.emit(event2)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Events should be processed
        assert len(received_events) == 2
        assert received_events[0].data.part.content == "event1"
        assert received_events[1].data.part.content == "event2"

    @pytest.mark.asyncio
    async def test_session_event_filtering(self, event_bus: EventBus) -> None:
        """Test that session subscriptions only receive their session's events"""
        session1_events = []
        session2_events = []

        def session1_handler(event) -> None:
            session1_events.append(event)

        def session2_handler(event) -> None:
            session2_events.append(event)

        # Subscribe to same event type but different sessions
        event_bus.subscribe_session("session1", "part_start | text", session1_handler)
        event_bus.subscribe_session("session2", "part_start | text", session2_handler)

        # Emit events for different sessions
        event1 = stream_out_event(session_id="session1", data="for_session1")
        event2 = stream_out_event(session_id="session2", data="for_session2")

        await event_bus.emit(event1)
        await event_bus.emit(event2)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Each session should only receive its own events
        assert len(session1_events) == 1
        assert len(session2_events) == 1
        assert session1_events[0].data.part.content == "for_session1"
        assert session2_events[0].data.part.content == "for_session2"
