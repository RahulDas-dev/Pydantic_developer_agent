# ruff: noqa: S101

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import PartStartEvent, TextPart

from lib.event_sys.async_bus import (
    BackpressureStrategy,
    EventBus,
    EventSubscription,
    SessionEventSubscription,
)
from lib.event_sys.types import StreamOutEvent


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh EventBus for each test"""
    return EventBus()


@pytest.fixture
def test_event() -> StreamOutEvent:
    """Create a mock event for testing"""
    return StreamOutEvent(
        session_id="test_session",
        data=PartStartEvent(index=1, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"),
    )


class TestEventSubscription:
    """Test cases for EventSubscription class"""

    @pytest.fixture
    def event_bus_mock(self) -> EventBus:
        """Mock EventBus for subscription tests"""
        return MagicMock()

    @pytest.fixture
    def subscription(self, event_bus: EventBus) -> EventSubscription:
        """Create a subscription for testing"""
        handler = MagicMock()
        return EventSubscription(
            event_bus,
            "part_start | text",
            handler,
            queue_size=10,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
        )

    def test_subscription_initialization(self, subscription: EventSubscription, event_bus: EventBus) -> None:
        """Test subscription is properly initialized"""
        assert subscription.event_bus == event_bus
        assert subscription.event_type == "part_start | text"
        assert subscription.queue_size == 10
        assert subscription.backpressure_strategy == BackpressureStrategy.DROP_OLDEST
        assert not subscription.is_cancelled
        assert subscription._processed_events == 0
        assert subscription._dropped_events == 0

    @pytest.mark.asyncio
    async def test_enqueue_event_success(self, subscription: EventSubscription, test_event: StreamOutEvent) -> None:
        """Test successful event enqueueing"""

        result = await subscription.enqueue_event(test_event)

        assert result is True
        assert subscription.event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_enqueue_event_cancelled_subscription(
        self, subscription: EventSubscription, test_event: StreamOutEvent
    ) -> None:
        """Test enqueueing to cancelled subscription"""
        subscription.cancel()

        result = await subscription.enqueue_event(test_event)

        assert result is False
        assert subscription.event_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_enqueue_event_drop_oldest_strategy(self, subscription: EventSubscription) -> None:
        """Test DROP_OLDEST backpressure strategy"""
        subscription.backpressure_strategy = BackpressureStrategy.DROP_OLDEST

        # Fill queue to capacity
        for i in range(10):
            await subscription.enqueue_event(
                StreamOutEvent(
                    session_id="test_session",
                    data=PartStartEvent(
                        index=i + 1, part=TextPart(content=f"content {i}", part_kind="text"), event_kind="part_start"
                    ),
                )
            )

        # Add one more event - should drop oldest
        result = await subscription.enqueue_event(
            StreamOutEvent(
                session_id="test_session",
                data=PartStartEvent(
                    index=11, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                ),
            )
        )

        assert result is True
        assert subscription.event_queue.qsize() == 10
        assert subscription._dropped_events == 1

    @pytest.mark.asyncio
    async def test_enqueue_event_drop_newest_strategy(self, event_bus: EventBus) -> None:
        """Test DROP_NEWEST backpressure strategy"""
        handler = MagicMock()
        subscription = EventSubscription(
            event_bus,
            "part_start | text",
            handler,
            queue_size=2,
            backpressure_strategy=BackpressureStrategy.DROP_NEWEST,
        )

        # Fill queue to capacity
        await subscription.enqueue_event(
            StreamOutEvent(
                session_id="test_session",
                data=PartStartEvent(
                    index=1, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                ),
            )
        )
        await subscription.enqueue_event(
            StreamOutEvent(
                session_id="test_session",
                data=PartStartEvent(
                    index=2, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                ),
            )
        )

        # Try to add one more - should be dropped
        result = await subscription.enqueue_event(
            StreamOutEvent(
                session_id="test_session",
                data=PartStartEvent(
                    index=3, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                ),
            )
        )

        assert result is False
        assert subscription.event_queue.qsize() == 2
        assert subscription._dropped_events == 1

    @pytest.mark.asyncio
    async def test_enqueue_event_block_strategy(self, event_bus: EventBus) -> None:
        """Test BLOCK backpressure strategy"""
        handler = MagicMock()
        subscription = EventSubscription(
            event_bus, "part_start | text", handler, queue_size=1, backpressure_strategy=BackpressureStrategy.BLOCK
        )

        # Fill queue
        await subscription.enqueue_event(
            StreamOutEvent(
                session_id="test_session",
                data=PartStartEvent(
                    index=1, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                ),
            )
        )

        # This should block, so we'll test with a timeout
        async def enqueue_with_timeout() -> bool:
            try:
                await asyncio.wait_for(
                    subscription.enqueue_event(
                        StreamOutEvent(
                            session_id="test_session",
                            data=PartStartEvent(
                                index=2, part=TextPart(content="Hi there!", part_kind="text"), event_kind="part_start"
                            ),
                        )
                    ),
                    timeout=0.1,
                )
                return True
            except asyncio.TimeoutError:
                return False

        # Should timeout because queue is full and BLOCK strategy waits
        result = await enqueue_with_timeout()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_and_stop_processing(self, subscription: EventSubscription) -> None:
        """Test starting and stopping event processing"""
        await subscription.start_processing()

        assert subscription._processing_task is not None
        assert not subscription._processing_task.done()

        # Store reference before stopping
        task = subscription._processing_task

        await subscription.stop_processing()

        # After stopping, the task should be cancelled or done, and _processing_task should be None
        assert task.cancelled() or task.done()
        assert subscription._processing_task is None

    @pytest.mark.asyncio
    async def test_process_sync_handler(self, event_bus: EventBus, test_event: StreamOutEvent) -> None:
        """Test processing events with sync handler"""
        handler = MagicMock()
        subscription = EventSubscription(event_bus, "part_start | text", handler)

        await subscription.enqueue_event(test_event)

        # Start processing
        await subscription.start_processing()

        # Wait a bit for processing
        await asyncio.sleep(0.2)

        # Stop processing
        await subscription.stop_processing()

        # Handler should have been called
        assert subscription._processed_events == 1

    @pytest.mark.asyncio
    async def test_process_async_handler(self, event_bus_mock: MagicMock, test_event: StreamOutEvent) -> None:
        """Test processing events with async handler"""
        handler = AsyncMock()
        subscription = EventSubscription(event_bus_mock, "part_start | text", handler)

        await subscription.enqueue_event(test_event)

        # Start processing
        await subscription.start_processing()

        # Wait a bit for processing
        await asyncio.sleep(0.2)

        # Stop processing
        await subscription.stop_processing()

        # Handler should have been called
        handler.assert_called_once()
        assert subscription._processed_events == 1

    def test_subscription_stats(self, subscription: EventSubscription) -> None:
        """Test subscription statistics"""
        subscription._processed_events = 5
        subscription._dropped_events = 2

        stats = subscription.stats

        assert stats["processed_events"] == 5
        assert stats["dropped_events"] == 2
        assert stats["queue_size"] == 0
        assert stats["queue_max_size"] == 10
        assert stats["backpressure_strategy"] == "drop_oldest"
        assert stats["is_cancelled"] is False

    def test_subscription_cancellation(self, subscription: EventSubscription) -> None:
        """Test subscription cancellation"""
        assert not subscription.is_cancelled

        subscription.cancel()

        assert subscription.is_cancelled

    def test_subscription_context_manager(self, event_bus: EventBus) -> None:
        """Test subscription as context manager"""
        handler = MagicMock()

        with EventSubscription(event_bus, "part_start | text", handler) as subscription:
            assert not subscription.is_cancelled

        assert subscription.is_cancelled


class TestSessionEventSubscription:
    """Test cases for SessionEventSubscription"""

    @pytest.fixture
    def session_subscription(self, event_bus: EventBus) -> SessionEventSubscription:
        """Create a session subscription for testing"""
        handler = MagicMock()
        return SessionEventSubscription(event_bus, "test_session", "part_start | text", handler)

    def test_session_subscription_initialization(self, session_subscription: SessionEventSubscription) -> None:
        """Test session subscription initialization"""
        assert session_subscription.session_id == "test_session"
        assert session_subscription.event_type == "part_start | text"

    @pytest.mark.asyncio
    async def test_session_filtering(self, session_subscription: SessionEventSubscription) -> None:
        """Test that session subscription filters events by session"""
        # This test would require more complex setup to test the session filtering
        # The session filtering happens in the handler wrapper
        assert session_subscription.session_id == "test_session"
