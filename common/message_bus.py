"""
Message bus for inter-module communication.
Provides publish/subscribe pattern for simulator and future hardware integration.
"""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Awaitable, Any, Dict, List, Optional
from dataclasses import dataclass, field
import uuid
from loguru import logger


@dataclass
class Message:
    """Internal message envelope."""
    topic: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0)


class MessageBus(ABC):
    """Abstract message bus interface."""
    
    @abstractmethod
    async def publish(self, topic: str, payload: Any) -> None:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> str:
        """Subscribe to a topic. Returns subscription ID."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe by subscription ID."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the message bus."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the message bus."""
        pass


class InMemoryMessageBus(MessageBus):
    """In-memory message bus for simulator and testing."""
    
    def __init__(self):
        self._subscriptions: Dict[str, List[tuple[str, Callable[[Any], Awaitable[None]]]]] = defaultdict(list)
        self._running = False
        self._subscription_counter = 0
    
    async def publish(self, topic: str, payload: Any) -> None:
        """Publish a message to all subscribers of a topic."""
        if not self._running:
            logger.warning(f"MessageBus not running, dropping message to {topic}")
            return
        
        subscribers = self._subscriptions.get(topic, [])
        if not subscribers:
            logger.debug(f"No subscribers for topic: {topic}")
            return
        
        message = Message(topic=topic, payload=payload)
        logger.debug(f"Publishing to {topic}: {type(payload).__name__}")
        
        # Call all handlers concurrently
        tasks = []
        for sub_id, handler in subscribers:
            task = asyncio.create_task(self._safe_handle(handler, payload, sub_id))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handle(self, handler: Callable, payload: Any, sub_id: str) -> None:
        """Safely execute a handler, logging errors."""
        try:
            await handler(payload)
        except Exception as e:
            logger.error(f"Handler {sub_id} failed for message: {e}")
    
    async def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> str:
        """Subscribe to a topic."""
        self._subscription_counter += 1
        sub_id = f"sub_{self._subscription_counter}"
        self._subscriptions[topic].append((sub_id, handler))
        logger.debug(f"Subscribed {sub_id} to {topic}")
        return sub_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe by subscription ID."""
        for topic, handlers in self._subscriptions.items():
            self._subscriptions[topic] = [
                (sid, h) for sid, h in handlers if sid != subscription_id
            ]
        logger.debug(f"Unsubscribed {subscription_id}")
    
    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        logger.info("InMemoryMessageBus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        self._subscriptions.clear()
        logger.info("InMemoryMessageBus stopped")
    
    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self._subscriptions.get(topic, []))


# Topic constants
class Topics:
    RAW_TRANSCRIPT = "voxlens.raw_transcript"
    CAPTION = "voxlens.caption"
    SUMMARY = "voxlens.summary"
    NAVIGATION = "voxlens.navigation"
    POSITION_UPDATE = "voxlens.position_update"
    DESTINATION_SET = "voxlens.destination_set"
    SUMMARY_REQUEST = "voxlens.summary_request"  # on-demand summary trigger