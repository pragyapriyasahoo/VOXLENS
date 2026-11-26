"""
Shared JSON message schema for VoxLens intelligence modules.

All messages share common fields:
- version: "1.0"
- type: one of "caption", "summary", "navigation"
- id: unique string (UUID)
- timestamp: ISO 8601 UTC

Message-specific fields defined in each model.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, List, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator
import json


class MessageType(str, Enum):
    CAPTION = "caption"
    SUMMARY = "summary"
    NAVIGATION = "navigation"


class BaseMessage(BaseModel):
    """Base fields common to all message types."""
    version: Literal["1.0"] = "1.0"
    type: MessageType
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: str | datetime) -> str:
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc).isoformat()
        # Validate it's a valid ISO 8601 string
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError("timestamp must be valid ISO 8601 UTC")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json(cls, json_str: str) -> BaseMessage:
        """Deserialize from JSON string, determining type automatically."""
        data = json.loads(json_str)
        msg_type = data.get("type")
        if msg_type == MessageType.CAPTION:
            return CaptionMessage(**data)
        elif msg_type == MessageType.SUMMARY:
            return SummaryMessage(**data)
        elif msg_type == MessageType.NAVIGATION:
            return NavigationMessage(**data)
        else:
            raise SchemaValidationError(f"Unknown message type: {msg_type}")


class CaptionMessage(BaseMessage):
    """Real-time caption with original and clarified text."""
    type: Literal[MessageType.CAPTION] = MessageType.CAPTION
    text_original: str = Field(..., min_length=1, max_length=500)
    text_clarified: str = Field(..., min_length=1, max_length=500)
    speaker_id: str = Field(..., min_length=1, max_length=20)
    speaker_name: str = Field(..., min_length=1, max_length=50)
    colour: str = Field(..., pattern=r"^[0-9A-Fa-f]{6}$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_times(self) -> CaptionMessage:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class SummaryMessage(BaseMessage):
    """Conversation summary with key points and action items."""
    type: Literal[MessageType.SUMMARY] = MessageType.SUMMARY
    scope: Literal["on_demand", "interval"] = "interval"
    window_start_ms: int = Field(..., ge=0)
    window_end_ms: int = Field(..., ge=0)
    summary: str = Field(..., min_length=1, max_length=200)
    key_points: List[str] = Field(default_factory=list, max_length=5)
    action_items: List[str] = Field(default_factory=list)

    @field_validator("key_points", mode="before")
    @classmethod
    def validate_key_points(cls, v: List[str]) -> List[str]:
        if len(v) > 5:
            raise ValueError("key_points max 5 items")
        for i, point in enumerate(v):
            if len(point) > 60:
                raise ValueError(f"key_points[{i}] max 60 characters")
        return v

    @model_validator(mode="after")
    def validate_window(self) -> SummaryMessage:
        if self.window_end_ms < self.window_start_ms:
            raise ValueError("window_end_ms must be >= window_start_ms")
        return self


class NavigationInstruction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    U_TURN = "u_turn"
    ARRIVE = "arrive"
    RECALCULATING = "recalculating"


class NavigationMessage(BaseMessage):
    """Turn-by-turn navigation cue."""
    type: Literal[MessageType.NAVIGATION] = MessageType.NAVIGATION
    instruction: NavigationInstruction
    distance_m: Optional[int] = Field(default=None, ge=0)
    next_label: Optional[str] = Field(default=None, max_length=20)


class SchemaValidationError(Exception):
    """Raised when a message fails schema validation."""
    pass


def validate_message(json_str: str) -> BaseMessage:
    """
    Validate a JSON message against the schema.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        Validated message object
        
    Raises:
        SchemaValidationError: If message is invalid
    """
    try:
        return BaseMessage.from_json(json_str)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")
    except Exception as e:
        raise SchemaValidationError(f"Schema validation failed: {e}")


def create_caption(
    text_original: str,
    text_clarified: str,
    speaker_id: str,
    speaker_name: str,
    colour: str,
    confidence: float,
    start_ms: int,
    end_ms: int,
) -> CaptionMessage:
    """Factory for creating caption messages."""
    return CaptionMessage(
        text_original=text_original,
        text_clarified=text_clarified,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        colour=colour.upper(),
        confidence=confidence,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def create_summary(
    scope: Literal["on_demand", "interval"],
    window_start_ms: int,
    window_end_ms: int,
    summary: str,
    key_points: List[str],
    action_items: List[str],
) -> SummaryMessage:
    """Factory for creating summary messages."""
    return SummaryMessage(
        scope=scope,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        summary=summary,
        key_points=key_points,
        action_items=action_items,
    )


def create_navigation(
    instruction: NavigationInstruction,
    distance_m: Optional[int] = None,
    next_label: Optional[str] = None,
) -> NavigationMessage:
    """Factory for creating navigation messages."""
    return NavigationMessage(
        instruction=instruction,
        distance_m=distance_m,
        next_label=next_label,
    )