"""
Unit tests for the shared message schema.
Tests validation, creation, and rejection of malformed messages.
"""

import pytest
import json
from schema.messages import (
    MessageType,
    BaseMessage,
    CaptionMessage,
    SummaryMessage,
    NavigationMessage,
    validate_message,
    create_caption,
    create_summary,
    create_navigation,
    SchemaValidationError,
    NavigationInstruction,
)


class TestCaptionMessage:
    """Tests for caption message schema."""
    
    def test_valid_caption(self):
        """Test creating a valid caption message."""
        caption = create_caption(
            text_original="hello world",
            text_clarified="Hello world.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.95,
            start_ms=1000,
            end_ms=2500,
        )
        
        assert caption.type == MessageType.CAPTION
        assert caption.version == "1.0"
        assert caption.text_original == "hello world"
        assert caption.text_clarified == "Hello world."
        assert caption.speaker_id == "S1"
        assert caption.speaker_name == "Teacher"
        assert caption.colour == "FF6B6B"
        assert caption.confidence == 0.95
        assert caption.start_ms == 1000
        assert caption.end_ms == 2500
        assert caption.id is not None
        assert caption.timestamp is not None
    
    def test_caption_validation_confidence_bounds(self):
        """Test confidence must be 0.0-1.0."""
        with pytest.raises(Exception):
            create_caption(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="FF6B6B",
                confidence=1.5,  # Invalid
                start_ms=0,
                end_ms=1000,
            )
        
        with pytest.raises(Exception):
            create_caption(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="FF6B6B",
                confidence=-0.1,  # Invalid
                start_ms=0,
                end_ms=1000,
            )
    
    def test_caption_validation_colour_format(self):
        """Test colour must be 6-char hex."""
        with pytest.raises(Exception):
            create_caption(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="FF6B6",  # Too short
                confidence=0.9,
                start_ms=0,
                end_ms=1000,
            )
        
        with pytest.raises(Exception):
            create_caption(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="GG6B6B",  # Invalid hex
                confidence=0.9,
                start_ms=0,
                end_ms=1000,
            )
    
    def test_caption_validation_time_order(self):
        """Test end_ms must be >= start_ms."""
        with pytest.raises(Exception):
            create_caption(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="FF6B6B",
                confidence=0.9,
                start_ms=2000,
                end_ms=1000,  # Invalid
            )
    
    def test_caption_serialization(self):
        """Test JSON serialization and deserialization."""
        caption = create_caption(
            text_original="hello",
            text_clarified="Hello.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        
        json_str = caption.to_json()
        parsed = CaptionMessage.model_validate_json(json_str)
        
        assert parsed.text_original == "hello"
        assert parsed.text_clarified == "Hello."
        assert parsed.speaker_id == "S1"


class TestSummaryMessage:
    """Tests for summary message schema."""
    
    def test_valid_summary(self):
        """Test creating a valid summary message."""
        summary = create_summary(
            scope="interval",
            window_start_ms=0,
            window_end_ms=300000,
            summary="Discussion about VoxLens hardware and software.",
            key_points=["Teacher: VoxLens uses ESP8266", "Student: Raspberry Pi 5 option"],
            action_items=["Bring VoxLens glasses to class"],
        )
        
        assert summary.type == MessageType.SUMMARY
        assert summary.scope == "interval"
        assert summary.window_start_ms == 0
        assert summary.window_end_ms == 300000
        assert len(summary.key_points) == 2
        assert len(summary.action_items) == 1
    
    def test_summary_key_points_max_five(self):
        """Test key_points limited to 5 items."""
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary="Test",
                key_points=["1", "2", "3", "4", "5", "6"],  # 6 items
                action_items=[],
            )
    
    def test_summary_key_point_max_length(self):
        """Test each key_point max 60 chars."""
        long_point = "x" * 61
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary="Test",
                key_points=[long_point],
                action_items=[],
            )
    
    def test_summary_max_length(self):
        """Test summary max 200 chars."""
        long_summary = "x" * 201
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary=long_summary,
                key_points=[],
                action_items=[],
            )
    
    def test_summary_on_demand_scope(self):
        """Test on_demand scope."""
        summary = create_summary(
            scope="on_demand",
            window_start_ms=0,
            window_end_ms=300000,
            summary="Test",
            key_points=[],
            action_items=[],
        )
        assert summary.scope == "on_demand"


class TestNavigationMessage:
    """Tests for navigation message schema."""
    
    def test_valid_navigation_left(self):
        """Test left turn navigation."""
        nav = create_navigation(
            instruction=NavigationInstruction.LEFT,
            distance_m=10,
            next_label="Hallway A",
        )
        
        assert nav.type == MessageType.NAVIGATION
        assert nav.instruction == NavigationInstruction.LEFT
        assert nav.distance_m == 10
        assert nav.next_label == "Hallway A"
    
    def test_valid_navigation_arrive(self):
        """Test arrive instruction (no distance/label needed)."""
        nav = create_navigation(
            instruction=NavigationInstruction.ARRIVE,
            distance_m=None,
            next_label=None,
        )
        
        assert nav.instruction == NavigationInstruction.ARRIVE
        assert nav.distance_m is None
        assert nav.next_label is None
    
    def test_navigation_distance_non_negative(self):
        """Test distance_m must be non-negative."""
        with pytest.raises(Exception):
            create_navigation(
                instruction=NavigationInstruction.LEFT,
                distance_m=-5,
            )
    
    def test_navigation_next_label_max_length(self):
        """Test next_label max 20 chars."""
        with pytest.raises(Exception):
            create_navigation(
                instruction=NavigationInstruction.LEFT,
                distance_m=10,
                next_label="x" * 21,
            )
    
    def test_all_instruction_types(self):
        """Test all valid instruction types."""
        for instr in NavigationInstruction:
            nav = create_navigation(instruction=instr)
            assert nav.instruction == instr


class TestValidateMessage:
    """Tests for validate_message function."""
    
    def test_validate_valid_caption_json(self):
        """Test validating a valid caption JSON."""
        caption = create_caption(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Test",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        json_str = caption.to_json()
        
        result = validate_message(json_str)
        assert isinstance(result, CaptionMessage)
        assert result.text_original == "test"
    
    def test_validate_valid_summary_json(self):
        """Test validating a valid summary JSON."""
        summary = create_summary(
            scope="interval",
            window_start_ms=0,
            window_end_ms=300000,
            summary="Test summary",
            key_points=["Point 1"],
            action_items=[],
        )
        json_str = summary.to_json()
        
        result = validate_message(json_str)
        assert isinstance(result, SummaryMessage)
    
    def test_validate_valid_navigation_json(self):
        """Test validating a valid navigation JSON."""
        nav = create_navigation(
            instruction=NavigationInstruction.RIGHT,
            distance_m=5,
            next_label="Room 101",
        )
        json_str = nav.to_json()
        
        result = validate_message(json_str)
        assert isinstance(result, NavigationMessage)
    
    def test_reject_invalid_json(self):
        """Test rejection of malformed JSON."""
        with pytest.raises(SchemaValidationError) as exc:
            validate_message("not valid json")
        assert "Invalid JSON" in str(exc.value)
    
    def test_reject_missing_type(self):
        """Test rejection of message without type."""
        bad_json = json.dumps({"version": "1.0", "id": "123"})
        with pytest.raises(SchemaValidationError) as exc:
            validate_message(bad_json)
        assert "validation failed" in str(exc.value).lower()
    
    def test_reject_unknown_type(self):
        """Test rejection of unknown message type."""
        bad_json = json.dumps({"version": "1.0", "type": "unknown", "id": "123"})
        with pytest.raises(SchemaValidationError) as exc:
            validate_message(bad_json)
        assert "unknown message type" in str(exc.value).lower()
    
    def test_reject_caption_missing_fields(self):
        """Test rejection of caption with missing required fields."""
        bad_json = json.dumps({
            "version": "1.0",
            "type": "caption",
            "id": "123",
            "timestamp": "2024-01-01T00:00:00Z",
            # missing text_original, text_clarified, etc.
        })
        with pytest.raises(SchemaValidationError):
            validate_message(bad_json)


class TestBaseMessage:
    """Tests for base message functionality."""
    
    def test_timestamp_auto_generation(self):
        """Test timestamp is auto-generated if not provided."""
        caption = create_caption(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Test",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        assert caption.timestamp is not None
        # Should be valid ISO format
        from datetime import datetime
        dt = datetime.fromisoformat(caption.timestamp.replace("Z", "+00:00"))
        assert dt is not None
    
    def test_id_auto_generation(self):
        """Test UUID is auto-generated."""
        caption = create_caption(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Test",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        assert caption.id is not None
        assert len(caption.id) == 36  # UUID format
    
    def test_custom_timestamp(self):
        """Test custom timestamp is preserved."""
        custom_time = "2024-06-15T12:30:00Z"
        caption = CaptionMessage(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Test",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
            timestamp=custom_time,
        )
        assert caption.timestamp == custom_time
    
    def test_version_fixed(self):
        """Test version is always 1.0."""
        caption = create_caption(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Test",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        assert caption.version == "1.0"
        
        # Cannot override
        with pytest.raises(Exception):
            CaptionMessage(
                text_original="test",
                text_clarified="Test.",
                speaker_id="S1",
                speaker_name="Test",
                colour="FF6B6B",
                confidence=0.9,
                start_ms=0,
                end_ms=1000,
                version="2.0",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])