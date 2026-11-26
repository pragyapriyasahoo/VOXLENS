"""
Unit tests for Summarization module.
Tests chunking, merging, and JSON validation with retry.
"""

import pytest
from summarization.chunker import TextChunker, ChunkerConfig, Chunk
from summarization.summarizer import Summarizer, SummarizerConfig
from summarization.llm_client import StubLLMClient, LLMConfig
from schema.messages import SummaryMessage, create_summary


class TestTextChunker:
    """Tests for text chunking."""
    
    @pytest.fixture
    def chunker(self):
        return TextChunker(ChunkerConfig(max_tokens=100, overlap_tokens=20))
    
    @pytest.fixture
    def sample_segments(self):
        return [
            {"text_clarified": "Hello everyone welcome to the meeting.", "speaker_name": "Teacher", "speaker_id": "S1"},
            {"text_clarified": "Today we discuss the project timeline.", "speaker_name": "Teacher", "speaker_id": "S1"},
            {"text_clarified": "The deadline is next Friday.", "speaker_name": "Student 1", "speaker_id": "S2"},
            {"text_clarified": "We need to finish the design phase.", "speaker_name": "Student 2", "speaker_id": "S3"},
            {"text_clarified": "Testing starts in two weeks.", "speaker_name": "Teacher", "speaker_id": "S1"},
        ]
    
    def test_short_conversation_single_chunk(self, chunker, sample_segments):
        """Test short conversation fits in one chunk."""
        chunks = chunker.chunk_segments(sample_segments, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].token_estimate < 500
    
    def test_long_conversation_multiple_chunks(self, chunker, sample_segments):
        """Test long conversation splits into chunks."""
        chunks = chunker.chunk_segments(sample_segments, max_tokens=30)
        assert len(chunks) >= 2
        # Each chunk should be within token limit (approximately)
        for chunk in chunks:
            assert chunk.token_estimate <= 40  # Some tolerance
    
    def test_chunk_overlap(self, chunker, sample_segments):
        """Test chunks have overlap."""
        chunks = chunker.chunk_segments(sample_segments, max_tokens=30)
        if len(chunks) >= 2:
            # Check that segments appear in multiple chunks
            all_segments = []
            for chunk in chunks:
                all_segments.extend([s.get("index", -1) for s in chunk.speaker_segments])
            
            # At least one segment should appear twice (overlap)
            from collections import Counter
            counts = Counter(all_segments)
            assert any(c > 1 for c in counts.values())
    
    def test_merge_summaries(self, chunker):
        """Test merging multiple chunk summaries."""
        chunk_summaries = [
            {
                "summary": "First part of meeting discussed timeline.",
                "key_points": ["Teacher: timeline discussed", "Student: deadline Friday"],
                "action_items": ["Finish design phase"],
            },
            {
                "summary": "Second part covered testing schedule.",
                "key_points": ["Teacher: testing in two weeks", "Student: need resources"],
                "action_items": ["Allocate testing resources"],
            },
        ]
        
        merged = chunker.merge_summaries(chunk_summaries)
        
        assert "timeline" in merged["summary"].lower() or "testing" in merged["summary"].lower()
        assert len(merged["key_points"]) <= 5
        assert len(merged["action_items"]) == 2
        # Check deduplication
        points_text = " ".join(merged["key_points"]).lower()
        assert points_text.count("teacher") <= 2  # May appear in both but deduped
    
    def test_merge_empty(self, chunker):
        """Test merging empty list."""
        merged = chunker.merge_summaries([])
        assert merged == {"summary": "", "key_points": [], "action_items": []}
    
    def test_merge_single(self, chunker):
        """Test merging single summary."""
        single = {"summary": "Test", "key_points": ["Point 1"], "action_items": ["Action 1"]}
        merged = chunker.merge_summaries([single])
        assert merged == single
    
    def test_key_point_length_limit(self, chunker):
        """Test key points truncated to 60 chars."""
        chunk_summaries = [{
            "summary": "Test",
            "key_points": ["x" * 70],  # Too long
            "action_items": [],
        }]
        merged = chunker.merge_summaries(chunk_summaries)
        assert len(merged["key_points"][0]) <= 60


class TestSummarizer:
    """Tests for the main Summarizer class."""
    
    @pytest.fixture
    def stub_llm(self):
        return StubLLMClient(LLMConfig())
    
    @pytest.fixture
    def summarizer(self, stub_llm):
        config = SummarizerConfig(
            window_minutes=15,
            interval_seconds=300,
            chunk_tokens=2000,
            llm=LLMConfig(),
            latency_target_seconds=20,
        )
        return Summarizer(config=config, llm_client=stub_llm)
    
    def test_add_caption(self, summarizer):
        """Test adding captions to window."""
        from schema.messages import CaptionMessage
        
        caption = CaptionMessage(
            text_original="test",
            text_clarified="Test message.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        
        summarizer.add_caption(caption)
        stats = summarizer.get_window_stats()
        assert stats["caption_count"] == 1
    
    def test_window_pruning(self, summarizer):
        """Test old captions are pruned."""
        from schema.messages import CaptionMessage
        
        # Add old caption
        old_caption = CaptionMessage(
            text_original="old",
            text_clarified="Old message.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        summarizer.add_caption(old_caption)
        
        # Add new caption far in future (beyond 15 min window)
        new_caption = CaptionMessage(
            text_original="new",
            text_clarified="New message.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=20 * 60 * 1000,  # 20 minutes later
            end_ms=20 * 60 * 1000 + 1000,
        )
        summarizer.add_caption(new_caption)
        
        stats = summarizer.get_window_stats()
        assert stats["caption_count"] == 1  # Old one pruned
    
    def test_request_summary_on_demand(self, summarizer, stub_llm):
        """Test on-demand summary generation."""
        from schema.messages import CaptionMessage
        
        # Add some captions
        for i in range(3):
            caption = CaptionMessage(
                text_original=f"segment {i}",
                text_clarified=f"Segment {i} discussed.",
                speaker_id="S1",
                speaker_name="Teacher",
                colour="FF6B6B",
                confidence=0.9,
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
            )
            summarizer.add_caption(caption)
        
        summary = summarizer.request_summary(scope="on_demand")
        
        assert summary is not None
        assert isinstance(summary, SummaryMessage)
        assert summary.scope == "on_demand"
        assert len(summary.summary) > 0
        assert len(summary.key_points) > 0
    
    def test_auto_summary_interval(self, summarizer):
        """Test automatic summary triggers at interval."""
        from schema.messages import CaptionMessage
        
        # Add captions spanning more than interval
        for i in range(10):
            caption = CaptionMessage(
                text_original=f"segment {i}",
                text_clarified=f"Segment {i}.",
                speaker_id="S1",
                speaker_name="Teacher",
                colour="FF6B6B",
                confidence=0.9,
                start_ms=i * 40000,  # 40 seconds each
                end_ms=(i + 1) * 40000,
            )
            summarizer.add_caption(caption)
        
        # Should have triggered auto-summary
        metrics = summarizer.get_metrics()
        assert metrics["summaries_generated"] >= 1
    
    def test_empty_window_summary(self, summarizer):
        """Test summary request on empty window returns None."""
        summary = summarizer.request_summary()
        assert summary is None
    
    def test_metrics_tracking(self, summarizer):
        """Test metrics are tracked."""
        from schema.messages import CaptionMessage
        
        caption = CaptionMessage(
            text_original="test",
            text_clarified="Test.",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        summarizer.add_caption(caption)
        summarizer.request_summary()
        
        metrics = summarizer.get_metrics()
        assert metrics["summaries_generated"] == 1
        assert metrics["avg_latency_s"] >= 0
        assert "window_stats" in metrics


class TestStubLLMClient:
    """Tests for stub LLM client."""
    
    def test_generate_chunk_response(self):
        """Test stub generates valid chunk summary."""
        client = StubLLMClient(LLMConfig())
        prompt = "Teacher: Hello\nStudent: Hi\n"
        
        response = client.generate(prompt)
        
        import json
        parsed = json.loads(response)
        assert "summary" in parsed
        assert "key_points" in parsed
        assert "action_items" in parsed
        assert isinstance(parsed["key_points"], list)
        assert isinstance(parsed["action_items"], list)
    
    def test_generate_merge_response(self):
        """Test stub generates merge response."""
        client = StubLLMClient(LLMConfig())
        prompt = "MERGE these summaries: Chunk 1: ... Chunk 2: ..."
        
        response = client.generate(prompt)
        
        import json
        parsed = json.loads(response)
        assert "summary" in parsed
        assert len(parsed["key_points"]) > 0
    
    def test_call_counting(self):
        """Test call count tracking."""
        client = StubLLMClient(LLMConfig())
        client.generate("test 1")
        client.generate("test 2")
        assert client.call_count == 2


class TestSummaryValidation:
    """Tests for summary message validation."""
    
    def test_valid_summary_message(self):
        """Test creating valid summary."""
        summary = create_summary(
            scope="interval",
            window_start_ms=0,
            window_end_ms=300000,
            summary="Meeting discussed project timeline and deadlines.",
            key_points=["Teacher: deadline Friday", "Student: need resources"],
            action_items=["Finish design", "Start testing"],
        )
        
        assert summary.type.value == "summary"
        assert summary.scope == "interval"
        assert len(summary.key_points) == 2
        assert len(summary.action_items) == 2
    
    def test_summary_key_points_limit(self):
        """Test key points limited to 5."""
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary="Test",
                key_points=["1", "2", "3", "4", "5", "6"],
                action_items=[],
            )
    
    def test_summary_key_point_length(self):
        """Test key point max 60 chars."""
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary="Test",
                key_points=["x" * 61],
                action_items=[],
            )
    
    def test_summary_length(self):
        """Test summary max 200 chars."""
        with pytest.raises(Exception):
            create_summary(
                scope="interval",
                window_start_ms=0,
                window_end_ms=300000,
                summary="x" * 201,
                key_points=[],
                action_items=[],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])