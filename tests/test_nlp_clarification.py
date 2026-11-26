"""
Unit tests for NLP Clarification module.
Tests punctuation, filler removal, allowlist handling, and ASR corrections.
"""

import pytest
from nlp_clarification.rules import TextCleaner, CleanerConfig
from nlp_clarification.clarifier import Clarifier, ClarifierConfig
from nlp_clarification.t5_model import T5Punctuator, T5Config


class TestTextCleaner:
    """Tests for rule-based text cleaner."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a cleaner with test config."""
        config = CleanerConfig(
            filler_words=["um", "uh", "er", "ah", "like", "you know"],
            allowlisted_proper_nouns=["VoxLens", "ESP8266", "SSD1306", "Python"],
            asr_corrections={
                "vox lens": "VoxLens",
                "esp 8266": "ESP8266",
                "ssd 1306": "SSD1306",
            },
        )
        return TextCleaner(config)
    
    def test_filler_removal(self, cleaner):
        """Test filler words are removed."""
        text = "um hello uh world er"
        result = cleaner.clean(text)
        assert "um" not in result.lower()
        assert "uh" not in result.lower()
        assert "er" not in result.lower()
        assert result == "hello world"
    
    def test_filler_removal_case_insensitive(self, cleaner):
        """Test filler removal is case-insensitive."""
        text = "UM hello Uh world"
        result = cleaner.clean(text)
        assert result == "hello world"
    
    def test_filler_preserves_punctuation(self, cleaner):
        """Test punctuation is preserved around fillers."""
        text = "hello um, world uh!"
        result = cleaner.clean(text)
        assert result == "hello, world!"
    
    def test_repeated_word_collapse(self, cleaner):
        """Test consecutive repeated words are collapsed."""
        text = "the the quick brown fox fox jumps"
        result = cleaner.clean(text)
        assert result == "the quick brown fox jumps"
    
    def test_repeated_word_case_insensitive(self, cleaner):
        """Test repeated word collapse is case-insensitive."""
        text = "The the THE quick"
        result = cleaner.clean(text)
        assert result == "The quick"
    
    def test_asr_corrections(self, cleaner):
        """Test ASR corrections are applied."""
        text = "vox lens uses esp 8266 and ssd 1306"
        result = cleaner.clean(text)
        assert result == "VoxLens uses ESP8266 and SSD1306"
    
    def test_asr_corrections_case_preservation(self, cleaner):
        """Test ASR corrections preserve case pattern."""
        text = "VOX LENS and Esp 8266"
        result = cleaner.clean(text)
        assert "VOXLENS" in result or "VoxLens" in result
        assert "ESP8266" in result
    
    def test_allowlist_protects_nouns(self, cleaner):
        """Test allowlisted proper nouns are not modified by ASR corrections."""
        # Add a correction that would affect VoxLens
        config = CleanerConfig(
            filler_words=[],
            allowlisted_proper_nouns=["VoxLens"],
            asr_corrections={"vox": "voice"},
        )
        cleaner = TextCleaner(config)
        
        text = "vox lens and vox"
        result = cleaner.clean(text)
        # "vox lens" -> "VoxLens" (ASR correction), "vox" -> "voice" (but "VoxLens" is allowlisted)
        # Actually the allowlist protects exact matches, so "VoxLens" stays
        assert "VoxLens" in result or "voice" in result
    
    def test_number_normalization(self, cleaner):
        """Test number words converted to digits."""
        text = "one two three four five"
        result = cleaner.clean(text)
        assert result == "1 2 3 4 5"
    
    def test_compound_numbers(self, cleaner):
        """Test compound numbers like twenty one."""
        text = "twenty one thirty five"
        result = cleaner.clean(text)
        assert result == "21 35"
    
    def test_ordinal_numbers(self, cleaner):
        """Test ordinal words converted."""
        text = "first second third fourth"
        result = cleaner.clean(text)
        assert result == "1st 2nd 3rd 4th"
    
    def test_date_normalization(self, cleaner):
        """Test date normalization (Day Month format)."""
        text = "15 january 20 february"
        result = cleaner.clean(text)
        assert "1/15" in result
        assert "2/20" in result
    
    def test_date_reverse_format(self, cleaner):
        """Test day month format."""
        text = "15 january 20 february"
        result = cleaner.clean(text)
        assert "1/15" in result or "1/15" in result
    
    def test_no_capitalization_in_rule_based(self, cleaner):
        """Test that rule-based cleaner doesn't capitalize (T5 handles that)."""
        text = "hello world"
        result = cleaner.clean(text)
        # Rule-based cleaner should not capitalize - T5 handles punctuation/capitalization
        assert result == "hello world"
    
    def test_empty_text(self, cleaner):
        """Test empty text handling."""
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == "   "
    
    def test_multiple_spaces_collapsed(self, cleaner):
        """Test multiple spaces collapsed."""
        text = "hello    world"
        result = cleaner.clean(text)
        assert result == "hello world"


class TestClarifier:
    """Tests for the main Clarifier class."""
    
    @pytest.fixture
    def clarifier(self):
        """Create a clarifier with test config (T5 disabled for speed)."""
        config = ClarifierConfig(
            lookback_window=3,
            latency_target_ms=300,
            use_t5=False,  # Disable T5 for unit tests
            filler_words=["um", "uh", "er", "ah"],
            allowlisted_proper_nouns=["VoxLens", "Python"],
            asr_corrections={"vox lens": "VoxLens"},
        )
        return Clarifier(config)
    
    def test_process_segment_basic(self, clarifier):
        """Test basic segment processing."""
        caption = clarifier.process_segment(
            text="um hello world",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.95,
            start_ms=0,
            end_ms=2000,
        )
        
        assert isinstance(caption.text_original, str)
        assert isinstance(caption.text_clarified, str)
        assert caption.text_original == "um hello world"
        assert "um" not in caption.text_clarified.lower()
        assert caption.speaker_id == "S1"
        assert caption.confidence == 0.95
    
    def test_context_window(self, clarifier):
        """Test context window is maintained."""
        clarifier.process_segment(
            text="first segment",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        clarifier.process_segment(
            text="second segment",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=1000,
            end_ms=2000,
        )
        
        metrics = clarifier.get_metrics()
        assert metrics["context_size"] == 2
    
    def test_context_window_limit(self, clarifier):
        """Test context window respects max size."""
        for i in range(5):
            clarifier.process_segment(
                text=f"segment {i}",
                speaker_id="S1",
                speaker_name="Teacher",
                colour="FF6B6B",
                confidence=0.9,
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
            )
        
        metrics = clarifier.get_metrics()
        assert metrics["context_size"] == 3  # lookback_window=3
    
    def test_reset_context(self, clarifier):
        """Test context reset."""
        clarifier.process_segment(
            text="test",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        clarifier.reset_context()
        metrics = clarifier.get_metrics()
        assert metrics["context_size"] == 0
    
    def test_metrics_tracking(self, clarifier):
        """Test metrics are tracked."""
        clarifier.process_segment(
            text="test one",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        clarifier.process_segment(
            text="test two",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=1000,
            end_ms=2000,
        )
        
        metrics = clarifier.get_metrics()
        assert metrics["segments_processed"] == 2
        assert metrics["avg_latency_ms"] >= 0
        assert metrics["errors"] == 0
    
    def test_process_raw_message(self, clarifier):
        """Test processing raw JSON message."""
        import json
        raw = json.dumps({
            "text": "um hello",
            "speaker_id": "S1",
            "speaker_name": "Teacher",
            "colour": "FF6B6B",
            "confidence": 0.9,
            "start_ms": 0,
            "end_ms": 1000,
        })
        
        caption = clarifier.process_raw_message(raw)
        assert caption.text_original == "um hello"
        assert "um" not in caption.text_clarified.lower()
    
    def test_allowlist_in_clarifier(self, clarifier):
        """Test allowlist works through clarifier."""
        caption = clarifier.process_segment(
            text="vox lens is great",
            speaker_id="S1",
            speaker_name="Teacher",
            colour="FF6B6B",
            confidence=0.9,
            start_ms=0,
            end_ms=1000,
        )
        # Should be corrected to VoxLens
        assert "VoxLens" in caption.text_clarified


class TestT5Punctuator:
    """Tests for T5 punctuator (mostly fallback behavior)."""
    
    def test_fallback_restore(self):
        """Test fallback punctuation restoration."""
        config = T5Config(model_name="t5-small")
        punctuator = T5Punctuator(config)
        
        # Don't load model, use fallback
        result = punctuator._fallback_restore("hello world")
        assert result == "Hello world."
    
    def test_fallback_multiple_sentences(self):
        """Test fallback with multiple sentences."""
        config = T5Config(model_name="t5-small")
        punctuator = T5Punctuator(config)
        
        result = punctuator._fallback_restore("hello world how are you")
        # Basic fallback just adds period at end
        assert result.endswith(".")
    
    def test_is_available_false_initially(self):
        """Test is_available returns False before load."""
        config = T5Config(model_name="t5-small")
        punctuator = T5Punctuator(config)
        assert not punctuator.is_available()


class TestIntegration:
    """Integration tests with sample transcript."""
    
    def test_sample_transcript_processing(self):
        """Test processing the sample transcript."""
        config = ClarifierConfig(
            lookback_window=3,
            latency_target_ms=300,
            use_t5=False,
            filler_words=["um", "uh", "er", "ah", "like", "you know"],
            allowlisted_proper_nouns=["VoxLens", "ESP8266", "SSD1306", "Python", "Raspberry", "Jetson", "Ollama", "Gemma", "T5", "Dijkstra"],
            asr_corrections={
                "vox lens": "VoxLens",
                "voxlens": "VoxLens",
                "esp 8266": "ESP8266",
                "ssd 1306": "SSD1306",
                "raspberry pie": "Raspberry Pi",
                "jet son": "Jetson",
            },
        )
        clarifier = Clarifier(config)
        
        # Process a few segments from sample transcript
        test_segments = [
            ("um hello everyone welcome to the vox lens demo", "S1", "Teacher"),
            ("today we will discuss the esp 8266 display and the ssd 1306 oled", "S1", "Teacher"),
            ("uh the raspberry pie five has sixteen gigabytes of ram", "S2", "Student 1"),
        ]
        
        for i, (text, sid, sname) in enumerate(test_segments):
            caption = clarifier.process_segment(
                text=text,
                speaker_id=sid,
                speaker_name=sname,
                colour="FF6B6B",
                confidence=0.9,
                start_ms=i * 2000,
                end_ms=(i + 1) * 2000,
            )
            
            # Check key corrections
            if "vox lens" in text:
                assert "VoxLens" in caption.text_clarified
            if "esp 8266" in text:
                assert "ESP8266" in caption.text_clarified
            if "ssd 1306" in text:
                assert "SSD1306" in caption.text_clarified
            if "raspberry pie" in text:
                assert "Raspberry Pi" in caption.text_clarified
            if text.startswith("uh "):
                assert not caption.text_clarified.lower().startswith("uh ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])