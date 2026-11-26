"""
Main NLP Clarification module.
Combines rule-based cleanup with T5 punctuation restoration.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
from loguru import logger

from .rules import TextCleaner, CleanerConfig, create_cleaner_from_config
from .t5_model import T5Punctuator, T5Config, create_punctuator_from_config
from schema.messages import (
    CaptionMessage,
    create_caption,
    validate_message,
    SchemaValidationError,
)
from common.config import get_config
from common.logging import get_logger, log_input, log_output, log_error


@dataclass
class ClarifierConfig:
    lookback_window: int = 3
    latency_target_ms: int = 300
    use_t5: bool = True
    t5_model: str = "t5-small"
    filler_words: List[str] = field(default_factory=list)
    allowlisted_proper_nouns: List[str] = field(default_factory=list)
    asr_corrections: dict = field(default_factory=dict)


class Clarifier:
    """
    NLP Clarification processor.
    
    Processes raw transcript segments and outputs clarified caption messages.
    Maintains a look-back window for context.
    """
    
    def __init__(self, config: Optional[ClarifierConfig] = None):
        self.config = config or self._load_config()
        self.logger = get_logger("nlp_clarification")
        
        # Initialize components
        cleaner_config = CleanerConfig(
            filler_words=self.config.filler_words,
            allowlisted_proper_nouns=self.config.allowlisted_proper_nouns,
            asr_corrections=self.config.asr_corrections,
        )
        self.cleaner = TextCleaner(cleaner_config)
        
        self.punctuator: Optional[T5Punctuator] = None
        if self.config.use_t5:
            t5_config = T5Config(model_name=self.config.t5_model)
            self.punctuator = T5Punctuator(t5_config)
        
        # Look-back window for context
        self._context: Deque[str] = deque(maxlen=self.config.lookback_window)
        
        # Metrics
        self._segment_count = 0
        self._total_latency_ms = 0.0
        self._errors = 0
        
        self.logger.info(f"Clarifier initialized: lookback={self.config.lookback_window}, "
                         f"use_t5={self.config.use_t5}, target_latency={self.config.latency_target_ms}ms")
    
    def _load_config(self) -> ClarifierConfig:
        """Load config from global config."""
        cfg = get_config()
        return ClarifierConfig(
            lookback_window=cfg.nlp.lookback_window,
            latency_target_ms=cfg.nlp.latency_target_ms,
            use_t5=cfg.nlp.use_t5,
            t5_model=cfg.nlp.t5_model,
            filler_words=cfg.nlp.filler_words,
            allowlisted_proper_nouns=cfg.nlp.allowlisted_proper_nouns,
            asr_corrections=cfg.nlp.asr_corrections,
        )
    
    def process_segment(
        self,
        text: str,
        speaker_id: str,
        speaker_name: str,
        colour: str,
        confidence: float,
        start_ms: int,
        end_ms: int,
    ) -> CaptionMessage:
        """
        Process a raw transcript segment into a clarified caption.
        
        Args:
            text: Raw STT output
            speaker_id: Speaker identifier (e.g., "S1")
            speaker_name: Speaker display name (e.g., "Teacher")
            colour: Hex colour code (6 chars, no #)
            confidence: STT confidence 0.0-1.0
            start_ms: Segment start offset from session start
            end_ms: Segment end offset from session start
            
        Returns:
            CaptionMessage with original and clarified text
        """
        start_time = time.perf_counter()
        
        log_input(self.logger, "nlp_clarification", {
            "text": text,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "confidence": confidence,
        })
        
        try:
            # 1. Rule-based cleanup
            cleaned = self.cleaner.clean(text, list(self._context))
            
            # 2. T5 punctuation and capitalization restoration
            if self.punctuator and self.punctuator.is_available():
                clarified = self.punctuator.restore(cleaned, list(self._context))
            else:
                clarified = cleaned
                # Apply basic capitalization fallback
                clarified = self._basic_capitalize(clarified)
            
            # 3. Create caption message
            caption = create_caption(
                text_original=text,
                text_clarified=clarified,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                colour=colour,
                confidence=confidence,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            
            # 4. Update context window
            self._context.append(clarified)
            
            # 5. Metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._segment_count += 1
            self._total_latency_ms += latency_ms
            
            log_output(self.logger, "nlp_clarification", {
                "text_original": text,
                "text_clarified": clarified,
                "latency_ms": round(latency_ms, 2),
            })
            
            if latency_ms > self.config.latency_target_ms:
                self.logger.warning(f"Latency target exceeded: {latency_ms:.1f}ms > {self.config.latency_target_ms}ms")
            
            return caption
            
        except Exception as e:
            self._errors += 1
            log_error(self.logger, "nlp_clarification", e)
            # Return minimal caption on error
            return create_caption(
                text_original=text,
                text_clarified=text,  # Fallback to original
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                colour=colour,
                confidence=confidence,
                start_ms=start_ms,
                end_ms=end_ms,
            )
    
    def _basic_capitalize(self, text: str) -> str:
        """Basic capitalization fallback."""
        if not text:
            return text
        # Capitalize first letter
        for i, ch in enumerate(text):
            if ch.isalpha():
                text = text[:i] + ch.upper() + text[i+1:]
                break
        # Ensure ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        return text
    
    def process_raw_message(self, raw_json: str) -> CaptionMessage:
        """
        Process a raw transcript message in JSON format.
        
        Expected format:
        {
            "text": "...",
            "speaker_id": "S1",
            "speaker_name": "Teacher",
            "colour": "FF6B6B",
            "confidence": 0.95,
            "start_ms": 1000,
            "end_ms": 2500
        }
        """
        import json
        data = json.loads(raw_json)
        return self.process_segment(
            text=data["text"],
            speaker_id=data["speaker_id"],
            speaker_name=data["speaker_name"],
            colour=data["colour"],
            confidence=data["confidence"],
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
        )
    
    def get_metrics(self) -> dict:
        """Get processing metrics."""
        avg_latency = self._total_latency_ms / self._segment_count if self._segment_count > 0 else 0
        return {
            "segments_processed": self._segment_count,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "target_latency_ms": self.config.latency_target_ms,
            "errors": self._errors,
            "context_size": len(self._context),
        }
    
    def reset_context(self) -> None:
        """Clear the context window."""
        self._context.clear()
        self.logger.debug("Context window cleared")


def create_clarifier_from_config(config_dict: dict = None) -> Clarifier:
    """Factory to create Clarifier from config."""
    if config_dict is None:
        cfg = get_config()
        config_dict = {
            "lookback_window": cfg.nlp.lookback_window,
            "latency_target_ms": cfg.nlp.latency_target_ms,
            "use_t5": cfg.nlp.use_t5,
            "t5_model": cfg.nlp.t5_model,
            "filler_words": cfg.nlp.filler_words,
            "allowlisted_proper_nouns": cfg.nlp.allowlisted_proper_nouns,
            "asr_corrections": cfg.nlp.asr_corrections,
        }
    return Clarifier(ClarifierConfig(**config_dict))