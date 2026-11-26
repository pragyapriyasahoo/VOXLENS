"""
Main Summarization module.
Manages rolling window, triggers summaries, and coordinates chunking/merging.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque, Callable, Awaitable
from collections import deque
from datetime import datetime, timezone
from loguru import logger

from .chunker import TextChunker, ChunkerConfig, create_chunker_from_config
from .llm_client import LLMClient, LLMConfig, StubLLMClient, create_llm_client
from schema.messages import (
    CaptionMessage,
    SummaryMessage,
    create_summary,
    validate_message,
    SchemaValidationError,
)
from common.config import get_config
from common.logging import get_logger, log_input, log_output, log_error


@dataclass
class SummarizerConfig:
    window_minutes: int = 15
    interval_seconds: int = 300
    max_summary_chars: int = 200
    max_key_points: int = 5
    max_key_point_chars: int = 60
    chunk_tokens: int = 2000
    llm: LLMConfig = field(default_factory=LLMConfig)
    latency_target_seconds: int = 20


class Summarizer:
    """
    Local AI Summarization processor.
    
    Maintains a rolling window of clarified captions.
    Triggers summaries on demand or at intervals.
    Uses chunking for long conversations.
    """
    
    def __init__(
        self,
        config: Optional[SummarizerConfig] = None,
        llm_client: Optional[LLMClient] = None,
        chunker: Optional[TextChunker] = None,
        on_summary: Optional[Callable[[SummaryMessage], Awaitable[None]]] = None,
    ):
        self.config = config or self._load_config()
        self.logger = get_logger("summarization")
        
        # Components
        self.chunker = chunker or create_chunker_from_config({
            "chunk_tokens": self.config.chunk_tokens,
        })
        self.llm_client = llm_client or create_llm_client({
            "model": self.config.llm.model,
            "backend": self.config.llm.backend,
            "host": self.config.llm.host,
            "timeout_seconds": self.config.llm.timeout_seconds,
            "temperature": self.config.llm.temperature,
            "max_retries": self.config.llm.max_retries,
        })
        
        # Rolling window of captions (max 15 minutes)
        self.window_ms = self.config.window_minutes * 60 * 1000
        self._captions: Deque[CaptionMessage] = deque()
        
        # Callback for emitting summaries
        self.on_summary = on_summary
        
        # Auto-summary timer
        self._last_auto_summary_ms = 0
        self._auto_summary_interval_ms = self.config.interval_seconds * 1000
        
        # Metrics
        self._summary_count = 0
        self._total_latency_s = 0.0
        self._errors = 0
        
        self.logger.info(f"Summarizer initialized: window={self.config.window_minutes}min, "
                         f"interval={self.config.interval_seconds}s, "
                         f"chunk_tokens={self.config.chunk_tokens}")
    
    def _load_config(self) -> SummarizerConfig:
        """Load config from global config."""
        cfg = get_config()
        return SummarizerConfig(
            window_minutes=cfg.summarization.window_minutes,
            interval_seconds=cfg.summarization.interval_seconds,
            max_summary_chars=cfg.summarization.max_summary_chars,
            max_key_points=cfg.summarization.max_key_points,
            max_key_point_chars=cfg.summarization.max_key_point_chars,
            chunk_tokens=cfg.summarization.chunk_tokens,
            llm=LLMConfig(
                model=cfg.summarization.llm.model,
                backend=cfg.summarization.llm.backend,
                host=cfg.summarization.llm.host,
                timeout_seconds=cfg.summarization.llm.timeout_seconds,
                temperature=cfg.summarization.llm.temperature,
                max_retries=cfg.summarization.llm.max_retries,
            ),
            latency_target_seconds=cfg.summarization.latency_target_seconds,
        )
    
    def add_caption(self, caption: CaptionMessage) -> None:
        """Add a caption to the rolling window."""
        self._captions.append(caption)
        self._prune_window()
        
        # Check for auto-summary trigger
        if caption.end_ms - self._last_auto_summary_ms >= self._auto_summary_interval_ms:
            self._trigger_summary(scope="interval", window_end_ms=caption.end_ms)
    
    def _prune_window(self) -> None:
        """Remove captions older than the window."""
        if not self._captions:
            return
        
        current_time = self._captions[-1].end_ms
        cutoff = current_time - self.window_ms
        
        while self._captions and self._captions[0].end_ms < cutoff:
            self._captions.popleft()
    
    def request_summary(self, scope: str = "on_demand") -> Optional[SummaryMessage]:
        """
        Request an on-demand summary of current window.
        
        Args:
            scope: "on_demand" or "interval"
            
        Returns:
            SummaryMessage or None if window empty
        """
        if not self._captions:
            self.logger.warning("Summary requested but window is empty")
            return None
        
        window_start = self._captions[0].start_ms
        window_end = self._captions[-1].end_ms
        
        return self._trigger_summary(scope=scope, window_start_ms=window_start, window_end_ms=window_end)
    
    def _trigger_summary(
        self,
        scope: str,
        window_start_ms: Optional[int] = None,
        window_end_ms: Optional[int] = None,
    ) -> Optional[SummaryMessage]:
        """Generate and emit a summary."""
        if not self._captions:
            return None
        
        start_time = time.perf_counter()
        
        if window_start_ms is None:
            window_start_ms = self._captions[0].start_ms
        if window_end_ms is None:
            window_end_ms = self._captions[-1].end_ms
        
        log_input(self.logger, "summarization", {
            "scope": scope,
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "caption_count": len(self._captions),
        })
        
        try:
            # Convert captions to segment dicts for chunker
            segments = []
            for cap in self._captions:
                if cap.start_ms >= window_start_ms and cap.end_ms <= window_end_ms:
                    segments.append({
                        "text_clarified": cap.text_clarified,
                        "speaker_name": cap.speaker_name,
                        "speaker_id": cap.speaker_id,
                        "start_ms": cap.start_ms,
                        "end_ms": cap.end_ms,
                    })
            
            if not segments:
                self.logger.warning("No segments in requested window")
                return None
            
            # Chunk if needed
            chunks = self.chunker.chunk_segments(segments, self.config.chunk_tokens)
            
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Summarizing chunk {i+1}/{len(chunks)} (~{chunk.token_estimate} tokens)")
                summary = self._summarize_chunk(chunk.text)
                if summary:
                    chunk_summaries.append(summary)
            
            if not chunk_summaries:
                self.logger.error("All chunk summaries failed")
                return None
            
            # Merge if multiple chunks
            if len(chunk_summaries) > 1:
                merged = self.chunker.merge_summaries(chunk_summaries)
                final_summary = self._merge_summaries(merged)
            else:
                final_summary = chunk_summaries[0]
            
            # Validate and create message
            summary_msg = create_summary(
                scope=scope,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                summary=final_summary["summary"][:self.config.max_summary_chars],
                key_points=[kp[:self.config.max_key_point_chars] for kp in final_summary["key_points"][:self.config.max_key_points]],
                action_items=final_summary["action_items"],
            )
            
            # Validate with schema
            validate_message(summary_msg.to_json())
            
            # Update metrics
            latency_s = time.perf_counter() - start_time
            self._summary_count += 1
            self._total_latency_s += latency_s
            
            if scope == "interval":
                self._last_auto_summary_ms = window_end_ms
            
            log_output(self.logger, "summarization", {
                "scope": scope,
                "summary": summary_msg.summary[:50] + "...",
                "key_points": len(summary_msg.key_points),
                "action_items": len(summary_msg.action_items),
                "latency_s": round(latency_s, 2),
            })
            
            if latency_s > self.config.latency_target_seconds:
                self.logger.warning(f"Latency target exceeded: {latency_s:.1f}s > {self.config.latency_target_seconds}s")
            
            # Emit via callback
            if self.on_summary:
                import asyncio
                import inspect
                if inspect.iscoroutinefunction(self.on_summary):
                    asyncio.create_task(self.on_summary(summary_msg))
                else:
                    # Sync callback - call directly
                    self.on_summary(summary_msg)
            
            return summary_msg
            
        except SchemaValidationError as e:
            self._errors += 1
            log_error(self.logger, "summarization", e)
            self.logger.error(f"Summary validation failed: {e}")
            return None
        except Exception as e:
            self._errors += 1
            log_error(self.logger, "summarization", e)
            self.logger.error(f"Summary generation failed: {e}")
            return None
    
    def _summarize_chunk(self, text: str) -> Optional[dict]:
        """Summarize a single text chunk using LLM."""
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_chunk_prompt(text)
        
        for attempt in range(self.config.llm.max_retries + 1):
            try:
                response = self.llm_client.generate(user_prompt, system_prompt)
                parsed = self._parse_llm_response(response)
                if parsed:
                    return parsed
                
                if attempt < self.config.llm.max_retries:
                    self.logger.warning(f"LLM returned invalid JSON, retrying (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"LLM generation error (attempt {attempt + 1}): {e}")
                if attempt >= self.config.llm.max_retries:
                    raise
        
        return None
    
    def _merge_summaries(self, merged: dict) -> dict:
        """Final merge pass using LLM for coherent summary."""
        # For now, use the simple merge from chunker
        # Could add a second LLM pass here for better coherence
        return merged
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for summarization."""
        return """You are a concise meeting summarizer. Output ONLY valid JSON with these fields:
- summary: string, max 200 characters, overview of conversation
- key_points: array of up to 5 strings, each max 60 characters, attribute to speakers (e.g. "Teacher: exam Friday")
- action_items: array of strings, specific tasks mentioned

Do not include any extra text, explanations, or formatting outside the JSON."""
    
    def _get_chunk_prompt(self, text: str) -> str:
        """Get user prompt for chunk summarization."""
        return f"""Summarize this conversation segment:

{text}

Output JSON only."""
    
    def _get_merge_prompt(self, summaries: List[dict]) -> str:
        """Get prompt for merging chunk summaries."""
        combined = "\n\n".join([f"Chunk {i+1}: {s['summary']}" for i, s in enumerate(summaries)])
        return f"""Merge these chunk summaries into a coherent overall summary:

{combined}

Output JSON only with merged summary, key_points, and action_items."""
    
    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse and validate LLM JSON response."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Find JSON object
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            json_str = response[start:end+1]
        else:
            json_str = response
        
        try:
            parsed = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(parsed.get("summary"), str):
                return None
            if not isinstance(parsed.get("key_points"), list):
                parsed["key_points"] = []
            if not isinstance(parsed.get("action_items"), list):
                parsed["action_items"] = []
            
            return parsed
            
        except json.JSONDecodeError:
            return None
    
    def get_window_stats(self) -> dict:
        """Get current window statistics."""
        if not self._captions:
            return {"caption_count": 0, "window_ms": 0, "speakers": []}
        
        speakers = set(c.speaker_name for c in self._captions)
        return {
            "caption_count": len(self._captions),
            "window_ms": self._captions[-1].end_ms - self._captions[0].start_ms,
            "speakers": list(speakers),
            "oldest_ms": self._captions[0].start_ms,
            "newest_ms": self._captions[-1].end_ms,
        }
    
    def get_metrics(self) -> dict:
        """Get processing metrics."""
        avg_latency = self._total_latency_s / self._summary_count if self._summary_count > 0 else 0
        return {
            "summaries_generated": self._summary_count,
            "total_latency_s": round(self._total_latency_s, 2),
            "avg_latency_s": round(avg_latency, 2),
            "target_latency_s": self.config.latency_target_seconds,
            "errors": self._errors,
            "window_stats": self.get_window_stats(),
        }
    
    def clear_window(self) -> None:
        """Clear the rolling window."""
        self._captions.clear()
        self.logger.debug("Rolling window cleared")


def create_summarizer_from_config(config_dict: dict = None) -> Summarizer:
    """Factory to create Summarizer from config."""
    if config_dict is None:
        cfg = get_config()
        config_dict = {
            "window_minutes": cfg.summarization.window_minutes,
            "interval_seconds": cfg.summarization.interval_seconds,
            "max_summary_chars": cfg.summarization.max_summary_chars,
            "max_key_points": cfg.summarization.max_key_points,
            "max_key_point_chars": cfg.summarization.max_key_point_chars,
            "chunk_tokens": cfg.summarization.chunk_tokens,
            "llm": LLMConfig(
                model=cfg.summarization.llm.model,
                backend=cfg.summarization.llm.backend,
                host=cfg.summarization.llm.host,
                timeout_seconds=cfg.summarization.llm.timeout_seconds,
                temperature=cfg.summarization.llm.temperature,
                max_retries=cfg.summarization.llm.max_retries,
            ),
            "latency_target_seconds": cfg.summarization.latency_target_seconds,
        }
    elif "llm" in config_dict and isinstance(config_dict["llm"], dict):
        config_dict["llm"] = LLMConfig(**config_dict["llm"])
    return Summarizer(SummarizerConfig(**config_dict))