"""
Text chunking for summarization.
Splits long conversations into token-bounded chunks and merges summaries.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class ChunkerConfig:
    max_tokens: int = 2000
    overlap_tokens: int = 100
    # Approximate token estimation: 1 token ~= 4 chars for English
    chars_per_token: float = 4.0


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    token_estimate: int
    speaker_segments: List[dict]  # Original segments in this chunk


class TextChunker:
    """
    Splits conversation text into token-bounded chunks for LLM processing.
    
    Strategy:
    1. Group segments by speaker for coherent chunks
    2. Estimate tokens (roughly 4 chars/token for English)
    3. Split at natural boundaries (speaker changes, sentence ends)
    4. Add overlap for context continuity
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.logger = logger.bind(module="summarization.chunker")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return max(1, int(len(text) / self.config.chars_per_token))
    
    def chunk_segments(
        self,
        segments: List[dict],
        max_tokens: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Split segments into chunks.
        
        Args:
            segments: List of caption message dicts with text_clarified, speaker_name, etc.
            max_tokens: Override max tokens per chunk
            
        Returns:
            List of Chunk objects
        """
        max_tokens = max_tokens or self.config.max_tokens
        
        if not segments:
            return []
        
        # Build full text with speaker attribution
        full_text_parts = []
        segment_boundaries = []  # (start_char, end_char, segment_index)
        
        for i, seg in enumerate(segments):
            speaker = seg.get("speaker_name", seg.get("speaker_id", "Unknown"))
            text = seg.get("text_clarified", seg.get("text", ""))
            
            start = len("".join(full_text_parts))
            part = f"{speaker}: {text}\n"
            full_text_parts.append(part)
            end = len("".join(full_text_parts))
            segment_boundaries.append((start, end, i))
        
        full_text = "".join(full_text_parts)
        total_tokens = self.estimate_tokens(full_text)
        
        self.logger.debug(f"Total conversation: {len(segments)} segments, ~{total_tokens} tokens")
        
        if total_tokens <= max_tokens:
            return [Chunk(
                text=full_text,
                start_idx=0,
                end_idx=len(segments),
                token_estimate=total_tokens,
                speaker_segments=segments,
            )]
        
        # Split into chunks
        chunks = []
        current_start = 0
        current_segments = []
        current_start_idx = 0
        
        for start_char, end_char, seg_idx in segment_boundaries:
            # Check if adding this segment would exceed token limit
            candidate_text = full_text[current_start:end_char]
            candidate_tokens = self.estimate_tokens(candidate_text)
            
            if candidate_tokens > max_tokens and current_segments:
                # Finalize current chunk
                chunk_text = full_text[current_start:segment_boundaries[seg_idx - 1][1]]
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=current_start_idx,
                    end_idx=seg_idx,
                    token_estimate=self.estimate_tokens(chunk_text),
                    speaker_segments=current_segments.copy(),
                ))
                
                # Start new chunk with overlap
                overlap_start = max(0, seg_idx - self._calculate_overlap_segments(segments, max_tokens))
                current_start = segment_boundaries[overlap_start][0]
                current_segments = segments[overlap_start:seg_idx]
                current_start_idx = overlap_start
            
            # Add segment with index tracking
            if not current_segments:
                current_start_idx = seg_idx
            current_segments.append({**segments[seg_idx], "index": seg_idx})
        
        # Add final chunk
        if current_segments:
            chunk_text = full_text[current_start:]
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=current_start_idx,
                end_idx=len(segments),
                token_estimate=self.estimate_tokens(chunk_text),
                speaker_segments=current_segments,
            ))
        
        self.logger.info(f"Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"  Chunk {i}: ~{chunk.token_estimate} tokens, "
                             f"segments {chunk.start_idx}-{chunk.end_idx}")
        
        return chunks
    
    def _calculate_overlap_segments(self, segments: List[dict], max_tokens: int) -> int:
        """Calculate how many previous segments to include for overlap."""
        overlap_chars = int(self.config.overlap_tokens * self.config.chars_per_token)
        count = 0
        char_count = 0
        
        for seg in reversed(segments):
            text = seg.get("text_clarified", seg.get("text", ""))
            speaker = seg.get("speaker_name", seg.get("speaker_id", "Unknown"))
            char_count += len(f"{speaker}: {text}\n")
            count += 1
            if char_count >= overlap_chars:
                break
        
        return count
    
    def merge_summaries(self, chunk_summaries: List[dict]) -> dict:
        """
        Merge multiple chunk summaries into a final summary.
        
        Args:
            chunk_summaries: List of summary dicts with summary, key_points, action_items
            
        Returns:
            Merged summary dict
        """
        if not chunk_summaries:
            return {"summary": "", "key_points": [], "action_items": []}
        
        # Combine all key points and action items (even for single chunk)
        all_key_points = []
        all_action_items = []
        all_summaries = []
        
        for cs in chunk_summaries:
            all_key_points.extend(cs.get("key_points", []))
            all_action_items.extend(cs.get("action_items", []))
            all_summaries.append(cs.get("summary", ""))
        
        # Deduplicate key points (simple approach: keep first occurrence)
        seen_points = set()
        unique_key_points = []
        for point in all_key_points:
            key = point.lower().strip()
            if key not in seen_points:
                seen_points.add(key)
                unique_key_points.append(point)
        
        # Deduplicate action items
        seen_actions = set()
        unique_actions = []
        for action in all_action_items:
            key = action.lower().strip()
            if key not in seen_actions:
                seen_actions.add(key)
                unique_actions.append(action)
        
        # Create merged summary by combining chunk summaries
        merged_summary = " ".join(all_summaries)
        if len(merged_summary) > 200:
            merged_summary = merged_summary[:197] + "..."
        
        # Limit key points to 5
        unique_key_points = unique_key_points[:5]
        # Limit each key point to 60 chars
        unique_key_points = [p[:60] for p in unique_key_points]
        
        return {
            "summary": merged_summary,
            "key_points": unique_key_points,
            "action_items": unique_actions,
        }


def create_chunker_from_config(config_dict: dict) -> TextChunker:
    """Factory to create TextChunker from config."""
    chunker_config = ChunkerConfig(
        max_tokens=config_dict.get("chunk_tokens", 2000),
        overlap_tokens=config_dict.get("overlap_tokens", 100),
    )
    return TextChunker(chunker_config)