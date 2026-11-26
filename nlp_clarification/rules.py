"""
Rule-based text cleanup for NLP clarification.
Handles filler words, repeated words, number/date normalization, and ASR corrections.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from loguru import logger


@dataclass
class CleanerConfig:
    filler_words: List[str] = field(default_factory=list)
    allowlisted_proper_nouns: List[str] = field(default_factory=list)
    asr_corrections: Dict[str, str] = field(default_factory=dict)


class TextCleaner:
    """
    Rule-based text cleanup pipeline.
    
    Processing order:
    1. ASR corrections (case-insensitive exact phrase replacement)
    2. Filler word removal
    3. Repeated word collapse
    4. Number normalization
    5. Date normalization
    6. Punctuation and capitalization (handled by T5 model)
    """
    
    # Compiled regex patterns
    REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
    MULTIPLE_SPACES = re.compile(r"\s+")
    SENTENCE_END = re.compile(r"[.!?]+")
    
    # Number words to digits
    NUMBER_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    }
    
    # Ordinal words
    ORDINAL_WORDS = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
        "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    }
    
    def __init__(self, config: CleanerConfig):
        self.config = config
        self._filler_pattern = self._build_filler_pattern()
        self._allowlist_set = set(w.lower() for w in config.allowlisted_proper_nouns)
        self._asr_patterns = self._build_asr_patterns()
        logger.debug(f"TextCleaner initialized with {len(config.filler_words)} filler words, "
                     f"{len(config.allowlisted_proper_nouns)} allowlisted nouns, "
                     f"{len(config.asr_corrections)} ASR corrections")
    
    def _build_filler_pattern(self) -> re.Pattern:
        """Build regex pattern for filler word removal."""
        if not self.config.filler_words:
            return re.compile(r"(?!)$")  # Never matches
        
        # Escape and join filler words
        escaped = [re.escape(w) for w in self.config.filler_words]
        # Match as whole words, case-insensitive
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
    
    def _build_asr_patterns(self) -> List[tuple[re.Pattern, str]]:
        """Build compiled regex patterns for ASR corrections."""
        patterns = []
        for incorrect, correct in self.config.asr_corrections.items():
            # Match as whole phrase, case-insensitive
            pattern = re.compile(r"\b" + re.escape(incorrect) + r"\b", re.IGNORECASE)
            patterns.append((pattern, correct))
        return patterns
    
    def clean(self, text: str, context: Optional[List[str]] = None) -> str:
        """
        Apply all cleanup rules to text.
        
        Args:
            text: Raw text to clean
            context: Previous segments for context (unused in rule-based, for future use)
            
        Returns:
            Cleaned text (without capitalization - that's handled by T5)
        """
        if not text or not text.strip():
            return text
        
        original = text
        
        # 1. ASR corrections (protect allowlisted nouns)
        text = self._apply_asr_corrections(text)
        
        # 2. Remove filler words
        text = self._remove_fillers(text)
        
        # 3. Collapse repeated words
        text = self._collapse_repeated(text)
        
        # 4. Normalize numbers
        text = self._normalize_numbers(text)
        
        # 5. Normalize dates
        text = self._normalize_dates(text)
        
        # 6. Clean up spacing
        text = self.MULTIPLE_SPACES.sub(" ", text).strip()
        
        # NOTE: Capitalization is now handled by T5 model, not here
        # This keeps the rule-based cleaner focused on structural cleanup
        
        if text != original:
            logger.debug(f"Cleaned: '{original}' -> '{text}'")
        
        return text
    
    def _apply_asr_corrections(self, text: str) -> str:
        """Apply ASR correction mappings, skipping allowlisted nouns."""
        for pattern, correct in self._asr_patterns:
            def replace_fn(match):
                matched = match.group(0)
                # Check if matched text (lowercase) is in allowlist
                if matched.lower() in self._allowlist_set:
                    return matched
                # Use correction as-is (with proper casing from config)
                # Only adjust case if matched is ALL UPPER or ALL LOWER
                if matched.isupper():
                    return correct.upper()
                elif matched.islower():
                    return correct  # Use config casing (e.g., "VoxLens")
                else:
                    # Mixed case - use config casing
                    return correct
            
            text = pattern.sub(replace_fn, text)
        return text
    
    def _remove_fillers(self, text: str) -> str:
        """Remove filler words."""
        # Replace filler words with empty string, then clean up spaces
        text = self._filler_pattern.sub("", text)
        # Clean up spaces around punctuation
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])\s+", r"\1 ", text)
        return text
    
    def _collapse_repeated(self, text: str) -> str:
        """Collapse repeated consecutive words (e.g., 'the the' -> 'the')."""
        def replace_fn(match):
            return match.group(1)
        return self.REPEATED_WORD_PATTERN.sub(replace_fn, text)
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert number words to digits where appropriate."""
        # Handle simple cardinal numbers
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            word = words[i].lower().rstrip(",.!?;:")
            punct = words[i][len(word):] if len(word) < len(words[i]) else ""
            
            if word in self.NUMBER_WORDS:
                # Check next word for compound numbers (e.g., "twenty one")
                if i + 1 < len(words):
                    next_word = words[i + 1].lower().rstrip(",.!?;:")
                    if word in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"] \
                       and next_word in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]:
                        tens = self.NUMBER_WORDS[word]
                        ones = self.NUMBER_WORDS[next_word]
                        result.append(str(int(tens) + int(ones)) + punct)
                        i += 2
                        continue
                
                result.append(self.NUMBER_WORDS[word] + punct)
            elif word in self.ORDINAL_WORDS:
                result.append(self.ORDINAL_WORDS[word] + punct)
            else:
                result.append(words[i])
            i += 1
        
        return " ".join(result)
    
    def _normalize_dates(self, text: str) -> str:
        """Normalize common date formats.
        
        Supports "Day Month" format (e.g., "15 January" -> "1/15").
        "Month Day" format (e.g., "January 15") is not supported in rule-based
        cleaner due to ambiguity with adjacent dates; use T5 for that.
        """
        # Month names to numbers
        months = {
            "january": "1", "february": "2", "march": "3", "april": "4",
            "may": "5", "june": "6", "july": "7", "august": "8",
            "september": "9", "october": "10", "november": "11", "december": "12",
            "jan": "1", "feb": "2", "mar": "3", "apr": "4", "jun": "6",
            "jul": "7", "aug": "8", "sep": "9", "sept": "9", "oct": "10", "nov": "11", "dec": "12",
        }
        
        # Build regex for "Day Month" format only (e.g., "15 January" -> "1/15")
        month_names = "|".join(re.escape(m) for m in months.keys())
        pattern_day_month = re.compile(rf"\b(\d{{1,2}})\s+({month_names})\b", re.IGNORECASE)
        
        def replace_day_month(match):
            day = match.group(1)
            month_name = match.group(2).lower()
            month_num = months.get(month_name, month_name)
            return f"{month_num}/{day}"
        
        # Apply pattern
        text = pattern_day_month.sub(replace_day_month, text)
        
        # "Monday" -> keep as is (day names)
        # "today", "tomorrow", "yesterday" -> keep as is
        
        return text
    
    def _capitalize_first(self, text: str) -> str:
        """Capitalize first character of text."""
        if not text:
            return text
        # Find first alphabetic character
        for i, ch in enumerate(text):
            if ch.isalpha():
                return text[:i] + ch.upper() + text[i+1:]
        return text


def create_cleaner_from_config(config_dict: dict) -> TextCleaner:
    """Factory to create TextCleaner from config dictionary."""
    cleaner_config = CleanerConfig(
        filler_words=config_dict.get("filler_words", []),
        allowlisted_proper_nouns=config_dict.get("allowlisted_proper_nouns", []),
        asr_corrections=config_dict.get("asr_corrections", {}),
    )
    return TextCleaner(cleaner_config)