"""
T5-small wrapper for punctuation restoration and capitalization.
Uses HuggingFace transformers with CPU optimization.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List
from loguru import logger

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    T5ForConditionalGeneration = None
    T5Tokenizer = None


@dataclass
class T5Config:
    model_name: str = "t5-small"
    device: str = "cpu"
    max_length: int = 128
    num_beams: int = 1  # Greedy decoding for speed
    do_sample: bool = False


class T5Punctuator:
    """
    T5-based punctuation and capitalization restoration.
    
    Uses a fine-tuned T5 model for punctuation restoration.
    Falls back to rule-based if model unavailable.
    """
    
    # Prompt template for punctuation restoration
    PROMPT_TEMPLATE = "restore punctuation: {text}"
    
    def __init__(self, config: T5Config):
        self.config = config
        self.model: Optional[any] = None
        self.tokenizer: Optional[any] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        
        if not TRANSFORMERS_AVAILABLE:
            self._load_error = "transformers/torch not installed"
            logger.warning("T5Punctuator: transformers not available, will use fallback")
    
    def load(self) -> bool:
        """Load the T5 model and tokenizer."""
        if self._loaded:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            logger.info(f"Loading T5 model: {self.config.model_name}")
            start = time.time()
            
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.config.device)
            self.model.eval()
            
            # Warm up
            _ = self._generate("hello world")
            
            load_time = time.time() - start
            self._loaded = True
            logger.info(f"T5 model loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load T5 model: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded and self.model is not None
    
    def restore(self, text: str, context: Optional[List[str]] = None) -> str:
        """
        Restore punctuation and capitalization.
        
        Args:
            text: Text to process
            context: Previous segments for context (prepended to input)
            
        Returns:
            Text with restored punctuation and capitalization
        """
        if not text or not text.strip():
            return text
        
        if not self.is_available():
            if not self.load():
                return self._fallback_restore(text)
        
        try:
            # Build input with context
            input_text = text
            if context:
                # Use last 2 context segments
                ctx = " ".join(context[-2:])
                input_text = f"{ctx} {text}"
            
            # Apply prompt template
            prompt = self.PROMPT_TEMPLATE.format(text=input_text)
            
            result = self._generate(prompt)
            
            # Extract only the new text part (after context)
            if context and result.startswith(ctx):
                result = result[len(ctx):].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"T5 generation failed: {e}")
            return self._fallback_restore(text)
    
    def _generate(self, prompt: str) -> str:
        """Run model generation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                do_sample=self.config.do_sample,
                early_stopping=True,
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()
    
    def _fallback_restore(self, text: str) -> str:
        """Rule-based fallback for punctuation and capitalization."""
        if not text:
            return text
        
        # Basic sentence splitting on conjunctions and clause boundaries
        # This is a minimal fallback - not as good as T5
        text = text.strip()
        
        # Capitalize first letter
        for i, ch in enumerate(text):
            if ch.isalpha():
                text = text[:i] + ch.upper() + text[i+1:]
                break
        
        # Add period at end if missing
        if text and text[-1] not in ".!?":
            text += "."
        
        # Simple heuristic: capitalize after period
        parts = re.split(r"([.!?]\s+)", text)
        result = []
        capitalize_next = True
        for part in parts:
            if capitalize_next and part:
                for i, ch in enumerate(part):
                    if ch.isalpha():
                        part = part[:i] + ch.upper() + part[i+1:]
                        break
                capitalize_next = False
            result.append(part)
            if re.match(r"[.!?]\s+", part):
                capitalize_next = True
        
        return "".join(result)


import re  # Moved here to avoid circular import issue


def create_punctuator_from_config(config_dict: dict) -> T5Punctuator:
    """Factory to create T5Punctuator from config dictionary."""
    t5_config = T5Config(
        model_name=config_dict.get("t5_model", "t5-small"),
        device=config_dict.get("device", "cpu"),
    )
    return T5Punctuator(t5_config)