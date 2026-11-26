"""
LLM client for summarization.
Supports Ollama and provides a stub for testing.
"""

from __future__ import annotations
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


@dataclass
class LLMConfig:
    model: str = "gemma2:2b-instruct-q4_k_m"
    backend: str = "ollama"  # "ollama" or "llamacpp"
    host: str = "http://localhost:11434"
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_retries: int = 1


class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if client is ready."""
        pass


class OllamaClient(LLMClient):
    """Ollama API client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client: Optional[ollama.Client] = None
        self._loaded = False
        self.logger = logger.bind(module="summarization.ollama")
    
    def load(self) -> bool:
        """Initialize Ollama client."""
        if not OLLAMA_AVAILABLE:
            self.logger.error("ollama package not installed")
            return False
        
        try:
            self.client = ollama.Client(host=self.config.host)
            # Test connection
            self.client.list()
            self._loaded = True
            self.logger.info(f"Ollama client connected to {self.config.host}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def is_available(self) -> bool:
        return self._loaded and self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using Ollama."""
        if not self.is_available():
            if not self.load():
                raise RuntimeError("Ollama client not available")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": 512,  # Limit output tokens
                },
            )
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise


class StubLLMClient(LLMClient):
    """Stub LLM client for testing without a real model."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._loaded = True
        self.logger = logger.bind(module="summarization.stub")
        self.call_count = 0
        self.last_prompt = ""
    
    def is_available(self) -> bool:
        return True
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a fake but structured summary for testing."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Detect if this is a merge prompt (contains multiple summaries)
        if "MERGE" in prompt.upper() or "COMBINE" in prompt.upper():
            return self._generate_merge_response(prompt)
        
        # Regular chunk summary
        return self._generate_chunk_response(prompt)
    
    def _generate_chunk_response(self, prompt: str) -> str:
        """Generate a fake chunk summary."""
        # Extract speaker names from prompt for realism
        speakers = []
        for line in prompt.split("\n"):
            if ":" in line and not line.strip().startswith("{"):
                speaker = line.split(":")[0].strip()
                if speaker and speaker not in speakers:
                    speakers.append(speaker)
        
        speaker_str = ", ".join(speakers[:3]) if speakers else "participants"
        
        return json.dumps({
            "summary": f"Discussion between {speaker_str} about VoxLens system components.",
            "key_points": [
                f"{speakers[0] if speakers else 'Teacher'}: VoxLens uses ESP8266 and SSD1306 OLED",
                f"{speakers[1] if len(speakers) > 1 else 'Student'}: Raspberry Pi 5 and Jetson are hardware options",
                "Teacher: Gemma 2B model used for summarization",
            ],
            "action_items": [
                "Bring VoxLens glasses to class",
                "Review navigation module implementation",
            ],
        })
    
    def _generate_merge_response(self, prompt: str) -> str:
        """Generate a fake merged summary."""
        return json.dumps({
            "summary": "VoxLens team discussed hardware (ESP8266, Raspberry Pi 5, Jetson), software (Gemma 2B, T5-small), and navigation algorithms (Dijkstra, A*). Exam scheduled for Friday.",
            "key_points": [
                "Teacher: VoxLens uses ESP8266 display with SSD1306 OLED",
                "Student 1: Raspberry Pi 5 with 16GB RAM is primary target",
                "Teacher: Gemma 2B 4-bit quantized for local summarization",
                "Student 1: Dijkstra algorithm for routing, A* for larger maps",
                "Teacher: Exam on Friday January 15 at 2 PM",
            ],
            "action_items": [
                "Bring VoxLens glasses to class",
                "Review path shower module implementation",
            ],
        })


def create_llm_client(config_dict: dict) -> LLMClient:
    """Factory to create appropriate LLM client."""
    config = LLMConfig(
        model=config_dict.get("model", "gemma2:2b-instruct-q4_k_m"),
        backend=config_dict.get("backend", "ollama"),
        host=config_dict.get("host", "http://localhost:11434"),
        timeout_seconds=config_dict.get("timeout_seconds", 60),
        temperature=config_dict.get("temperature", 0.1),
        max_retries=config_dict.get("max_retries", 1),
    )
    
    if config.backend == "ollama":
        return OllamaClient(config)
    else:
        # For now, fallback to stub
        return StubLLMClient(config)