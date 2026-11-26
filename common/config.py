"""
Configuration loading for VoxLens.
Loads from config.yaml with environment variable overrides.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class NLPConfig(BaseModel):
    lookback_window: int = 3
    latency_target_ms: int = 300
    t5_model: str = "t5-small"
    use_t5: bool = True
    filler_words: list[str] = Field(default_factory=list)
    allowlisted_proper_nouns: list[str] = Field(default_factory=list)
    asr_corrections: dict[str, str] = Field(default_factory=dict)


class SummarizationLLMConfig(BaseModel):
    model: str = "gemma2:2b-instruct-q4_k_m"
    model_path: Optional[str] = None
    backend: str = "ollama"  # "ollama" or "llamacpp"
    host: str = "http://localhost:11434"
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_retries: int = 1


class SummarizationConfig(BaseModel):
    window_minutes: int = 15
    interval_seconds: int = 300
    max_summary_chars: int = 200
    max_key_points: int = 5
    max_key_point_chars: int = 60
    chunk_tokens: int = 2000
    llm: SummarizationLLMConfig = Field(default_factory=SummarizationLLMConfig)
    latency_target_seconds: int = 20


class PathShowerConfig(BaseModel):
    recalculation_threshold_m: float = 5.0
    turn_angle_thresholds: dict[str, list[float]] = Field(default_factory=dict)
    distance_rounding: int = 1
    max_next_label_chars: int = 20
    latency_target_ms: int = 1000


class DisplayConfig(BaseModel):
    width: int = 128
    height: int = 64
    chars_per_line: int = 20
    lines: int = 4


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file_rotation: str = "daily"
    max_files: int = 7


class Config(BaseSettings):
    """Main configuration model."""
    model_config = SettingsConfigDict(
        env_prefix="VOXLENS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    version: str = "1.0"
    language: str = "en"
    models_dir: str = "./models"
    map_file: str = "./data/sample_map.json"
    transcript_file: str = "./data/sample_transcript.json"
    log_dir: str = "./logs"

    nlp: NLPConfig = Field(default_factory=NLPConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    path_shower: PathShowerConfig = Field(default_factory=PathShowerConfig)
    speaker_palette: list[str] = Field(default_factory=list)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


_config_instance: Optional[Config] = None


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches standard locations.
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is not None:
        return _config_instance

    # Find config file
    if config_path is None:
        for path in [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
            Path.home() / ".voxlens" / "config.yaml",
        ]:
            if path.exists():
                config_path = path
                break
    
    if config_path is None:
        raise FileNotFoundError("config.yaml not found in working directory, project root, or ~/.voxlens/")

    config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    # Create config instance - pydantic-settings will also pick up env vars
    _config_instance = Config(**yaml_data)
    
    # Resolve relative paths relative to config file directory
    config_dir = config_path.parent
    _config_instance.models_dir = str((config_dir / _config_instance.models_dir).resolve())
    _config_instance.map_file = str((config_dir / _config_instance.map_file).resolve())
    _config_instance.transcript_file = str((config_dir / _config_instance.transcript_file).resolve())
    _config_instance.log_dir = str((config_dir / _config_instance.log_dir).resolve())
    
    return _config_instance


def get_config() -> Config:
    """Get the loaded config instance, loading if necessary."""
    if _config_instance is None:
        return load_config()
    return _config_instance