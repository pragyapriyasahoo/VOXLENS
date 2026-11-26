"""
NLP Clarification Simulator.
Replays a scripted transcript and processes through the clarifier.
"""

from __future__ import annotations
import json
import time
import argparse
from pathlib import Path
from loguru import logger

from .clarifier import Clarifier, create_clarifier_from_config
from common.config import load_config
from common.logging import setup_logging
from schema.messages import CaptionMessage


def load_transcript(filepath: str | Path) -> list[dict]:
    """Load transcript from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both array of segments and object with "segments" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "segments" in data:
        return data["segments"]
    else:
        raise ValueError("Transcript must be a list of segments or object with 'segments' key")


def run_simulator(
    transcript_path: str | Path,
    config_path: str | Path = None,
    output_json: bool = False,
    verbose: bool = False,
) -> list[CaptionMessage]:
    """
    Run the NLP clarification simulator.
    
    Args:
        transcript_path: Path to transcript JSON file
        config_path: Path to config.yaml
        output_json: If True, output JSON lines to stdout
        verbose: If True, log detailed info
        
    Returns:
        List of processed CaptionMessage objects
    """
    # Setup
    load_config(config_path)
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    logger.info("Starting NLP Clarification Simulator")
    
    # Load transcript
    segments = load_transcript(transcript_path)
    logger.info(f"Loaded {len(segments)} transcript segments")
    
    # Create clarifier
    clarifier = create_clarifier_from_config()
    
    # Process each segment
    results = []
    for i, seg in enumerate(segments):
        caption = clarifier.process_segment(
            text=seg["text"],
            speaker_id=seg.get("speaker_id", f"S{(i % 3) + 1}"),
            speaker_name=seg.get("speaker_name", f"Speaker {(i % 3) + 1}"),
            colour=seg.get("colour", "FF6B6B"),
            confidence=seg.get("confidence", 0.95),
            start_ms=seg.get("start_ms", i * 2000),
            end_ms=seg.get("end_ms", (i + 1) * 2000),
        )
        results.append(caption)
        
        # Print result
        if output_json:
            print(caption.to_json())
        else:
            print(f"\n--- Segment {i+1} ---")
            print(f"Speaker: {caption.speaker_name} ({caption.speaker_id})")
            print(f"Original:  {caption.text_original}")
            print(f"Clarified: {caption.text_clarified}")
            print(f"Confidence: {caption.confidence:.2f}")
            print(f"Time: {caption.start_ms}-{caption.end_ms}ms")
        
        # Small delay to simulate real-time
        time.sleep(0.05)
    
    # Print metrics
    metrics = clarifier.get_metrics()
    logger.info(f"Simulator complete. Metrics: {metrics}")
    
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VoxLens NLP Clarification Simulator")
    parser.add_argument(
        "--transcript", "-t",
        default="./data/sample_transcript.json",
        help="Path to transcript JSON file"
    )
    parser.add_argument(
        "--config", "-c",
        default="./config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON lines"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    args = parser.parse_args()
    
    run_simulator(
        transcript_path=args.transcript,
        config_path=args.config,
        output_json=args.json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()