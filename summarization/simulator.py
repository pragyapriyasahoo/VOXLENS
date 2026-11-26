"""
Summarization Simulator.
Feeds scripted captions to the summarizer and triggers summaries.
"""

from __future__ import annotations
import json
import argparse
import asyncio
from pathlib import Path
from loguru import logger

from .summarizer import Summarizer, create_summarizer_from_config
from .llm_client import StubLLMClient, LLMConfig
from common.config import load_config
from common.logging import setup_logging
from schema.messages import CaptionMessage, SummaryMessage


def load_transcript(filepath: str | Path) -> list[dict]:
    """Load transcript from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "segments" in data:
        return data["segments"]
    else:
        raise ValueError("Transcript must be a list of segments or object with 'segments' key")


async def run_simulator(
    transcript_path: str | Path,
    config_path: str | Path = None,
    use_stub: bool = True,
    output_json: bool = False,
    verbose: bool = False,
    trigger_interval: int = 5,  # Trigger summary every N segments
) -> list[SummaryMessage]:
    """
    Run the summarization simulator.
    
    Args:
        transcript_path: Path to transcript JSON file
        config_path: Path to config.yaml
        use_stub: If True, use StubLLMClient instead of real Ollama
        output_json: If True, output JSON lines to stdout
        verbose: If True, log detailed info
        trigger_interval: Generate summary every N segments
        
    Returns:
        List of generated SummaryMessage objects
    """
    # Setup
    load_config(config_path)
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    logger.info("Starting Summarization Simulator")
    
    # Load transcript
    segments = load_transcript(transcript_path)
    logger.info(f"Loaded {len(segments)} transcript segments")
    
    # Create summarizer
    if use_stub:
        llm_client = StubLLMClient(LLMConfig())
        summarizer = create_summarizer_from_config()
        summarizer.llm_client = llm_client
        logger.info("Using StubLLMClient for testing")
    else:
        summarizer = create_summarizer_from_config()
        logger.info("Using real LLM client (requires Ollama running)")
    
    # Track summaries
    summaries = []
    
    def on_summary(summary: SummaryMessage):
        summaries.append(summary)
        if output_json:
            print(summary.to_json())
        else:
            print(f"\n{'='*60}")
            print(f"SUMMARY ({summary.scope})")
            print(f"{'='*60}")
            print(f"Window: {summary.window_start_ms}ms - {summary.window_end_ms}ms")
            print(f"Summary: {summary.summary}")
            print(f"Key Points ({len(summary.key_points)}):")
            for i, kp in enumerate(summary.key_points, 1):
                print(f"  {i}. {kp}")
            if summary.action_items:
                print(f"Action Items ({len(summary.action_items)}):")
                for i, ai in enumerate(summary.action_items, 1):
                    print(f"  {i}. {ai}")
    
    summarizer.on_summary = on_summary
    
    # Process each segment as caption
    for i, seg in enumerate(segments):
        caption = CaptionMessage(
            text_original=seg["text"],
            text_clarified=seg["text"],  # Use raw for simulator
            speaker_id=seg.get("speaker_id", f"S{(i % 3) + 1}"),
            speaker_name=seg.get("speaker_name", f"Speaker {(i % 3) + 1}"),
            colour=seg.get("colour", "FF6B6B"),
            confidence=seg.get("confidence", 0.95),
            start_ms=seg.get("start_ms", i * 2000),
            end_ms=seg.get("end_ms", (i + 1) * 2000),
        )
        
        summarizer.add_caption(caption)
        
        # Trigger summary periodically
        if (i + 1) % trigger_interval == 0:
            logger.info(f"Triggering interval summary after segment {i+1}")
            summarizer.request_summary(scope="interval")
        
        # Small delay
        await asyncio.sleep(0.01)
    
    # Final summary
    logger.info("Triggering final on-demand summary")
    summarizer.request_summary(scope="on_demand")
    
    # Print metrics
    metrics = summarizer.get_metrics()
    logger.info(f"Simulator complete. Metrics: {metrics}")
    
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    for key, value in metrics.items():
        if key != "window_stats":
            print(f"  {key}: {value}")
    print(f"  window_stats: {metrics.get('window_stats', {})}")
    
    return summaries


def main():
    parser = argparse.ArgumentParser(description="VoxLens Summarization Simulator")
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
        "--stub", "-s",
        action="store_true",
        default=True,
        help="Use stub LLM client (default: True)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Ollama client (requires Ollama running)"
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
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Trigger summary every N segments"
    )
    args = parser.parse_args()
    
    use_stub = not args.real
    
    asyncio.run(run_simulator(
        transcript_path=args.transcript,
        config_path=args.config,
        use_stub=use_stub,
        output_json=args.json,
        verbose=args.verbose,
        trigger_interval=args.interval,
    ))


if __name__ == "__main__":
    main()