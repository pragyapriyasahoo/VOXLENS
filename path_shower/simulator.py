"""
Path Shower Simulator.
Simulates a walk through the navigation graph and generates navigation cues.
"""

from __future__ import annotations
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional
from loguru import logger

from .graph import NavigationGraph, load_graph, create_sample_graph, save_graph
from .router import Router, RouterConfig, create_router_from_config
from .turn_detector import TurnDetector, TurnDetectorConfig, create_turn_detector_from_config
from common.config import load_config
from common.logging import setup_logging
from schema.messages import NavigationMessage, NavigationInstruction, create_navigation


def run_simulator(
    map_path: str | Path = None,
    config_path: str | Path = None,
    start_node: str = "entrance",
    dest_node: str = "room_101",
    walk_path: List[str] = None,
    output_json: bool = False,
    verbose: bool = False,
) -> List[NavigationMessage]:
    """
    Run the Path Shower simulator.
    
    Args:
        map_path: Path to graph JSON file (uses sample if None)
        config_path: Path to config.yaml
        start_node: Starting node ID
        dest_node: Destination node ID
        walk_path: List of node IDs to simulate walking through
        output_json: If True, output JSON lines to stdout
        verbose: If True, log detailed info
        
    Returns:
        List of generated NavigationMessage objects
    """
    # Setup
    load_config(config_path)
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    logger.info("Starting Path Shower Simulator")
    
    # Load or create graph
    if map_path and Path(map_path).exists():
        graph = load_graph(map_path)
    else:
        graph = create_sample_graph()
        if map_path:
            save_graph(graph, map_path)
            logger.info(f"Created sample graph at {map_path}")
    
    logger.info(f"Graph: {len(graph.nodes)} nodes, {len(graph._edge_lookup)} edges")
    
    # Create router and turn detector
    cfg = load_config(config_path)
    router = create_router_from_config(graph, {
        "algorithm": cfg.path_shower.turn_angle_thresholds.get("algorithm", "dijkstra"),
        "heuristic_weight": 1.0,
    })
    turn_detector = create_turn_detector_from_config(cfg.path_shower.turn_angle_thresholds)
    
    # Set destination
    route = router.set_destination(dest_node, start_node)
    if not route:
        logger.error(f"No route found from {start_node} to {dest_node}")
        return []
    
    logger.info(f"Route: {' -> '.join(n.id for n in route.nodes)} ({route.total_distance_m:.1f}m)")
    
    # Generate initial cues
    cues = turn_detector.generate_cues(route.nodes, route.edges, 0)
    logger.info(f"Generated {len(cues)} navigation cues")
    
    # Print initial route
    print(f"\n{'='*60}")
    print(f"ROUTE: {start_node} -> {dest_node}")
    print(f"{'='*60}")
    print(f"Total distance: {route.total_distance_m:.1f}m")
    print(f"Nodes: {' -> '.join(n.name for n in route.nodes)}")
    print()
    
    # Print all cues
    for i, cue in enumerate(cues):
        inst = cue["instruction"]
        dist = cue.get("distance_m", 0)
        label = cue.get("next_label", "")
        cumulative = cue.get("cumulative_distance_m", dist)
        
        if output_json:
            nav_msg = create_navigation(
                instruction=inst,
                distance_m=dist if inst != NavigationInstruction.ARRIVE else None,
                next_label=label if label else None,
            )
            print(nav_msg.to_json())
        else:
            if inst == NavigationInstruction.ARRIVE:
                print(f"  {i+1}. [ARRIVE] at destination")
            else:
                print(f"  {i+1}. {inst.value.upper():12} | {dist:3}m to {label} (cum: {cumulative}m)")
    
    # Simulate walking
    if walk_path:
        print(f"\n{'='*60}")
        print("SIMULATING WALK")
        print(f"{'='*60}")
        
        current_node = start_node
        current_index = 0
        messages = []
        
        for step, next_node_id in enumerate(walk_path):
            print(f"\nStep {step+1}: Moving to {next_node_id}")
            
            # Update position
            route, recalculated = router.update_position(next_node_id)
            current_node = next_node_id
            
            if recalculated:
                print(f"  [RECALCULATING] Off route detected!")
                cues = turn_detector.generate_cues(route.nodes, route.edges, 0)
                current_index = 0
            
            # Get next instruction
            next_instruction = router.get_next_instruction(current_node)
            if next_instruction:
                node, edge = next_instruction
                print(f"  Next: {edge.length_m:.0f}m to {route.nodes[router._next_node_index].name}")
            
            # Generate cue for this position
            cue = turn_detector.generate_cues(route.nodes, route.edges, router._next_node_index - 1)
            if cue:
                next_cue = cue[0]
                inst = next_cue["instruction"]
                dist = next_cue.get("distance_m", 0)
                label = next_cue.get("next_label", "")
                
                nav_msg = create_navigation(
                    instruction=inst,
                    distance_m=dist if inst != NavigationInstruction.ARRIVE else None,
                    next_label=label if label else None,
                )
                messages.append(nav_msg)
                
                if output_json:
                    print(nav_msg.to_json())
                else:
                    if inst == NavigationInstruction.ARRIVE:
                        print(f"  [ARRIVE] at destination!")
                    else:
                        print(f"  [NAV] {inst.value.upper():12} | {dist}m to {label}")
            
            # Check if arrived
            if router._next_node_index >= len(route.nodes):
                print(f"\n  [DESTINATION REACHED]")
                break
    
    # Metrics
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    
    return []


def main():
    parser = argparse.ArgumentParser(description="VoxLens Path Shower Simulator")
    parser.add_argument(
        "--map", "-m",
        default="./data/sample_map.json",
        help="Path to graph JSON file"
    )
    parser.add_argument(
        "--config", "-c",
        default="./config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--start", "-s",
        default="entrance",
        help="Start node ID"
    )
    parser.add_argument(
        "--dest", "-d",
        default="room_101",
        help="Destination node ID"
    )
    parser.add_argument(
        "--walk", "-w",
        nargs="+",
        help="Walk path as sequence of node IDs"
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
        map_path=args.map,
        config_path=args.config,
        start_node=args.start,
        dest_node=args.dest,
        walk_path=args.walk,
        output_json=args.json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()