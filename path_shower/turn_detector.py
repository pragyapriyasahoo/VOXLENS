"""
Turn detection for Path Shower.
Computes heading changes at nodes and maps to navigation instructions.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
from loguru import logger

from .graph import GraphNode, GraphEdge
from schema.messages import NavigationInstruction


@dataclass
class TurnDetectorConfig:
    # Angle thresholds in degrees
    # Negative = left turn, Positive = right turn
    left_range: Tuple[float, float] = (-135, -45)
    right_range: Tuple[float, float] = (45, 135)
    u_turn_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((-180, -135), (135, 180))
    straight_range: Tuple[float, float] = (-45, 45)
    
    distance_rounding: int = 1  # Round to nearest metre
    max_next_label_chars: int = 20


class TurnDetector:
    """
    Detects turns from a sequence of nodes/edges.
    
    Computes heading change at each node and maps to:
    - left: -135 to -45 degrees
    - right: 45 to 135 degrees
    - u_turn: beyond 135 or below -135
    - straight: -45 to 45 degrees
    """
    
    def __init__(self, config: Optional[TurnDetectorConfig] = None):
        self.config = config or TurnDetectorConfig()
        self.logger = logger.bind(module="path_shower.turn_detector")
    
    def compute_heading(self, from_node: GraphNode, to_node: GraphNode) -> float:
        """
        Compute heading from one node to another in degrees.
        
        Returns:
            Heading in degrees: 0 = East, 90 = North, 180 = West, -90 = South
            (Standard mathematical convention: 0 at +X, counter-clockwise)
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        
        # atan2 returns radians, -pi to pi
        # Convert to degrees: 0 = East, 90 = North
        heading = math.degrees(math.atan2(dy, dx))
        return heading
    
    def compute_turn_angle(self, prev_heading: float, next_heading: float) -> float:
        """
        Compute signed turn angle from previous heading to next heading.
        
        In standard math coordinates (0=East, 90=North), a left turn (East to North)
        gives a positive angle change. But our config expects negative = left.
        So we negate the difference to match config convention.
        
        Returns:
            Angle in degrees: negative = left turn, positive = right turn
            Range: -180 to 180
        """
        diff = prev_heading - next_heading  # Negated to match config convention
        
        # Normalize to -180 to 180
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        
        return diff
    
    def classify_turn(self, turn_angle: float) -> NavigationInstruction:
        """
        Classify turn angle into navigation instruction.
        
        Args:
            turn_angle: Signed angle in degrees (positive = right)
            
        Returns:
            NavigationInstruction enum
        """
        left_min, left_max = self.config.left_range
        right_min, right_max = self.config.right_range
        straight_min, straight_max = self.config.straight_range
        
        # Check U-turn first (largest magnitude)
        for u_min, u_max in self.config.u_turn_range:
            if u_min <= turn_angle <= u_max:
                return NavigationInstruction.U_TURN
        
        # Check left
        if left_min <= turn_angle <= left_max:
            return NavigationInstruction.LEFT
        
        # Check right
        if right_min <= turn_angle <= right_max:
            return NavigationInstruction.RIGHT
        
        # Default to straight
        return NavigationInstruction.STRAIGHT
    
    def detect_turns(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
    ) -> List[dict]:
        """
        Detect turns along a route.
        
        Args:
            nodes: List of nodes in route order
            edges: List of edges connecting nodes (len = len(nodes) - 1)
            
        Returns:
            List of turn info dicts with instruction, distance, next_label
        """
        if len(nodes) < 2:
            return []
        
        if len(edges) != len(nodes) - 1:
            raise ValueError(f"Expected {len(nodes)-1} edges for {len(nodes)} nodes, got {len(edges)}")
        
        turns = []
        
        # Compute headings for each segment
        headings = []
        for i in range(len(nodes) - 1):
            heading = self.compute_heading(nodes[i], nodes[i + 1])
            headings.append(heading)
        
        # At each intermediate node, compute turn
        for i in range(1, len(nodes) - 1):
            prev_heading = headings[i - 1]
            next_heading = headings[i]
            
            turn_angle = self.compute_turn_angle(prev_heading, next_heading)
            instruction = self.classify_turn(turn_angle)
            
            # Distance to this turn from previous node
            distance = edges[i - 1].length_m
            
            # Next node label (truncated)
            next_label = nodes[i + 1].name
            if len(next_label) > self.config.max_next_label_chars:
                next_label = next_label[:self.config.max_next_label_chars]
            
            turns.append({
                "node": nodes[i],
                "instruction": instruction,
                "turn_angle": turn_angle,
                "distance_m": round(distance),
                "next_label": next_label,
                "prev_heading": prev_heading,
                "next_heading": next_heading,
            })
        
        # Add final "arrive" at destination
        if nodes:
            last_distance = edges[-1].length_m if edges else 0
            turns.append({
                "node": nodes[-1],
                "instruction": NavigationInstruction.ARRIVE,
                "turn_angle": 0.0,
                "distance_m": round(last_distance),
                "next_label": None,
                "prev_heading": headings[-1] if headings else 0,
                "next_heading": 0,
            })
        
        return turns
    
    def generate_cues(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        current_node_index: int = 0,
    ) -> List[dict]:
        """
        Generate navigation cues from current position.
        
        Args:
            nodes: Full route nodes
            edges: Full route edges
            current_node_index: Index of current node in route
            
        Returns:
            List of cue dicts for remaining route
        """
        if current_node_index >= len(nodes) - 1:
            return [{
                "instruction": NavigationInstruction.ARRIVE,
                "distance_m": 0,
                "next_label": None,
            }]
        
        # Get remaining route
        remaining_nodes = nodes[current_node_index:]
        remaining_edges = edges[current_node_index:]
        
        turns = self.detect_turns(remaining_nodes, remaining_edges)
        
        # Add cumulative distance from current position
        cumulative = 0
        for turn in turns:
            cumulative += turn["distance_m"]
            turn["cumulative_distance_m"] = cumulative
        
        return turns


def create_turn_detector_from_config(config_dict: dict) -> TurnDetector:
    """Factory to create TurnDetector from config."""
    detector_config = TurnDetectorConfig(
        left_range=tuple(config_dict.get("left", [-135, -45])),
        right_range=tuple(config_dict.get("right", [45, 135])),
        u_turn_range=(
            tuple(config_dict.get("u_turn", [-180, -135])),
            tuple(config_dict.get("u_turn_positive", [135, 180])),
        ),
        straight_range=tuple(config_dict.get("straight", [-45, 45])),
        distance_rounding=config_dict.get("distance_rounding", 1),
        max_next_label_chars=config_dict.get("max_next_label_chars", 20),
    )
    return TurnDetector(detector_config)