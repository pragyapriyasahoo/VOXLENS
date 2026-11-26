"""
Navigation graph data structures and loading.
Graph represents a building floor plan with nodes and walkable edges.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class GraphNode:
    """A node in the navigation graph (room, hallway intersection, landmark)."""
    id: str
    name: str
    x: float  # metres from building origin
    y: float  # metres from building origin
    node_type: str = "waypoint"  # "room", "hallway", "entrance", "waypoint", "exit"
    floor: int = 0
    metadata: dict = field(default_factory=dict)
    
    def position(self) -> Tuple[float, float]:
        """Get (x, y) position."""
        return (self.x, self.y)
    
    def distance_to(self, other: GraphNode) -> float:
        """Euclidean distance to another node."""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class GraphEdge:
    """A walkable edge between two nodes."""
    from_id: str
    to_id: str
    length_m: float
    walkable: bool = True
    edge_type: str = "corridor"  # "corridor", "door", "stairs", "ramp"
    metadata: dict = field(default_factory=dict)
    
    def is_bidirectional(self) -> bool:
        """Assume bidirectional unless specified."""
        return self.metadata.get("bidirectional", True)


class NavigationGraph:
    """
    Navigation graph for indoor routing.
    
    Nodes represent locations (rooms, intersections, landmarks).
    Edges represent walkable paths with distances.
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = {}  # adjacency list: from_id -> [edges]
        self._edge_lookup: Dict[Tuple[str, str], GraphEdge] = {}
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists, overwriting")
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if edge.from_id not in self.nodes:
            raise ValueError(f"From node {edge.from_id} does not exist")
        if edge.to_id not in self.nodes:
            raise ValueError(f"To node {edge.to_id} does not exist")
        
        self.edges.setdefault(edge.from_id, []).append(edge)
        self._edge_lookup[(edge.from_id, edge.to_id)] = edge
        
        # Add reverse edge if bidirectional
        if edge.is_bidirectional():
            reverse_edge = GraphEdge(
                from_id=edge.to_id,
                to_id=edge.from_id,
                length_m=edge.length_m,
                walkable=edge.walkable,
                edge_type=edge.edge_type,
                metadata=edge.metadata.copy(),
            )
            self.edges.setdefault(edge.to_id, []).append(reverse_edge)
            self._edge_lookup[(edge.to_id, edge.from_id)] = reverse_edge
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, from_id: str, to_id: str) -> Optional[GraphEdge]:
        """Get edge between two nodes."""
        return self._edge_lookup.get((from_id, to_id))
    
    def get_neighbors(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get walkable neighbors of a node."""
        if node_id not in self.edges:
            return []
        
        neighbors = []
        for edge in self.edges[node_id]:
            if edge.walkable:
                node = self.nodes.get(edge.to_id)
                if node:
                    neighbors.append((node, edge))
        return neighbors
    
    def validate(self) -> List[str]:
        """
        Validate graph consistency.
        
        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []
        
        # Check all edge endpoints exist
        for (from_id, to_id), edge in self._edge_lookup.items():
            if from_id not in self.nodes:
                issues.append(f"Edge references missing from_node: {from_id}")
            if to_id not in self.nodes:
                issues.append(f"Edge references missing to_node: {to_id}")
        
        # Check for isolated nodes (no walkable edges)
        for node_id, node in self.nodes.items():
            edges = self.edges.get(node_id, [])
            walkable_edges = [e for e in edges if e.walkable]
            if not walkable_edges:
                issues.append(f"Node {node_id} ({node.name}) has no walkable edges")
        
        # Check for negative lengths
        for edge in self._edge_lookup.values():
            if edge.length_m < 0:
                issues.append(f"Edge {edge.from_id}->{edge.to_id} has negative length")
        
        return issues
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get graph bounding box (min_x, min_y, max_x, max_y)."""
        if not self.nodes:
            return (0, 0, 0, 0)
        
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes


def load_graph(filepath: str | Path) -> NavigationGraph:
    """
    Load navigation graph from JSON file.
    
    Expected format:
    {
        "nodes": [
            {"id": "n1", "name": "Entrance", "x": 0, "y": 0, "node_type": "entrance", "floor": 0},
            ...
        ],
        "edges": [
            {"from": "n1", "to": "n2", "length_m": 5.0, "walkable": true, "edge_type": "corridor"},
            ...
        ]
    }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    graph = NavigationGraph()
    
    # Load nodes
    for node_data in data.get("nodes", []):
        node = GraphNode(
            id=node_data["id"],
            name=node_data["name"],
            x=node_data["x"],
            y=node_data["y"],
            node_type=node_data.get("node_type", "waypoint"),
            floor=node_data.get("floor", 0),
            metadata=node_data.get("metadata", {}),
        )
        graph.add_node(node)
    
    # Load edges
    for edge_data in data.get("edges", []):
        edge = GraphEdge(
            from_id=edge_data["from"],
            to_id=edge_data["to"],
            length_m=edge_data["length_m"],
            walkable=edge_data.get("walkable", True),
            edge_type=edge_data.get("edge_type", "corridor"),
            metadata=edge_data.get("metadata", {}),
        )
        graph.add_edge(edge)
    
    # Validate
    issues = graph.validate()
    for issue in issues:
        logger.warning(f"Graph validation: {issue}")
    
    logger.info(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph._edge_lookup)} edges")
    return graph


def create_sample_graph() -> NavigationGraph:
    """Create a sample building graph for testing."""
    graph = NavigationGraph()
    
    # Nodes: simple office floor plan
    nodes = [
        GraphNode("entrance", "Main Entrance", 0, 0, "entrance"),
        GraphNode("lobby", "Lobby", 5, 0, "hallway"),
        GraphNode("hallway_n", "North Hallway", 5, 10, "hallway"),
        GraphNode("hallway_s", "South Hallway", 5, -10, "hallway"),
        GraphNode("room_101", "Room 101", 15, 10, "room"),
        GraphNode("room_102", "Room 102", 15, 5, "room"),
        GraphNode("room_103", "Room 103", 15, 0, "room"),
        GraphNode("room_104", "Room 104", 15, -5, "room"),
        GraphNode("room_105", "Room 105", 15, -10, "room"),
        GraphNode("restroom", "Restroom", 10, -10, "room"),
        GraphNode("stairs", "Stairs to Floor 2", 0, 10, "stairs"),
        GraphNode("exit", "Emergency Exit", 0, -10, "exit"),
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Edges: connections
    edges = [
        GraphEdge("entrance", "lobby", 5.0),
        GraphEdge("lobby", "hallway_n", 10.0),
        GraphEdge("lobby", "hallway_s", 10.0),
        GraphEdge("lobby", "room_103", 10.0),
        GraphEdge("hallway_n", "room_101", 10.0),
        GraphEdge("hallway_n", "room_102", 5.0),
        GraphEdge("hallway_n", "stairs", 5.0),
        GraphEdge("hallway_s", "room_104", 5.0),
        GraphEdge("hallway_s", "room_105", 10.0),
        GraphEdge("hallway_s", "restroom", 5.0),
        GraphEdge("hallway_s", "exit", 10.0),
        # Cross connections
        GraphEdge("room_101", "room_102", 5.0),
        GraphEdge("room_102", "room_103", 5.0),
        GraphEdge("room_103", "room_104", 5.0),
        GraphEdge("room_104", "room_105", 5.0),
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph


def save_graph(graph: NavigationGraph, filepath: str | Path) -> None:
    """Save graph to JSON file."""
    data = {
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "x": node.x,
                "y": node.y,
                "node_type": node.node_type,
                "floor": node.floor,
                "metadata": node.metadata,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "from": edge.from_id,
                "to": edge.to_id,
                "length_m": edge.length_m,
                "walkable": edge.walkable,
                "edge_type": edge.edge_type,
                "metadata": edge.metadata,
            }
            for edge in graph._edge_lookup.values()
            # Only save one direction for bidirectional edges
            if edge.from_id < edge.to_id or not edge.is_bidirectional()
        ],
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)