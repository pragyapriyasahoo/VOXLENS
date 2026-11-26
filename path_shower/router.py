"""
Routing algorithms for Path Shower.
Implements Dijkstra and A* for finding shortest paths.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from loguru import logger

from .graph import NavigationGraph, GraphNode, GraphEdge


@dataclass
class RouterConfig:
    algorithm: str = "dijkstra"  # "dijkstra" or "astar"
    heuristic_weight: float = 1.0  # For A*, weight of heuristic


@dataclass
class Route:
    """A computed route from start to destination."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_distance_m: float
    start_node: GraphNode
    end_node: GraphNode
    
    def get_instructions(self) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get list of (node, edge_to_next) for turn detection."""
        instructions = []
        for i in range(len(self.nodes) - 1):
            node = self.nodes[i]
            edge = self.edges[i]
            instructions.append((node, edge))
        # Add final node with no outgoing edge
        if self.nodes:
            instructions.append((self.nodes[-1], None))
        return instructions


class Router:
    """
    Path router using Dijkstra or A* algorithm.
    
    Recalculates when user deviates from planned path.
    """
    
    def __init__(self, graph: NavigationGraph, config: Optional[RouterConfig] = None):
        self.graph = graph
        self.config = config or RouterConfig()
        self.logger = logger.bind(module="path_shower.router")
        
        # Current route state
        self._current_route: Optional[Route] = None
        self._destination_id: Optional[str] = None
        self._next_node_index: int = 0  # Index in route.nodes of next target
    
    def find_path(
        self,
        start_id: str,
        dest_id: str,
        algorithm: Optional[str] = None,
    ) -> Optional[Route]:
        """
        Find shortest path from start to destination.
        
        Args:
            start_id: Starting node ID
            dest_id: Destination node ID
            algorithm: Override algorithm ("dijkstra" or "astar")
            
        Returns:
            Route object or None if no path exists
        """
        start_node = self.graph.get_node(start_id)
        dest_node = self.graph.get_node(dest_id)
        
        if not start_node:
            raise ValueError(f"Start node not found: {start_id}")
        if not dest_node:
            raise ValueError(f"Destination node not found: {dest_id}")
        
        algo = algorithm or self.config.algorithm
        
        if algo == "astar":
            path_nodes, path_edges, distance = self._astar(start_node, dest_node)
        else:
            path_nodes, path_edges, distance = self._dijkstra(start_node, dest_node)
        
        if not path_nodes:
            self.logger.warning(f"No path found from {start_id} to {dest_id}")
            return None
        
        route = Route(
            nodes=path_nodes,
            edges=path_edges,
            total_distance_m=distance,
            start_node=start_node,
            end_node=dest_node,
        )
        
        self.logger.info(f"Route found: {start_id} -> {dest_id}, "
                         f"{len(path_nodes)} nodes, {distance:.1f}m")
        return route
    
    def _dijkstra(
        self,
        start: GraphNode,
        dest: GraphNode,
    ) -> Tuple[List[GraphNode], List[GraphEdge], float]:
        """Dijkstra's algorithm for shortest path."""
        # Priority queue: (distance, node_id)
        pq = [(0.0, start.id)]
        distances: Dict[str, float] = {start.id: 0.0}
        previous: Dict[str, Tuple[str, GraphEdge]] = {}  # node_id -> (prev_node_id, edge)
        visited: Set[str] = set()
        
        while pq:
            dist, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id == dest.id:
                break
            
            current_node = self.graph.get_node(current_id)
            if not current_node:
                continue
            
            for neighbor, edge in self.graph.get_neighbors(current_id):
                if not edge.walkable:
                    continue
                
                new_dist = dist + edge.length_m
                
                if neighbor.id not in distances or new_dist < distances[neighbor.id]:
                    distances[neighbor.id] = new_dist
                    previous[neighbor.id] = (current_id, edge)
                    heapq.heappush(pq, (new_dist, neighbor.id))
        
        # Reconstruct path
        if dest.id not in previous and start.id != dest.id:
            return [], [], 0.0
        
        path_nodes = []
        path_edges = []
        current = dest.id
        
        while current != start.id:
            path_nodes.append(self.graph.get_node(current))
            prev_id, edge = previous[current]
            path_edges.append(edge)
            current = prev_id
        
        path_nodes.append(start)
        path_nodes.reverse()
        path_edges.reverse()
        
        total_distance = distances.get(dest.id, 0.0)
        return path_nodes, path_edges, total_distance
    
    def _astar(
        self,
        start: GraphNode,
        dest: GraphNode,
    ) -> Tuple[List[GraphNode], List[GraphEdge], float]:
        """A* algorithm with Euclidean distance heuristic."""
        def heuristic(node: GraphNode) -> float:
            return node.distance_to(dest) * self.config.heuristic_weight
        
        # Priority queue: (f_score, g_score, node_id)
        pq = [(heuristic(start), 0.0, start.id)]
        g_scores: Dict[str, float] = {start.id: 0.0}
        previous: Dict[str, Tuple[str, GraphEdge]] = {}
        visited: Set[str] = set()
        
        while pq:
            f_score, g_score, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id == dest.id:
                break
            
            current_node = self.graph.get_node(current_id)
            if not current_node:
                continue
            
            for neighbor, edge in self.graph.get_neighbors(current_id):
                if not edge.walkable:
                    continue
                
                tentative_g = g_score + edge.length_m
                
                if neighbor.id not in g_scores or tentative_g < g_scores[neighbor.id]:
                    g_scores[neighbor.id] = tentative_g
                    previous[neighbor.id] = (current_id, edge)
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(pq, (f, tentative_g, neighbor.id))
        
        # Reconstruct path
        if dest.id not in previous and start.id != dest.id:
            return [], [], 0.0
        
        path_nodes = []
        path_edges = []
        current = dest.id
        
        while current != start.id:
            path_nodes.append(self.graph.get_node(current))
            prev_id, edge = previous[current]
            path_edges.append(edge)
            current = prev_id
        
        path_nodes.append(start)
        path_nodes.reverse()
        path_edges.reverse()
        
        total_distance = g_scores.get(dest.id, 0.0)
        return path_nodes, path_edges, total_distance
    
    def set_destination(self, dest_id: str, current_position_id: str) -> Optional[Route]:
        """Set a new destination and compute initial route."""
        self._destination_id = dest_id
        route = self.find_path(current_position_id, dest_id)
        if route:
            self._current_route = route
            self._next_node_index = 1  # Next node after current position
        return route
    
    def update_position(self, current_node_id: str) -> Tuple[Optional[Route], bool]:
        """
        Update user position and check if recalculation needed.
        
        Returns:
            (route, recalculated) - route is current/updated route, 
            recalculated is True if route was recomputed
        """
        if not self._current_route or not self._destination_id:
            return None, False
        
        current_node = self.graph.get_node(current_node_id)
        if not current_node:
            return self._current_route, False
        
        # Check if we've reached the next expected node
        if self._next_node_index < len(self._current_route.nodes):
            expected_node = self._current_route.nodes[self._next_node_index]
            
            if current_node_id == expected_node.id:
                # Reached next node, advance
                self._next_node_index += 1
                self.logger.debug(f"Reached node {current_node_id}, next index: {self._next_node_index}")
                
                # Check if destination reached
                if self._next_node_index >= len(self._current_route.nodes):
                    self.logger.info(f"Destination {self._destination_id} reached!")
                    return self._current_route, False
        
        # Check if off-route (current node not on remaining path and not the current position)
        remaining_nodes = {n.id for n in self._current_route.nodes[self._next_node_index:]}
        # Also allow the current position node (previous node in route)
        current_position_node = self._current_route.nodes[self._next_node_index - 1] if self._next_node_index > 0 else None
        current_position_id = current_position_node.id if current_position_node else None
        
        if (current_node_id not in remaining_nodes 
            and current_node_id != self._current_route.nodes[-1].id
            and current_node_id != current_position_id):
            # Off route - recalculate
            self.logger.info(f"Off route at {current_node_id}, recalculating...")
            new_route = self.find_path(current_node_id, self._destination_id)
            if new_route:
                self._current_route = new_route
                self._next_node_index = 1
                return new_route, True
        
        return self._current_route, False
    
    def get_next_instruction(self, current_node_id: str) -> Optional[Tuple[GraphNode, GraphEdge]]:
        """
        Get the next navigation instruction (node and edge to take).
        
        Returns:
            (current_node, edge_to_next) or None if at destination
        """
        if not self._current_route:
            return None
        
        if self._next_node_index >= len(self._current_route.nodes):
            return None
        
        current_node = self._current_route.nodes[self._next_node_index - 1]
        next_node = self._current_route.nodes[self._next_node_index]
        
        edge = self.graph.get_edge(current_node.id, next_node.id)
        if edge:
            return (current_node, edge)
        
        return None
    
    def get_remaining_distance(self) -> float:
        """Get remaining distance to destination."""
        if not self._current_route or self._next_node_index >= len(self._current_route.nodes):
            return 0.0
        
        distance = 0.0
        for i in range(self._next_node_index - 1, len(self._current_route.edges)):
            distance += self._current_route.edges[i].length_m
        return distance
    
    def get_current_route(self) -> Optional[Route]:
        """Get the current active route."""
        return self._current_route
    
    def clear_route(self) -> None:
        """Clear current route."""
        self._current_route = None
        self._destination_id = None
        self._next_node_index = 0


def create_router_from_config(graph: NavigationGraph, config_dict: dict) -> Router:
    """Factory to create Router from config."""
    router_config = RouterConfig(
        algorithm=config_dict.get("algorithm", "dijkstra"),
        heuristic_weight=config_dict.get("heuristic_weight", 1.0),
    )
    return Router(graph, router_config)