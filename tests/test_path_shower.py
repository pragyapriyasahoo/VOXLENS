"""
Unit tests for Path Shower module.
Tests route calculation, turn detection, and recalculation on off-route position.
"""

import pytest
from path_shower.graph import NavigationGraph, GraphNode, GraphEdge, load_graph, create_sample_graph
from path_shower.router import Router, RouterConfig, Route
from path_shower.turn_detector import TurnDetector, TurnDetectorConfig, NavigationInstruction
from schema.messages import NavigationMessage, NavigationInstruction as SchemaNavigationInstruction, create_navigation


class TestNavigationGraph:
    """Tests for navigation graph."""
    
    @pytest.fixture
    def sample_graph(self):
        return create_sample_graph()
    
    def test_graph_creation(self, sample_graph):
        """Test sample graph has expected nodes and edges."""
        assert len(sample_graph.nodes) == 12
        # Each undirected edge is stored in both directions (from->to and to->from)
        # 15 undirected edges = 30 directed edges in _edge_lookup
        assert len(sample_graph._edge_lookup) == 30
        assert "entrance" in sample_graph
        assert "room_101" in sample_graph
    
    def test_node_lookup(self, sample_graph):
        """Test node retrieval."""
        node = sample_graph.get_node("room_101")
        assert node is not None
        assert node.name == "Room 101"
        assert node.x == 15
        assert node.y == 10
    
    def test_edge_lookup(self, sample_graph):
        """Test edge retrieval."""
        edge = sample_graph.get_edge("entrance", "lobby")
        assert edge is not None
        assert edge.length_m == 5.0
        assert edge.walkable is True
    
    def test_neighbors(self, sample_graph):
        """Test getting walkable neighbors."""
        neighbors = sample_graph.get_neighbors("lobby")
        neighbor_ids = {n.id for n, e in neighbors}
        assert "entrance" in neighbor_ids
        assert "hallway_n" in neighbor_ids
        assert "hallway_s" in neighbor_ids
        assert "room_103" in neighbor_ids
    
    def test_distance_calculation(self, sample_graph):
        """Test node distance calculation."""
        n1 = sample_graph.get_node("entrance")
        n2 = sample_graph.get_node("lobby")
        dist = n1.distance_to(n2)
        assert dist == 5.0  # (5,0) from (0,0)
    
    def test_graph_validation(self, sample_graph):
        """Test graph validation passes."""
        issues = sample_graph.validate()
        # Should have no issues for valid sample graph
        assert len(issues) == 0
    
    def test_bounds(self, sample_graph):
        """Test graph bounding box."""
        min_x, min_y, max_x, max_y = sample_graph.get_bounds()
        assert min_x == 0
        assert min_y == -10
        assert max_x == 15
        assert max_y == 10


class TestRouter:
    """Tests for routing algorithms."""
    
    @pytest.fixture
    def graph(self):
        return create_sample_graph()
    
    @pytest.fixture
    def router(self, graph):
        return Router(graph, RouterConfig(algorithm="dijkstra"))
    
    def test_dijkstra_simple_route(self, router):
        """Test Dijkstra finds simple route."""
        route = router.find_path("entrance", "lobby")
        
        assert route is not None
        assert len(route.nodes) == 2
        assert route.nodes[0].id == "entrance"
        assert route.nodes[1].id == "lobby"
        assert route.total_distance_m == 5.0
    
    def test_dijkstra_multi_hop_route(self, router):
        """Test Dijkstra finds multi-hop route."""
        route = router.find_path("entrance", "room_101")
        
        assert route is not None
        # entrance -> lobby -> hallway_n -> room_101
        assert len(route.nodes) == 4
        assert route.nodes[0].id == "entrance"
        assert route.nodes[-1].id == "room_101"
        assert route.total_distance_m == 25.0  # 5 + 10 + 10
    
    def test_dijkstra_alternative_route(self, router):
        """Test Dijkstra finds shortest of multiple paths."""
        # entrance to room_103: direct via lobby (15m) vs via hallway_n (25m)
        route = router.find_path("entrance", "room_103")
        
        assert route is not None
        # Should take lobby -> room_103 direct (10m) not via hallway_n
        assert route.total_distance_m == 15.0  # 5 + 10
    
    def test_no_path_unreachable(self, router):
        """Test no path for disconnected nodes."""
        # Add isolated node
        from path_shower.graph import GraphNode
        isolated = GraphNode("isolated", "Isolated", 100, 100)
        router.graph.add_node(isolated)
        
        route = router.find_path("entrance", "isolated")
        assert route is None
    
    def test_astar_same_as_dijkstra(self, graph):
        """Test A* gives same result as Dijkstra for this graph."""
        router_dijkstra = Router(graph, RouterConfig(algorithm="dijkstra"))
        router_astar = Router(graph, RouterConfig(algorithm="astar"))
        
        route_d = router_dijkstra.find_path("entrance", "room_101")
        route_a = router_astar.find_path("entrance", "room_101")
        
        assert route_d is not None
        assert route_a is not None
        assert route_d.total_distance_m == route_a.total_distance_m
        assert [n.id for n in route_d.nodes] == [n.id for n in route_a.nodes]
    
    def test_set_destination(self, router):
        """Test setting destination initializes route."""
        route = router.set_destination("room_101", "entrance")
        
        assert route is not None
        assert router._destination_id == "room_101"
        assert router._next_node_index == 1
    
    def test_update_position_on_route(self, router):
        """Test position update along planned route."""
        router.set_destination("room_101", "entrance")
        
        # Move to lobby (next on route)
        route, recalculated = router.update_position("lobby")
        
        assert not recalculated
        assert router._next_node_index == 2  # Advanced to hallway_n
    
    def test_update_position_off_route_recalculates(self, router):
        """Test recalculation when off route."""
        router.set_destination("room_101", "entrance")
        
        # Move to hallway_s (not on route to room_101)
        route, recalculated = router.update_position("hallway_s")
        
        assert recalculated
        assert router._next_node_index == 1
        # New route should go hallway_s -> lobby -> hallway_n -> room_101
        assert route is not None
    
    def test_update_position_at_destination(self, router):
        """Test arrival at destination."""
        router.set_destination("lobby", "entrance")
        
        # Move to lobby (destination)
        route, recalculated = router.update_position("lobby")
        
        assert not recalculated
        assert router._next_node_index == 2  # Past end
    
    def test_get_next_instruction(self, router):
        """Test getting next navigation instruction."""
        router.set_destination("room_101", "entrance")
        
        # At entrance, next is lobby
        instruction = router.get_next_instruction("entrance")
        assert instruction is not None
        node, edge = instruction
        assert node.id == "entrance"
        assert edge.to_id == "lobby"
    
    def test_get_remaining_distance(self, router):
        """Test remaining distance calculation."""
        router.set_destination("room_101", "entrance")
        
        # At start, full distance
        assert router.get_remaining_distance() == 25.0
        
        # After reaching lobby
        router.update_position("lobby")
        assert router.get_remaining_distance() == 20.0  # 10 + 10
    
    def test_clear_route(self, router):
        """Test clearing route."""
        router.set_destination("room_101", "entrance")
        router.clear_route()
        
        assert router._current_route is None
        assert router._destination_id is None
        assert router._next_node_index == 0


class TestTurnDetector:
    """Tests for turn detection."""
    
    @pytest.fixture
    def detector(self):
        return TurnDetector(TurnDetectorConfig())
    
    @pytest.fixture
    def straight_nodes(self):
        """Nodes in a straight line."""
        return [
            GraphNode("n1", "Start", 0, 0),
            GraphNode("n2", "Mid", 10, 0),
            GraphNode("n3", "End", 20, 0),
        ]
    
    @pytest.fixture
    def left_turn_nodes(self):
        """Nodes making a left turn."""
        return [
            GraphNode("n1", "Start", 0, 0),
            GraphNode("n2", "Corner", 10, 0),
            GraphNode("n3", "End", 10, 10),
        ]
    
    @pytest.fixture
    def right_turn_nodes(self):
        """Nodes making a right turn."""
        return [
            GraphNode("n1", "Start", 0, 0),
            GraphNode("n2", "Corner", 10, 0),
            GraphNode("n3", "End", 10, -10),
        ]
    
    @pytest.fixture
    def u_turn_nodes(self):
        """Nodes making a U-turn."""
        return [
            GraphNode("n1", "Start", 0, 0),
            GraphNode("n2", "Corner", 10, 0),
            GraphNode("n3", "Back", 0, 0),
        ]
    
    def test_heading_east(self, detector, straight_nodes):
        """Test heading calculation for eastward movement."""
        heading = detector.compute_heading(straight_nodes[0], straight_nodes[1])
        assert heading == 0.0  # East
    
    def test_heading_north(self, detector):
        """Test heading for northward movement."""
        n1 = GraphNode("a", "A", 0, 0)
        n2 = GraphNode("b", "B", 0, 10)
        heading = detector.compute_heading(n1, n2)
        assert heading == 90.0  # North
    
    def test_heading_west(self, detector):
        """Test heading for westward movement."""
        n1 = GraphNode("a", "A", 10, 0)
        n2 = GraphNode("b", "B", 0, 0)
        heading = detector.compute_heading(n1, n2)
        assert heading == 180.0 or heading == -180.0  # West
    
    def test_heading_south(self, detector):
        """Test heading for southward movement."""
        n1 = GraphNode("a", "A", 0, 10)
        n2 = GraphNode("b", "B", 0, 0)
        heading = detector.compute_heading(n1, n2)
        assert heading == -90.0  # South
    
    def test_straight_turn(self, detector, straight_nodes):
        """Test straight movement detected as straight."""
        edges = [
            GraphEdge("n1", "n2", 10),
            GraphEdge("n2", "n3", 10),
        ]
        turns = detector.detect_turns(straight_nodes, edges)
        
        # Should have one turn at n2 (straight) and arrive at n3
        assert len(turns) == 2
        assert turns[0]["instruction"] == NavigationInstruction.STRAIGHT
        assert turns[1]["instruction"] == NavigationInstruction.ARRIVE
    
    def test_left_turn(self, detector, left_turn_nodes):
        """Test left turn detection."""
        edges = [
            GraphEdge("n1", "n2", 10),
            GraphEdge("n2", "n3", 10),
        ]
        turns = detector.detect_turns(left_turn_nodes, edges)
        
        assert turns[0]["instruction"] == NavigationInstruction.LEFT
        # Turn angle should be around -90 (left)
        assert -135 <= turns[0]["turn_angle"] <= -45
    
    def test_right_turn(self, detector, right_turn_nodes):
        """Test right turn detection."""
        edges = [
            GraphEdge("n1", "n2", 10),
            GraphEdge("n2", "n3", 10),
        ]
        turns = detector.detect_turns(right_turn_nodes, edges)
        
        assert turns[0]["instruction"] == NavigationInstruction.RIGHT
        # Turn angle should be around 90 (right)
        assert 45 <= turns[0]["turn_angle"] <= 135
    
    def test_u_turn(self, detector, u_turn_nodes):
        """Test U-turn detection."""
        edges = [
            GraphEdge("n1", "n2", 10),
            GraphEdge("n2", "n3", 10),
        ]
        turns = detector.detect_turns(u_turn_nodes, edges)
        
        assert turns[0]["instruction"] == NavigationInstruction.U_TURN
        # Turn angle should be around +/-180
        assert abs(turns[0]["turn_angle"]) >= 135
    
    def test_generate_cues(self, detector, straight_nodes):
        """Test cue generation from current position."""
        edges = [
            GraphEdge("n1", "n2", 10),
            GraphEdge("n2", "n3", 10),
        ]
        
        # At start (index 0)
        cues = detector.generate_cues(straight_nodes, edges, 0)
        assert len(cues) == 2
        assert cues[0]["instruction"] == NavigationInstruction.STRAIGHT
        
        # At middle (index 1)
        cues = detector.generate_cues(straight_nodes, edges, 1)
        assert len(cues) == 1
        assert cues[0]["instruction"] == NavigationInstruction.ARRIVE
    
    def test_distance_rounding(self, detector):
        """Test distance rounding."""
        nodes = [
            GraphNode("n1", "A", 0, 0),
            GraphNode("n2", "B", 10.7, 0),  # 10.7m
            GraphNode("n3", "C", 20.3, 0),  # 9.6m from n2
        ]
        edges = [
            GraphEdge("n1", "n2", 10.7),
            GraphEdge("n2", "n3", 9.6),
        ]
        turns = detector.detect_turns(nodes, edges)
        
        assert turns[0]["distance_m"] == 11  # Rounded
        assert turns[1]["distance_m"] == 10  # Rounded


class TestNavigationMessage:
    """Tests for navigation message schema."""
    
    def test_create_navigation_left(self):
        """Test left turn message."""
        msg = create_navigation(
            instruction=SchemaNavigationInstruction.LEFT,
            distance_m=10,
            next_label="Hallway A",
        )
        
        assert msg.instruction == SchemaNavigationInstruction.LEFT
        assert msg.distance_m == 10
        assert msg.next_label == "Hallway A"
    
    def test_create_navigation_arrive(self):
        """Test arrive message (no distance/label)."""
        msg = create_navigation(
            instruction=SchemaNavigationInstruction.ARRIVE,
            distance_m=None,
            next_label=None,
        )
        
        assert msg.instruction == SchemaNavigationInstruction.ARRIVE
        assert msg.distance_m is None
        assert msg.next_label is None
    
    def test_navigation_distance_non_negative(self):
        """Test distance must be non-negative."""
        with pytest.raises(Exception):
            create_navigation(
                instruction=SchemaNavigationInstruction.LEFT,
                distance_m=-5,
            )
    
    def test_navigation_label_max_length(self):
        """Test next_label max 20 chars."""
        with pytest.raises(Exception):
            create_navigation(
                instruction=SchemaNavigationInstruction.LEFT,
                distance_m=10,
                next_label="x" * 21,
            )
    
    def test_all_instructions(self):
        """Test all instruction types valid."""
        for instr in SchemaNavigationInstruction:
            msg = create_navigation(instruction=instr)
            assert msg.instruction == instr


class TestIntegration:
    """Integration tests for full path shower pipeline."""
    
    def test_full_route_with_turns(self):
        """Test complete route with multiple turn types."""
        graph = create_sample_graph()
        router = Router(graph, RouterConfig(algorithm="dijkstra"))
        detector = TurnDetector(TurnDetectorConfig())
        
        # Route: entrance -> lobby -> hallway_n -> room_101
        # entrance(0,0) -> lobby(5,0): East
        # lobby(5,0) -> hallway_n(5,10): North (LEFT turn from East)
        # hallway_n(5,10) -> room_101(15,10): East (RIGHT turn from North)
        route = router.find_path("entrance", "room_101")
        
        assert route is not None
        assert len(route.nodes) == 4
        
        turns = detector.detect_turns(route.nodes, route.edges)
        
        # Should have turns at lobby (LEFT), hallway_n (RIGHT into room), arrive
        assert len(turns) == 3
        assert turns[0]["instruction"] == NavigationInstruction.LEFT      # lobby: East to North
        assert turns[1]["instruction"] == NavigationInstruction.RIGHT     # hallway_n: North to East
        assert turns[2]["instruction"] == NavigationInstruction.ARRIVE
    
    def test_recalculation_scenario(self):
        """Test realistic recalculation scenario."""
        graph = create_sample_graph()
        router = Router(graph, RouterConfig(algorithm="dijkstra"))
        detector = TurnDetector(TurnDetectorConfig())
        
        # User wants to go to room_101 from entrance
        router.set_destination("room_101", "entrance")
        
        # User correctly goes to lobby
        route, recalc = router.update_position("lobby")
        assert not recalc
        
        # User mistakenly goes to hallway_s (wrong direction)
        route, recalc = router.update_position("hallway_s")
        assert recalc
        
        # New route should go: hallway_s -> room_104 -> room_103 -> room_102 -> room_101 (shortest path)
        assert route is not None
        node_ids = [n.id for n in route.nodes]
        # Shortest path from hallway_s to room_101 goes through the rooms
        assert node_ids[0] == "hallway_s"
        assert node_ids[-1] == "room_101"
        assert "room_104" in node_ids  # Goes through rooms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])