"""
Tests for map-based metrics (off-road detection).
"""

import pytest
import torch
import math
from womd_torch_metrics.map_metrics import (
    compute_bounding_box_corners,
    check_polyline_is_cyclic,
    compute_signed_distance_to_polylines,
    compute_distance_to_road_edge,
    compute_offroad_rate,
    compute_map_metrics,
    OFFROAD_DISTANCE_THRESHOLD,
)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBoundingBoxCorners:
    """Test bounding box corner computation."""
    
    def test_axis_aligned_box(self, device):
        """Test corner computation for axis-aligned box (heading=0)."""
        center_x = torch.tensor([0.0], device=device)
        center_y = torch.tensor([0.0], device=device)
        center_z = torch.tensor([0.0], device=device)
        length = torch.tensor([4.0], device=device)
        width = torch.tensor([2.0], device=device)
        heading = torch.tensor([0.0], device=device)
        
        corners = compute_bounding_box_corners(
            center_x, center_y, center_z, length, width, heading
        )
        
        assert corners.shape == (1, 4, 3)
        
        # Check corner positions (front=+x, left=+y for heading=0)
        expected_corners = torch.tensor([
            [[2.0, 1.0, 0.0],   # Front-left
             [2.0, -1.0, 0.0],  # Front-right
             [-2.0, -1.0, 0.0], # Back-right
             [-2.0, 1.0, 0.0]]  # Back-left
        ], device=device)
        
        assert torch.allclose(corners, expected_corners, atol=1e-5)
    
    def test_rotated_box(self, device):
        """Test corner computation for rotated box."""
        center_x = torch.tensor([0.0], device=device)
        center_y = torch.tensor([0.0], device=device)
        center_z = torch.tensor([0.0], device=device)
        length = torch.tensor([4.0], device=device)
        width = torch.tensor([2.0], device=device)
        heading = torch.tensor([math.pi / 2], device=device)  # 90 degrees
        
        corners = compute_bounding_box_corners(
            center_x, center_y, center_z, length, width, heading
        )
        
        # After 90 degree rotation, front points in +y direction
        expected_corners = torch.tensor([
            [[-1.0, 2.0, 0.0],  # Front-left (was +x, now +y)
             [1.0, 2.0, 0.0],   # Front-right
             [1.0, -2.0, 0.0],  # Back-right
             [-1.0, -2.0, 0.0]] # Back-left
        ], device=device)
        
        assert torch.allclose(corners, expected_corners, atol=1e-4)
    
    def test_batch_boxes(self, device):
        """Test corner computation for batch of boxes."""
        B, A, T = 2, 3, 4
        center_x = torch.randn(B, A, T, device=device)
        center_y = torch.randn(B, A, T, device=device)
        center_z = torch.randn(B, A, T, device=device)
        length = torch.rand(B, A, T, device=device) * 5 + 3
        width = torch.rand(B, A, T, device=device) * 2 + 1
        heading = torch.rand(B, A, T, device=device) * 2 * math.pi
        
        corners = compute_bounding_box_corners(
            center_x, center_y, center_z, length, width, heading
        )
        
        assert corners.shape == (B, A, T, 4, 3)


class TestPolylineCyclic:
    """Test polyline cyclic detection."""
    
    def test_open_polyline(self, device):
        """Test that open polyline is not detected as cyclic."""
        polyline = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], device=device)
        
        assert not check_polyline_is_cyclic(polyline)
    
    def test_closed_polyline(self, device):
        """Test that closed polyline is detected as cyclic."""
        polyline = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # Returns to start
        ], device=device)
        
        assert check_polyline_is_cyclic(polyline)
    
    def test_nearly_closed_polyline(self, device):
        """Test polyline that's almost closed."""
        polyline = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.1, 0.0],  # Close but not exact
        ], device=device)
        
        # Should be detected as cyclic due to tolerance
        is_cyclic = check_polyline_is_cyclic(polyline)
        assert is_cyclic


class TestSignedDistanceToPolylines:
    """Test signed distance computation to polylines."""
    
    def test_single_line_segment(self, device):
        """Test distance to a single line segment."""
        # Horizontal line from (0, 0) to (10, 0)
        polyline = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], device=device)
        
        # Points above and below the line
        query_points = torch.tensor([
            [5.0, 2.0, 0.0],   # Above line
            [5.0, -2.0, 0.0],  # Below line
            [5.0, 0.0, 0.0],   # On line
        ], device=device)
        
        distances = compute_signed_distance_to_polylines(
            query_points, [polyline]
        )
        
        assert distances.shape == (3,)
        # Point above should have positive distance (right side of vector)
        assert distances[0] > 0
        # Point below should have negative distance (left side of vector)
        assert distances[1] < 0
        # Point on line should be near zero
        assert torch.abs(distances[2]) < 0.1
    
    def test_road_boundaries(self, device):
        """Test distance to road boundaries (two parallel lines)."""
        # Road from y=-2 to y=2, extending in x direction
        left_boundary = torch.tensor([
            [0.0, -2.0, 0.0],
            [10.0, -2.0, 0.0],
        ], device=device)
        
        right_boundary = torch.tensor([
            [0.0, 2.0, 0.0],
            [10.0, 2.0, 0.0],
        ], device=device)
        
        # Query points inside and outside the road
        query_points = torch.tensor([
            [5.0, 0.0, 0.0],    # Inside center of road
            [5.0, 5.0, 0.0],    # Outside (above road)
            [5.0, -5.0, 0.0],   # Outside (below road)
        ], device=device)
        
        distances = compute_signed_distance_to_polylines(
            query_points, [left_boundary, right_boundary]
        )
        
        # Inside point should be negative (on-road)
        assert distances[0] < 0, f"Expected negative distance for point inside road, got {distances[0]}"
        # Outside points should be positive (off-road)
        assert distances[1] > 0, f"Expected positive distance for point outside road, got {distances[1]}"
        assert distances[2] > 0, f"Expected positive distance for point outside road, got {distances[2]}"
    
    def test_empty_polylines(self, device):
        """Test with empty polylines."""
        query_points = torch.tensor([
            [0.0, 0.0, 0.0],
        ], device=device)
        
        distances = compute_signed_distance_to_polylines(
            query_points, []
        )
        
        # Should return large distance
        assert distances[0] > 1e9


class TestDistanceToRoadEdge:
    """Test distance to road edge computation."""
    
    def test_box_fully_on_road(self, device):
        """Test box fully within road boundaries."""
        # Road is a 10m wide strip from y=-5 to y=5
        road_edges = [
            torch.tensor([
                [0.0, -5.0, 0.0],
                [100.0, -5.0, 0.0],
            ], device=device),
            torch.tensor([
                [0.0, 5.0, 0.0],
                [100.0, 5.0, 0.0],
            ], device=device),
        ]
        
        # Small box at center of road
        center_x = torch.tensor([[[50.0]]], device=device)  # [1, 1, 1]
        center_y = torch.tensor([[[0.0]]], device=device)
        center_z = torch.tensor([[[0.0]]], device=device)
        length = torch.tensor([[[4.0]]], device=device)
        width = torch.tensor([[[2.0]]], device=device)
        heading = torch.tensor([[[0.0]]], device=device)
        
        distances = compute_distance_to_road_edge(
            center_x, center_y, center_z, length, width, heading, road_edges
        )
        
        # All corners should be inside (negative distance)
        assert distances[0, 0, 0] < 0
    
    def test_box_partially_off_road(self, device):
        """Test box partially off road."""
        # Road is a 4m wide strip from y=-2 to y=2
        road_edges = [
            torch.tensor([
                [0.0, -2.0, 0.0],
                [100.0, -2.0, 0.0],
            ], device=device),
            torch.tensor([
                [0.0, 2.0, 0.0],
                [100.0, 2.0, 0.0],
            ], device=device),
        ]
        
        # Box at y=0 with 2m width extends from y=-1 to y=1
        # This should be fully on road
        center_x = torch.tensor([[[50.0]]], device=device)
        center_y = torch.tensor([[[0.0]]], device=device)
        center_z = torch.tensor([[[0.0]]], device=device)
        length = torch.tensor([[[4.0]]], device=device)
        width = torch.tensor([[[2.0]]], device=device)
        heading = torch.tensor([[[0.0]]], device=device)
        
        distances_on = compute_distance_to_road_edge(
            center_x, center_y, center_z, length, width, heading, road_edges
        )
        
        # Should be on road
        assert distances_on[0, 0, 0] < 0
        
        # Move box to y=2.5, so it extends from y=1.5 to y=3.5
        # This should be partially off road
        center_y = torch.tensor([[[2.5]]], device=device)
        
        distances_off = compute_distance_to_road_edge(
            center_x, center_y, center_z, length, width, heading, road_edges
        )
        
        # Should be off road (positive distance)
        assert distances_off[0, 0, 0] > 0
    
    def test_empty_road_edges_raises_error(self, device):
        """Test that empty road edges raises an error."""
        center_x = torch.tensor([[[0.0]]], device=device)
        center_y = torch.tensor([[[0.0]]], device=device)
        center_z = torch.tensor([[[0.0]]], device=device)
        length = torch.tensor([[[4.0]]], device=device)
        width = torch.tensor([[[2.0]]], device=device)
        heading = torch.tensor([[[0.0]]], device=device)
        
        with pytest.raises(ValueError, match="road_edge_polylines cannot be empty"):
            compute_distance_to_road_edge(
                center_x, center_y, center_z, length, width, heading, []
            )
    
    def test_valid_mask(self, device):
        """Test that valid mask is applied correctly."""
        road_edges = [
            torch.tensor([
                [0.0, -5.0, 0.0],
                [100.0, -5.0, 0.0],
            ], device=device),
        ]
        
        B, A, T = 1, 2, 3
        center_x = torch.randn(B, A, T, device=device)
        center_y = torch.randn(B, A, T, device=device)
        center_z = torch.randn(B, A, T, device=device)
        length = torch.ones(B, A, T, device=device) * 4.0
        width = torch.ones(B, A, T, device=device) * 2.0
        heading = torch.zeros(B, A, T, device=device)
        
        # Mask out some timesteps
        valid_mask = torch.tensor([
            [[True, False, True],
             [True, True, False]]
        ], device=device)
        
        distances = compute_distance_to_road_edge(
            center_x, center_y, center_z, length, width, heading,
            road_edges, valid_mask
        )
        
        # Invalid timesteps should have very negative distance
        assert distances[0, 0, 1] < -1e9
        assert distances[0, 1, 2] < -1e9


class TestOffRoadRate:
    """Test off-road rate computation."""
    
    def test_all_on_road(self, device):
        """Test scenario where all agents stay on road."""
        # Wide road from y=-10 to y=10
        road_edges = [
            torch.tensor([
                [0.0, -10.0, 0.0],
                [100.0, -10.0, 0.0],
            ], device=device),
            torch.tensor([
                [0.0, 10.0, 0.0],
                [100.0, 10.0, 0.0],
            ], device=device),
        ]
        
        B, A, T = 1, 2, 5
        # Agents stay near y=0 (center of road)
        center_x = torch.linspace(0, 100, T, device=device).repeat(B, A, 1)
        center_y = torch.zeros(B, A, T, device=device)
        center_z = torch.zeros(B, A, T, device=device)
        length = torch.ones(B, A, T, device=device) * 4.0
        width = torch.ones(B, A, T, device=device) * 2.0
        heading = torch.zeros(B, A, T, device=device)
        valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)
        
        offroad_rate = compute_offroad_rate(
            center_x, center_y, center_z, length, width, heading,
            road_edges, valid_mask
        )
        
        # Should be 0% off-road
        assert torch.allclose(offroad_rate, torch.zeros_like(offroad_rate), atol=1e-3)
    
    def test_all_off_road(self, device):
        """Test scenario where all agents are off road."""
        # Narrow road from y=-1 to y=1
        road_edges = [
            torch.tensor([
                [0.0, -1.0, 0.0],
                [100.0, -1.0, 0.0],
            ], device=device),
            torch.tensor([
                [0.0, 1.0, 0.0],
                [100.0, 1.0, 0.0],
            ], device=device),
        ]
        
        B, A, T = 1, 2, 5
        # Agents at y=10 (way off road)
        center_x = torch.linspace(0, 100, T, device=device).repeat(B, A, 1)
        center_y = torch.ones(B, A, T, device=device) * 10.0
        center_z = torch.zeros(B, A, T, device=device)
        length = torch.ones(B, A, T, device=device) * 4.0
        width = torch.ones(B, A, T, device=device) * 2.0
        heading = torch.zeros(B, A, T, device=device)
        valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)
        
        offroad_rate = compute_offroad_rate(
            center_x, center_y, center_z, length, width, heading,
            road_edges, valid_mask
        )
        
        # Should be 100% off-road
        assert torch.allclose(offroad_rate, torch.ones_like(offroad_rate), atol=1e-3)
    
    def test_partial_off_road(self, device):
        """Test scenario where agents are sometimes off road."""
        # Road from y=-5 to y=5
        road_edges = [
            torch.tensor([
                [0.0, -5.0, 0.0],
                [100.0, -5.0, 0.0],
            ], device=device),
            torch.tensor([
                [0.0, 5.0, 0.0],
                [100.0, 5.0, 0.0],
            ], device=device),
        ]
        
        B, A, T = 1, 1, 5
        center_x = torch.linspace(0, 100, T, device=device).repeat(B, A, 1)
        # Agent moves from on-road to off-road: [0, 0, 10, 10, 10]
        center_y = torch.tensor([[[0.0, 0.0, 10.0, 10.0, 10.0]]], device=device)
        center_z = torch.zeros(B, A, T, device=device)
        length = torch.ones(B, A, T, device=device) * 4.0
        width = torch.ones(B, A, T, device=device) * 2.0
        heading = torch.zeros(B, A, T, device=device)
        valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)
        
        offroad_rate = compute_offroad_rate(
            center_x, center_y, center_z, length, width, heading,
            road_edges, valid_mask
        )
        
        # Should be 60% off-road (3 out of 5 timesteps)
        expected_rate = torch.tensor([[0.6]], device=device)
        assert torch.allclose(offroad_rate, expected_rate, atol=0.1)


class TestComputeMapMetrics:
    """Test the main compute_map_metrics function."""
    
    def test_basic_metrics_computation(self, device):
        """Test basic metrics computation with ground truth."""
        B, A, T = 2, 3, 10
        
        # Create ground truth trajectories
        gt_traj = torch.zeros(B, A, T, 7, device=device)
        gt_traj[..., 0] = torch.linspace(0, 100, T).repeat(B, A, 1)  # x
        gt_traj[..., 1] = torch.randn(B, A, T, device=device) * 2    # y (within ±5)
        gt_traj[..., 2] = 0.0  # z
        gt_traj[..., 3] = 4.5  # length
        gt_traj[..., 4] = 2.0  # width
        gt_traj[..., 5] = 1.5  # height
        gt_traj[..., 6] = 0.0  # heading
        
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        obj_types = torch.tensor([[1, 2, 3], [1, 1, 2]], device=device)  # Mixed types
        
        # Create road edges for each scenario
        road_edges = []
        for b in range(B):
            scenario_edges = [
                [(0.0, -5.0, 0.0), (100.0, -5.0, 0.0)],
                [(0.0, 5.0, 0.0), (100.0, 5.0, 0.0)],
            ]
            road_edges.append(scenario_edges)
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        # Check that metrics are computed
        assert "VEHICLE/OffRoadRate" in metrics
        assert "PEDESTRIAN/OffRoadRate" in metrics
        assert "CYCLIST/OffRoadRate" in metrics
        
        # Check that values are reasonable (0 to 1)
        for key, value in metrics.items():
            assert 0.0 <= value.item() <= 1.0
    
    def test_with_2d_trajectories(self, device):
        """Test with 2D trajectory format."""
        B, A, T = 1, 2, 5
        
        # Only x, y coordinates
        gt_traj = torch.zeros(B, A, T, 2, device=device)
        gt_traj[..., 0] = torch.linspace(0, 10, T).repeat(B, A, 1)
        gt_traj[..., 1] = 0.0
        
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        obj_types = torch.tensor([[1, 1]], device=device)
        
        road_edges = [[
            [(0.0, -10.0, 0.0), (20.0, -10.0, 0.0)],
            [(0.0, 10.0, 0.0), (20.0, 10.0, 0.0)],
        ]]
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        assert "VEHICLE/OffRoadRate" in metrics
        # With default dimensions, should be on road
        assert metrics["VEHICLE/OffRoadRate"].item() < 0.5
    
    def test_empty_road_edges(self, device):
        """Test with scenario that has no road edges."""
        B, A, T = 1, 2, 5
        gt_traj = torch.randn(B, A, T, 7, device=device)
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        obj_types = torch.tensor([[1, 1]], device=device)
        
        # Empty road edges for scenario
        road_edges = [[]]
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        # Should return empty metrics (no scenarios with valid map data)
        assert len(metrics) == 0
    
    def test_multiple_object_types(self, device):
        """Test with multiple object types."""
        B, A, T = 1, 6, 8
        
        gt_traj = torch.zeros(B, A, T, 7, device=device)
        gt_traj[..., 0] = torch.linspace(0, 50, T).repeat(B, A, 1)
        gt_traj[..., 1] = torch.randn(B, A, T, device=device) * 3
        gt_traj[..., 3:5] = torch.tensor([4.5, 2.0], device=device)
        
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        # 2 vehicles, 2 pedestrians, 2 cyclists
        obj_types = torch.tensor([[1, 1, 2, 2, 3, 3]], device=device)
        
        road_edges = [[
            [(0.0, -10.0, 0.0), (100.0, -10.0, 0.0)],
            [(0.0, 10.0, 0.0), (100.0, 10.0, 0.0)],
        ]]
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        # All three types should have metrics
        assert "VEHICLE/OffRoadRate" in metrics
        assert "PEDESTRIAN/OffRoadRate" in metrics
        assert "CYCLIST/OffRoadRate" in metrics


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_straight_road_scenario(self, device):
        """Test agents driving straight on a road."""
        # Single agent driving straight down a road
        B, A, T = 1, 1, 20
        
        # Agent travels from x=0 to x=100, staying at y=0
        gt_traj = torch.zeros(B, A, T, 7, device=device)
        gt_traj[0, 0, :, 0] = torch.linspace(0, 100, T)  # x
        gt_traj[0, 0, :, 1] = 0.0  # y (center of road)
        gt_traj[0, 0, :, 3] = 4.5  # length
        gt_traj[0, 0, :, 4] = 2.0  # width
        
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        obj_types = torch.tensor([[1]], device=device)  # Vehicle
        
        # Road from y=-5 to y=5
        road_edges = [[
            [(0.0, -5.0, 0.0), (100.0, -5.0, 0.0)],
            [(0.0, 5.0, 0.0), (100.0, 5.0, 0.0)],
        ]]
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        # Agent should be fully on road
        assert metrics["VEHICLE/OffRoadRate"].item() < 0.01
    
    def test_agent_leaving_road(self, device):
        """Test agent that leaves the road."""
        B, A, T = 1, 1, 10
        
        gt_traj = torch.zeros(B, A, T, 7, device=device)
        # Agent moves in x direction
        gt_traj[0, 0, :, 0] = torch.linspace(0, 50, T)
        # Agent starts on road (y=0) then moves off road (y=10)
        y_positions = torch.cat([
            torch.zeros(5),  # On road for first 5 timesteps
            torch.ones(5) * 10  # Off road for last 5 timesteps
        ])
        gt_traj[0, 0, :, 1] = y_positions
        gt_traj[0, 0, :, 3] = 4.5  # length
        gt_traj[0, 0, :, 4] = 2.0  # width
        
        gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
        obj_types = torch.tensor([[1]], device=device)
        
        # Narrow road from y=-2 to y=2
        road_edges = [[
            [(0.0, -2.0, 0.0), (100.0, -2.0, 0.0)],
            [(0.0, 2.0, 0.0), (100.0, 2.0, 0.0)],
        ]]
        
        metrics = compute_map_metrics(
            gt_traj, gt_valid, obj_types, road_edges
        )
        
        # Should be about 50% off-road
        offroad_rate = metrics["VEHICLE/OffRoadRate"].item()
        assert 0.4 <= offroad_rate <= 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
