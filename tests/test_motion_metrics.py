"""Comprehensive tests for motion metrics computation."""

import torch
import pytest
import math
from womd_torch_metrics import (
    compute_motion_metrics,
    MotionMetricsConfig,
    StepConfig,
    compute_min_ade,
    compute_min_fde,
    compute_miss_rate,
    compute_overlap_rate,
    compute_map_metric,
    TYPE_VEHICLE,
    TYPE_PEDESTRIAN,
    TYPE_CYCLIST,
)
from womd_torch_metrics.motion_metrics import (
    classify_trajectory_type,
    compute_precision_recall,
    compute_average_precision,
    compute_speed,
    compute_speed_scale,
    TRAJECTORY_TYPE_STATIONARY,
    TRAJECTORY_TYPE_STRAIGHT,
    TRAJECTORY_TYPE_STRAIGHT_LEFT,
    TRAJECTORY_TYPE_STRAIGHT_RIGHT,
    TRAJECTORY_TYPE_LEFT_TURN,
    TRAJECTORY_TYPE_RIGHT_TURN,
    TRAJECTORY_TYPE_LEFT_U_TURN,
    TRAJECTORY_TYPE_RIGHT_U_TURN,
)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    """Default config for tests."""
    return MotionMetricsConfig(
        track_steps_per_second=10,
        prediction_steps_per_second=10,  # For tests, use same frequency
        step_configurations=[
            StepConfig(measurement_step=10, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
            StepConfig(measurement_step=30, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
            StepConfig(measurement_step=50, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
            StepConfig(measurement_step=80, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
        ]
    )


def create_ground_truth_7d(B, A, T_gt, device):
    """Helper to create 7D ground truth trajectory.
    
    Returns:
        [B, A, T_gt, 7] tensor with [x, y, length, width, heading, velocity_x, velocity_y]
    """
    gt = torch.zeros(B, A, T_gt, 7, device=device)
    gt[..., :2] = torch.randn(B, A, T_gt, 2, device=device) * 0.3  # x, y positions
    gt[..., 2] = 4.5  # length
    gt[..., 3] = 2.0  # width
    gt[..., 4] = torch.randn(B, A, T_gt, device=device) * 0.1  # heading (small random angles)
    gt[..., 5:7] = torch.randn(B, A, T_gt, 2, device=device) * 2.0  # velocity_x, velocity_y
    return gt


class TestIndividualMetrics:
    """Test individual metric computation functions."""
    
    def test_min_ade_perfect_prediction(self, device):
        """Test minADE with perfect predictions (should be 0)."""
        K, T = 6, 80
        gt_trajectory = torch.randn(T, 2, device=device)
        
        # Perfect predictions: all K predictions match ground truth
        pred_trajectories = gt_trajectory.unsqueeze(0).repeat(K, 1, 1)
        
        min_ade = compute_min_ade(pred_trajectories, gt_trajectory)
        
        assert min_ade.item() < 1e-6, f"Perfect prediction should have minADE ≈ 0, got {min_ade.item()}"
    
    def test_min_ade_with_offset(self, device):
        """Test minADE with known offset."""
        K, T = 6, 80
        offset = torch.tensor([1.0, 2.0], device=device)
        
        gt_trajectory = torch.randn(T, 2, device=device)
        pred_trajectories = gt_trajectory.unsqueeze(0).repeat(K, 1, 1) + offset
        
        min_ade = compute_min_ade(pred_trajectories, gt_trajectory)
        expected_ade = torch.norm(offset).item()
        
        assert abs(min_ade.item() - expected_ade) < 1e-5, \
            f"Expected minADE ≈ {expected_ade}, got {min_ade.item()}"
    
    def test_min_ade_with_measurement_step(self, device):
        """Test minADE with measurement step cutoff."""
        K, T = 6, 80
        measurement_step = 30
        
        gt_trajectory = torch.randn(T, 2, device=device)
        pred_trajectories = torch.randn(K, T, 2, device=device)
        
        min_ade_full = compute_min_ade(pred_trajectories, gt_trajectory)
        min_ade_partial = compute_min_ade(pred_trajectories, gt_trajectory, measurement_step=measurement_step)
        
        # Partial should only consider first measurement_step timesteps
        assert min_ade_partial.item() >= 0
        # They should be different (unless by coincidence)
        # But we can't guarantee which is larger
    
    def test_min_ade_with_valid_mask(self, device):
        """Test minADE with validity mask."""
        K, T = 6, 80
        gt_trajectory = torch.randn(T, 2, device=device)
        pred_trajectories = gt_trajectory.unsqueeze(0).repeat(K, 1, 1)
        
        # Mask out some timesteps
        valid_mask = torch.ones(T, dtype=torch.bool, device=device)
        valid_mask[40:] = False  # Invalidate last 40 timesteps
        
        min_ade_with_mask = compute_min_ade(pred_trajectories, gt_trajectory, valid_mask=valid_mask)
        min_ade_without_mask = compute_min_ade(pred_trajectories, gt_trajectory)
        
        # With perfect predictions, both should be ~0
        assert min_ade_with_mask.item() < 1e-6
        assert min_ade_without_mask.item() < 1e-6
    
    def test_min_fde_perfect_prediction(self, device):
        """Test minFDE with perfect final prediction."""
        K, T = 6, 80
        gt_trajectory = torch.randn(T, 2, device=device)
        
        # Perfect final position
        pred_trajectories = torch.randn(K, T, 2, device=device)
        pred_trajectories[:, -1, :] = gt_trajectory[-1, :]  # Perfect final step
        
        min_fde = compute_min_fde(pred_trajectories, gt_trajectory)
        
        assert min_fde.item() < 1e-6, f"Perfect final prediction should have minFDE ≈ 0, got {min_fde.item()}"
    
    def test_min_fde_with_known_error(self, device):
        """Test minFDE with known final error."""
        K, T = 6, 80
        final_error = torch.tensor([3.0, 4.0], device=device)
        expected_fde = torch.norm(final_error).item()  # Should be 5.0
        
        gt_trajectory = torch.randn(T, 2, device=device)
        pred_trajectories = torch.randn(K, T, 2, device=device)
        pred_trajectories[:, -1, :] = gt_trajectory[-1, :] + final_error
        
        min_fde = compute_min_fde(pred_trajectories, gt_trajectory)
        
        assert abs(min_fde.item() - expected_fde) < 1e-5, \
            f"Expected minFDE ≈ {expected_fde}, got {min_fde.item()}"
    
    def test_miss_rate_below_threshold(self, device):
        """Test miss rate when predictions are within threshold."""
        K, T = 6, 80
        lateral_threshold = 2.0
        longitudinal_threshold = 2.0
        
        # Use 7D format with heading
        gt_trajectory = torch.zeros(T, 7, device=device)
        gt_trajectory[:, :2] = torch.randn(T, 2, device=device)  # x, y
        gt_trajectory[:, 4] = torch.zeros(T, device=device)  # heading = 0 (pointing along x-axis)
        
        pred_trajectories = torch.randn(K, T, 2, device=device)
        
        # Make sure best prediction is within threshold (0.5m in both directions)
        pred_trajectories[0, -1, :] = gt_trajectory[-1, :2] + torch.tensor([0.5, 0.5], device=device)
        
        miss_rate = compute_miss_rate(
            pred_trajectories, gt_trajectory, lateral_threshold, longitudinal_threshold
        )
        
        assert miss_rate.item() == 0.0, "Should not miss when within threshold"
    
    def test_miss_rate_above_threshold(self, device):
        """Test miss rate when predictions are above threshold."""
        K, T = 6, 80
        lateral_threshold = 2.0
        longitudinal_threshold = 2.0
        
        # Use 7D format with heading
        gt_trajectory = torch.zeros(T, 7, device=device)
        gt_trajectory[:, :2] = torch.randn(T, 2, device=device)  # x, y
        gt_trajectory[:, 4] = torch.zeros(T, device=device)  # heading = 0
        
        pred_trajectories = torch.randn(K, T, 2, device=device)
        
        # Make sure all predictions are far from ground truth
        pred_trajectories[:, -1, :] = gt_trajectory[-1, :2] + torch.tensor([10.0, 10.0], device=device)
        
        miss_rate = compute_miss_rate(
            pred_trajectories, gt_trajectory, lateral_threshold, longitudinal_threshold
        )
        
        assert miss_rate.item() == 1.0, "Should miss when all predictions are far"
    
    def test_overlap_rate_simple(self, device):
        """Test overlap rate computation."""
        K, T = 6, 80
        
        gt_trajectory = torch.zeros(T, 2, device=device)
        pred_trajectories = torch.zeros(K, T, 2, device=device)
        
        # Default box dimensions
        gt_boxes = torch.ones(T, 4, device=device)
        gt_boxes[:, 0] = 4.5  # length
        gt_boxes[:, 1] = 2.0  # width
        
        overlap_rate = compute_overlap_rate(
            pred_trajectories, gt_trajectory, gt_boxes, threshold=0.5
        )
        
        # When predictions are at same location, should have high overlap
        assert overlap_rate.item() >= 0.0
        assert overlap_rate.item() <= 1.0


class TestFullMetrics:
    """Test full motion metrics computation."""
    
    def test_basic_computation(self, device, config):
        """Test basic metric computation."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)  # VEHICLE
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        assert len(metrics) > 0
        for step in config.step_configurations:
            step_num = step.measurement_step
            assert f"VEHICLE_{step_num}/minADE" in metrics
            assert f"VEHICLE_{step_num}/minFDE" in metrics
            assert f"VEHICLE_{step_num}/MissRate" in metrics
            assert f"VEHICLE_{step_num}/OverlapRate" in metrics
            assert f"VEHICLE_{step_num}/mAP" in metrics
            
            # Check metric values are reasonable
            assert 0.0 <= metrics[f"VEHICLE_{step_num}/minADE"].item()
            assert 0.0 <= metrics[f"VEHICLE_{step_num}/minFDE"].item()
            assert 0.0 <= metrics[f"VEHICLE_{step_num}/MissRate"].item() <= 1.0
            assert 0.0 <= metrics[f"VEHICLE_{step_num}/OverlapRate"].item() <= 1.0
            assert 0.0 <= metrics[f"VEHICLE_{step_num}/mAP"].item()
    
    def test_multiple_object_types(self, device, config):
        """Test metrics with multiple object types."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 3, 91  # 3 agents
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        
        # Different object types: VEHICLE, PEDESTRIAN, CYCLIST
        object_type = torch.tensor([[TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST]], device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should have metrics for VEHICLE (first agent)
        assert "VEHICLE_10/minADE" in metrics
    
    def test_batch_processing(self, device, config):
        """Test processing multiple scenarios in batch."""
        B, M, K, N, T = 3, 1, 6, 1, 80  # 3 scenarios
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should aggregate across all scenarios
        assert len(metrics) > 0
        assert "VEHICLE_10/minADE" in metrics
    
    def test_with_ground_truth_boxes(self, device, config):
        """Test metrics with ground truth boxes for overlap computation."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        # Ground truth boxes: [length, width, heading, velocity_heading]
        ground_truth_boxes = torch.ones(B, A, T_gt, 4, device=device)
        ground_truth_boxes[:, :, :, 0] = 4.5  # length
        ground_truth_boxes[:, :, :, 1] = 2.0  # width
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
            ground_truth_boxes=ground_truth_boxes,
        )
        
        assert "VEHICLE_10/OverlapRate" in metrics
        assert 0.0 <= metrics["VEHICLE_10/OverlapRate"].item() <= 1.0
    
    def test_invalid_mask_handling(self, device, config):
        """Test handling of invalid prediction-ground truth mappings."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        
        # Mask out all predictions (should skip this scenario)
        prediction_ground_truth_indices_mask = torch.zeros(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should handle gracefully (may have empty metrics or zeros)
        assert isinstance(metrics, dict)
    
    def test_partial_validity_mask(self, device, config):
        """Test with partial validity in ground truth."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        
        # Only first half of timesteps are valid
        ground_truth_is_valid = torch.zeros(B, A, T_gt, dtype=torch.bool, device=device)
        ground_truth_is_valid[:, :, :T_gt//2] = True
        
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        assert "VEHICLE_10/minADE" in metrics
    
    def test_different_measurement_steps(self, device):
        """Test with different measurement step configurations."""
        config = MotionMetricsConfig(
            track_steps_per_second=10,
            prediction_steps_per_second=10,
            step_configurations=[
                StepConfig(measurement_step=5, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
                StepConfig(measurement_step=20, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
                StepConfig(measurement_step=40, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
            ]
        )
        
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should have metrics for each step
        for step in config.step_configurations:
            step_num = step.measurement_step
            assert f"VEHICLE_{step_num}/minADE" in metrics


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_prediction(self, device, config):
        """Test with only one prediction (K=1)."""
        B, M, K, N, T = 1, 1, 1, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.ones(B, M, K, device=device)  # Single prediction
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        assert "VEHICLE_10/minADE" in metrics
    
    def test_very_short_trajectory(self, device, config):
        """Test with very short trajectories."""
        B, M, K, N, T = 1, 1, 6, 1, 10  # Only 10 timesteps
        A, T_gt = 1, 11
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        # Use smaller steps
        config.step_configurations = [
            StepConfig(measurement_step=5, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
            StepConfig(measurement_step=10, lateral_miss_threshold=2.0, longitudinal_miss_threshold=2.0),
        ]
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        assert len(metrics) > 0
    
    def test_zero_predictions(self, device, config):
        """Test with zero-valued predictions."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.zeros(B, M, K, N, T, 2, device=device)
        prediction_score = torch.ones(B, M, K, device=device) / K
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should compute metrics (likely high error)
        assert "VEHICLE_10/minADE" in metrics
        assert metrics["VEHICLE_10/minADE"].item() >= 0.0
    
    def test_out_of_bounds_indices(self, device, config):
        """Test with out-of-bounds ground truth indices."""
        B, M, K, N, T = 1, 1, 6, 1, 80
        A, T_gt = 1, 91
        
        prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5
        prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)
        ground_truth_trajectory = create_ground_truth_7d(B, A, T_gt, device)
        ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)
        
        # Index out of bounds (should be skipped)
        prediction_ground_truth_indices = torch.tensor([[[10]]], dtype=torch.long, device=device)  # A=1, so index 10 is invalid
        prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)
        object_type = torch.ones(B, A, dtype=torch.long, device=device)
        
        metrics = compute_motion_metrics(
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
            config=config,
        )
        
        # Should handle gracefully
        assert isinstance(metrics, dict)


class TestNumericalCorrectness:
    """Test numerical correctness with known values."""
    
    def test_min_ade_manual_calculation(self, device):
        """Manually verify minADE calculation."""
        K, T = 3, 5
        
        # Simple ground truth: straight line
        gt_trajectory = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ], device=device)
        
        # Prediction 0: perfect match (ADE = 0)
        # Prediction 1: offset by 1.0 at each step (ADE = 1.0)
        # Prediction 2: offset by 2.0 at each step (ADE = 2.0)
        pred_trajectories = torch.stack([
            gt_trajectory.clone(),  # Perfect
            gt_trajectory + torch.tensor([1.0, 0.0], device=device),  # Offset by 1.0
            gt_trajectory + torch.tensor([2.0, 0.0], device=device),  # Offset by 2.0
        ])
        
        min_ade = compute_min_ade(pred_trajectories, gt_trajectory)
        
        assert abs(min_ade.item() - 0.0) < 1e-6, f"Expected minADE=0.0, got {min_ade.item()}"
    
    def test_min_fde_manual_calculation(self, device):
        """Manually verify minFDE calculation."""
        K, T = 3, 5
        
        gt_trajectory = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ], device=device)
        
        # Prediction 0: perfect final position (FDE = 0)
        # Prediction 1: final position offset by (3, 4) (FDE = 5.0)
        # Prediction 2: final position offset by (6, 8) (FDE = 10.0)
        pred_trajectories = torch.stack([
            gt_trajectory.clone(),
            gt_trajectory.clone(),
            gt_trajectory.clone(),
        ])
        pred_trajectories[0, -1, :] = gt_trajectory[-1, :]  # Perfect
        pred_trajectories[1, -1, :] = gt_trajectory[-1, :] + torch.tensor([3.0, 4.0], device=device)
        pred_trajectories[2, -1, :] = gt_trajectory[-1, :] + torch.tensor([6.0, 8.0], device=device)
        
        min_fde = compute_min_fde(pred_trajectories, gt_trajectory)
        
        assert abs(min_fde.item() - 0.0) < 1e-6, f"Expected minFDE=0.0, got {min_fde.item()}"


class TestConfig:
    """Test configuration handling."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MotionMetricsConfig()
        
        assert len(config.step_configurations) == 3
        assert config.step_configurations[0].measurement_step == 5
        assert config.step_configurations[1].measurement_step == 9
        assert config.step_configurations[2].measurement_step == 15
        assert config.overlap_threshold == 0.5
        assert config.track_steps_per_second == 10
        assert config.prediction_steps_per_second == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MotionMetricsConfig(
            track_steps_per_second=10,
            prediction_steps_per_second=5,
            step_configurations=[
                StepConfig(measurement_step=5, lateral_miss_threshold=1.5, longitudinal_miss_threshold=1.5),
                StepConfig(measurement_step=15, lateral_miss_threshold=1.5, longitudinal_miss_threshold=1.5),
                StepConfig(measurement_step=25, lateral_miss_threshold=1.5, longitudinal_miss_threshold=1.5),
            ],
            overlap_threshold=0.7,
        )
        
        assert len(config.step_configurations) == 3
        assert config.step_configurations[0].measurement_step == 5
        assert config.step_configurations[1].measurement_step == 15
        assert config.step_configurations[2].measurement_step == 25
        assert config.step_configurations[0].lateral_miss_threshold == 1.5
        assert config.overlap_threshold == 0.7
        assert config.prediction_steps_per_second == 5


class TestTrajectoryClassification:
    """Test trajectory type classification."""
    
    def test_stationary_trajectory(self, device):
        """Test classification of stationary trajectory."""
        # Trajectory with minimal movement (< 2m)
        trajectory = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.05],
            [0.15, 0.1],
            [0.2, 0.15],
        ], device=device)
        
        traj_type = classify_trajectory_type(trajectory)
        assert traj_type == TRAJECTORY_TYPE_STATIONARY
    
    def test_straight_trajectory(self, device):
        """Test classification of straight trajectory."""
        # Straight line (heading change < 15 degrees)
        trajectory = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.1],  # Slight deviation
            [2.0, 0.15],
            [3.0, 0.2],
            [4.0, 0.25],
        ], device=device)
        
        traj_type = classify_trajectory_type(trajectory)
        assert traj_type == TRAJECTORY_TYPE_STRAIGHT
    
    def test_left_turn_trajectory(self, device):
        """Test classification of left turn trajectory."""
        # Left turn (counter-clockwise, 45-135 degrees)
        t = torch.linspace(0, math.pi/2, 10, device=device)  # 90 degree turn
        radius = 5.0
        trajectory = torch.stack([
            radius * torch.cos(t),
            radius * torch.sin(t),
        ], dim=1)
        
        traj_type = classify_trajectory_type(trajectory)
        assert traj_type in [TRAJECTORY_TYPE_LEFT_TURN, TRAJECTORY_TYPE_STRAIGHT_LEFT]
    
    def test_right_turn_trajectory(self, device):
        """Test classification of right turn trajectory."""
        # Right turn (clockwise, 45-135 degrees)
        t = torch.linspace(0, -math.pi/2, 10, device=device)  # -90 degree turn
        radius = 5.0
        trajectory = torch.stack([
            radius * torch.cos(t),
            radius * torch.sin(t),
        ], dim=1)
        
        traj_type = classify_trajectory_type(trajectory)
        assert traj_type in [TRAJECTORY_TYPE_RIGHT_TURN, TRAJECTORY_TYPE_STRAIGHT_RIGHT]
    
    def test_u_turn_trajectory(self, device):
        """Test classification of U-turn trajectory."""
        # Left U-turn (180 degrees)
        t = torch.linspace(0, math.pi, 10, device=device)
        radius = 3.0
        trajectory = torch.stack([
            radius * torch.cos(t),
            radius * torch.sin(t),
        ], dim=1)
        
        traj_type = classify_trajectory_type(trajectory)
        assert traj_type in [TRAJECTORY_TYPE_LEFT_U_TURN, TRAJECTORY_TYPE_LEFT_TURN]


class TestPrecisionRecall:
    """Test precision-recall curve computation."""
    
    def test_perfect_predictions(self, device):
        """Test PR curve with all predictions correct."""
        pred_scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        pred_matched = torch.tensor([True, True, True], device=device)
        num_gt = 3
        
        precision, recall, thresholds = compute_precision_recall(
            pred_scores, pred_matched, num_gt
        )
        
        # All predictions correct: precision should be 1.0
        assert torch.all(precision >= 0.99)
        # Recall should reach 1.0
        assert recall[-1].item() >= 0.99
    
    def test_mixed_predictions(self, device):
        """Test PR curve with mixed correct/incorrect predictions."""
        pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)
        pred_matched = torch.tensor([True, False, True, False], device=device)
        num_gt = 2  # 2 ground truth instances
        
        precision, recall, thresholds = compute_precision_recall(
            pred_scores, pred_matched, num_gt
        )
        
        # Check that precision decreases as we include more predictions
        # First prediction (0.9): TP=1, FP=0 -> precision=1.0
        assert precision[1].item() >= 0.99
        # After 2nd prediction (0.8): TP=1, FP=1 -> precision=0.5
        assert abs(precision[2].item() - 0.5) < 0.01
    
    def test_no_predictions(self, device):
        """Test PR curve with no predictions."""
        pred_scores = torch.tensor([], device=device)
        pred_matched = torch.tensor([], dtype=torch.bool, device=device)
        num_gt = 5
        
        precision, recall, thresholds = compute_precision_recall(
            pred_scores, pred_matched, num_gt
        )
        
        assert len(precision) == 1
        assert len(recall) == 1
        assert precision[0].item() == 0.0
        assert recall[0].item() == 0.0


class TestAveragePrecision:
    """Test average precision computation."""
    
    def test_perfect_ap(self, device):
        """Test AP with perfect precision-recall."""
        precision = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
        recall = torch.tensor([0.0, 0.33, 0.67, 1.0], device=device)
        
        ap = compute_average_precision(precision, recall)
        assert abs(ap.item() - 1.0) < 0.01
    
    def test_known_ap(self, device):
        """Test AP with known precision-recall values."""
        # Create a simple PR curve
        precision = torch.tensor([1.0, 1.0, 0.5, 0.33], device=device)
        recall = torch.tensor([0.0, 0.5, 0.5, 1.0], device=device)
        
        ap = compute_average_precision(precision, recall)
        # AP should be approximately 0.5 * 1.0 + 0.5 * 0.33 = 0.665
        assert 0.6 <= ap.item() <= 0.7
    
    def test_zero_ap(self, device):
        """Test AP with no correct predictions."""
        precision = torch.tensor([1.0, 0.0], device=device)
        recall = torch.tensor([0.0, 0.0], device=device)
        
        ap = compute_average_precision(precision, recall)
        assert ap.item() == 0.0


class TestMAPWithTrajectoryTypes:
    """Test mAP computation with trajectory type classification."""
    
    def test_map_same_trajectory_type(self, device):
        """Test mAP when predictions match GT trajectory type."""
        # Create a straight trajectory
        gt_trajectory = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ], device=device)
        
        # Predictions: also straight, with varying accuracy
        pred_trajectories = torch.stack([
            gt_trajectory + 0.1,  # Close match
            gt_trajectory + 0.5,  # Medium match
            gt_trajectory + 1.5,  # Far match
        ])
        
        pred_scores = torch.tensor([0.9, 0.7, 0.5], device=device)
        gt_boxes = torch.zeros(5, 4, device=device)
        
        gt_type, pred_types, pred_matched, scores = compute_map_metric(
            pred_trajectories, pred_scores, gt_trajectory, gt_boxes
        )
        
        # GT should be classified as straight
        assert gt_type in [TRAJECTORY_TYPE_STRAIGHT, TRAJECTORY_TYPE_STATIONARY]
        # All predictions should be same type
        assert torch.all(pred_types == gt_type)
        # At least the closest prediction should match
        assert pred_matched[0].item() == True
    
    def test_map_different_trajectory_types(self, device):
        """Test mAP when predictions have different trajectory types."""
        # GT: Straight trajectory
        gt_trajectory = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
            [8.0, 0.0],
        ], device=device)
        
        # Prediction 1: Straight (should match)
        pred1 = gt_trajectory + 0.1
        
        # Prediction 2: Turn (different type, should not match even if close)
        t = torch.linspace(0, math.pi/3, 5, device=device)
        pred2 = torch.stack([
            4.0 * torch.cos(t),
            4.0 * torch.sin(t),
        ], dim=1)
        
        pred_trajectories = torch.stack([pred1, pred2])
        pred_scores = torch.tensor([0.9, 0.8], device=device)
        gt_boxes = torch.zeros(5, 4, device=device)
        
        gt_type, pred_types, pred_matched, scores = compute_map_metric(
            pred_trajectories, pred_scores, gt_trajectory, gt_boxes
        )
        
        # First prediction should match (same type + close)
        # Second prediction might not match (different type or too far)
        assert pred_matched[0].item() == True


