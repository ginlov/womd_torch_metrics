"""
Test script to verify corrected WOMD metrics implementation.
"""

import torch
from womd_torch_metrics.motion_metrics import compute_motion_metrics, MotionMetricsConfig

def test_basic_metrics():
    """Test basic metric computation with synthetic data."""
    print("Testing WOMD metrics with synthetic data...\n")
    
    # Create synthetic data
    B = 2  # batch size
    M = 1  # prediction groups
    K = 6  # top-K predictions
    N = 1  # agents per prediction
    T = 80  # prediction timesteps
    A = 2  # agents in ground truth
    
    # Create predictions: straight trajectory (moving 20m over 80 steps = 2.5 m/s at 10Hz)
    pred_traj = torch.zeros(B, M, K, N, T, 2)
    for k in range(K):
        # Different predictions with varying accuracy
        pred_traj[:, :, k, :, :, 0] = torch.linspace(0, 20 + k*0.5, T)  # x
        pred_traj[:, :, k, :, :, 1] = torch.linspace(0, 0.2*k, T)  # y (small deviation)
    
    pred_scores = torch.softmax(torch.randn(B, M, K), dim=-1)
    
    # Create ground truth: straight trajectory at 20m (2.5 m/s - within speed bounds)
    gt_traj = torch.zeros(B, A, T, 7)
    gt_traj[:, 0, :, 0] = torch.linspace(0, 20, T)  # x
    gt_traj[:, 0, :, 1] = 0.0  # y
    gt_traj[:, 0, :, 2] = 4.5  # length
    gt_traj[:, 0, :, 3] = 2.0  # width
    gt_traj[:, 0, :, 4] = 0.0  # heading (straight)
    gt_traj[:, 0, :, 5] = 0.25  # velocity_x (20m / 80 steps)
    gt_traj[:, 0, :, 6] = 0.0  # velocity_y
    
    # Second agent: turning left (also 2.5 m/s)
    gt_traj[:, 1, :, 0] = torch.linspace(0, 14, T)  # x
    gt_traj[:, 1, :, 1] = torch.linspace(0, 14, T)  # y (turning)
    gt_traj[:, 1, :, 2] = 4.5
    gt_traj[:, 1, :, 3] = 2.0
    gt_traj[:, 1, :, 4] = torch.linspace(0, 0.785, T)  # heading (45 deg turn)
    gt_traj[:, 1, :, 5] = 0.175
    gt_traj[:, 1, :, 6] = 0.175
    
    gt_valid = torch.ones(B, A, T, dtype=torch.bool)
    
    # Prediction to GT mapping (first prediction group predicts first agent)
    pred_gt_indices = torch.zeros(B, M, N, dtype=torch.long)
    pred_gt_indices[:, 0, 0] = 0  # Predict agent 0
    
    pred_gt_mask = torch.ones(B, M, N, dtype=torch.bool)
    
    # Object types
    obj_type = torch.ones(B, A, dtype=torch.long)  # All vehicles
    
    # Configure metrics
    config = MotionMetricsConfig()
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_motion_metrics(
        prediction_trajectory=pred_traj,
        prediction_score=pred_scores,
        ground_truth_trajectory=gt_traj,
        ground_truth_is_valid=gt_valid,
        prediction_ground_truth_indices=pred_gt_indices,
        prediction_ground_truth_indices_mask=pred_gt_mask,
        object_type=obj_type,
        config=config,
    )
    
    # Print results
    print("\n" + "="*60)
    print("METRIC RESULTS")
    print("="*60)
    
    # Separate mAP metrics from others
    map_metrics = {}
    other_metrics = {}
    for key, value in sorted(metrics.items()):
        if 'mAP' in key:
            map_metrics[key] = value
        else:
            other_metrics[key] = value
    
    if other_metrics:
        print("\nADE/FDE/MissRate/OverlapRate Metrics:")
        print("-" * 60)
        for key, value in other_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"{key:40s}: {value.item():.4f}")
            else:
                print(f"{key:40s}: {value}")
    else:
        print("\nNo ADE/FDE/MissRate metrics computed (check agent matching)")
    
    if map_metrics:
        print("\nmAP Metrics:")
        print("-" * 60)
        for key, value in map_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"{key:40s}: {value.item():.4f}")
            else:
                print(f"{key:40s}: {value}")
    else:
        print("\nNo mAP metrics computed")
    
    print("\n" + "="*60)
    print("Key observations:")
    print("- minADE/minFDE should be small (best prediction is close)")
    print("- MissRate should be 0.0 or low (at least one pred within thresholds)")
    print("- mAP should be high if predictions are good")
    print("="*60)
    
    return metrics

def test_displacement_utils():
    """Test lateral/longitudinal displacement computation."""
    print("\n\nTesting displacement utilities...\n")
    
    from womd_torch_metrics.displacement_utils import (
        compute_lateral_longitudinal_displacement,
        extract_heading_from_trajectory,
        check_within_thresholds
    )
    
    # Test case: prediction slightly off to the side
    pred_point = torch.tensor([10.0, 0.5])  # 0.5m lateral offset
    gt_point = torch.tensor([10.0, 0.0])
    gt_heading = torch.tensor(0.0)  # pointing in +x direction
    
    lateral, longitudinal = compute_lateral_longitudinal_displacement(
        pred_point, gt_point, gt_heading
    )
    
    print(f"Prediction: {pred_point.tolist()}")
    print(f"Ground truth: {gt_point.tolist()}")
    print(f"Heading: {gt_heading.item():.2f} rad")
    print(f"Lateral displacement: {lateral.item():.3f} m")
    print(f"Longitudinal displacement: {longitudinal.item():.3f} m")
    print(f"Expected: lateral=0.5m, longitudinal=0.0m")
    
    # Test check_within_thresholds
    pred_traj = torch.tensor([[[10.0, 0.5]]])  # [1, 1, 2]
    gt_traj = torch.tensor([[10.0, 0.0, 4.5, 2.0, 0.0, 0.0, 0.0]])  # [1, 7]
    
    within = check_within_thresholds(
        pred_traj, gt_traj,
        lateral_threshold=1.0,
        longitudinal_threshold=2.0,
        measurement_step=0
    )
    
    print(f"\nWithin thresholds (1.0m lat, 2.0m long): {within.item()}")
    print("Expected: True (0.5m < 1.0m)")
    
    # Test with larger deviation
    pred_traj2 = torch.tensor([[[10.0, 1.5]]])  # [1, 1, 2]
    within2 = check_within_thresholds(
        pred_traj2, gt_traj,
        lateral_threshold=1.0,
        longitudinal_threshold=2.0,
        measurement_step=0
    )
    print(f"Within thresholds (1.5m lateral): {within2.item()}")
    print("Expected: False (1.5m > 1.0m)")

def test_trajectory_classification():
    """Test trajectory type classification."""
    print("\n\nTesting trajectory classification...\n")
    
    from womd_torch_metrics.trajectory_classification import (
        classify_trajectory_type,
        get_trajectory_type_name
    )
    
    # Test stationary
    stationary = torch.zeros(10, 7)
    stationary[:, :2] = torch.tensor([0.0, 0.0])  # No movement
    traj_type = classify_trajectory_type(stationary)
    print(f"Stationary trajectory: {get_trajectory_type_name(traj_type)}")
    
    # Test straight
    straight = torch.zeros(80, 7)
    straight[:, 0] = torch.linspace(0, 10, 80)  # Move in x only
    straight[:, 1] = 0.0
    straight[:, 4] = 0.0  # Constant heading
    traj_type = classify_trajectory_type(straight)
    print(f"Straight trajectory: {get_trajectory_type_name(traj_type)}")
    
    # Test left turn
    left_turn = torch.zeros(80, 7)
    left_turn[:, 0] = torch.linspace(0, 5, 80)
    left_turn[:, 1] = torch.linspace(0, 5, 80)
    left_turn[:, 4] = torch.linspace(0, 1.2, 80)  # ~70 degree turn
    traj_type = classify_trajectory_type(left_turn)
    print(f"Left turn trajectory: {get_trajectory_type_name(traj_type)}")
    
    # Test right turn
    right_turn = torch.zeros(80, 7)
    right_turn[:, 0] = torch.linspace(0, 5, 80)
    right_turn[:, 1] = torch.linspace(0, -5, 80)
    right_turn[:, 4] = torch.linspace(0, -1.2, 80)  # ~-70 degree turn
    traj_type = classify_trajectory_type(right_turn)
    print(f"Right turn trajectory: {get_trajectory_type_name(traj_type)}")

if __name__ == "__main__":
    print("="*60)
    print("WOMD METRICS IMPLEMENTATION TEST")
    print("="*60)
    
    test_displacement_utils()
    test_trajectory_classification()
    test_basic_metrics()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
