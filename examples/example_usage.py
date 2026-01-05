"""Example usage of Waymo Open Motion Dataset metrics for PyTorch."""

import torch
from womd_torch_metrics import compute_motion_metrics, MotionMetricsConfig, StepConfig

# Set random seed for reproducibility
torch.manual_seed(42)

# Configuration - using official Waymo challenge format
config = MotionMetricsConfig(
    track_steps_per_second=10,  # Ground truth at 10Hz
    prediction_steps_per_second=2,  # Predictions at 2Hz
    track_history_samples=10,
    track_future_samples=80,
    speed_lower_bound=1.4,
    speed_upper_bound=11.0,
    speed_scale_lower=0.5,
    speed_scale_upper=1.0,
    step_configurations=[
        StepConfig(measurement_step=5, lateral_miss_threshold=1.0, longitudinal_miss_threshold=2.0),  # 3s
        StepConfig(measurement_step=9, lateral_miss_threshold=1.8, longitudinal_miss_threshold=3.6),  # 5s
        StepConfig(measurement_step=15, lateral_miss_threshold=3.0, longitudinal_miss_threshold=6.0),  # 8s
    ],
    max_predictions=6,
    overlap_threshold=0.5,
)

# Example: Single scenario with 1 agent
B = 1  # batch size (scenarios)
M = 1  # number of joint prediction groups
K = 6  # top-K predictions
N = 1  # number of agents in joint prediction
T = 80  # prediction timesteps (8 seconds at 10Hz)
A = 1  # number of agents in ground truth
T_gt = 91  # ground truth timesteps (including history)

# Generate dummy data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction trajectories: [B, M, K, N, T, 2]
prediction_trajectory = torch.randn(B, M, K, N, T, 2, device=device) * 0.5

# Prediction scores: [B, M, K]
prediction_score = torch.softmax(torch.randn(B, M, K, device=device), dim=-1)

# Ground truth trajectory: [B, A, T_gt, 7]
# Format: [x, y, length, width, heading, velocity_x, velocity_y]
ground_truth_trajectory = torch.zeros(B, A, T_gt, 7, device=device)
ground_truth_trajectory[..., :2] = torch.randn(B, A, T_gt, 2, device=device) * 0.3  # x, y positions
ground_truth_trajectory[..., 2] = 4.5  # length (meters)
ground_truth_trajectory[..., 3] = 2.0  # width (meters)
ground_truth_trajectory[..., 4] = torch.randn(B, A, T_gt, device=device) * 0.1  # heading (radians)
ground_truth_trajectory[..., 5:7] = torch.randn(B, A, T_gt, 2, device=device) * 2.0  # velocity_x, velocity_y

# Ground truth validity: [B, A, T_gt]
ground_truth_is_valid = torch.ones(B, A, T_gt, dtype=torch.bool, device=device)

# Prediction to ground truth mapping: [B, M, N]
prediction_ground_truth_indices = torch.zeros(B, M, N, dtype=torch.long, device=device)

# Mask for valid mappings: [B, M, N]
prediction_ground_truth_indices_mask = torch.ones(B, M, N, dtype=torch.bool, device=device)

# Object types: [B, A] (1=VEHICLE, 2=PEDESTRIAN, 3=CYCLIST)
object_type = torch.ones(B, A, dtype=torch.long, device=device)  # VEHICLE

# Compute metrics (ground_truth_boxes not needed with 7D trajectory format)
print("Computing motion metrics...")
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

# Print results
print("\nMetrics Results:")
print("=" * 60)
for key, value in sorted(metrics.items()):
    print(f"{key:40s}: {value.item():.4f}")

