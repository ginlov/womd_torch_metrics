# Waymo Open Motion Dataset Metrics for PyTorch

A pure PyTorch implementation of the Waymo Open Motion Dataset metrics for motion forecasting evaluation.

## Features

Implements the core motion forecasting metrics:
- **minADE**: Minimum Average Displacement Error
- **minFDE**: Minimum Final Displacement Error
- **MissRate**: Rate of predictions that miss the ground truth (with lateral/longitudinal decomposition)
- **OverlapRate**: Rate of predictions that overlap with ground truth (using proper polygon IoU)
- **mAP**: Mean Average Precision (with trajectory type classification)

### Key Implementation Details

**MissRate**: Uses heading-based decomposition to compute lateral and longitudinal errors separately, matching the official Waymo implementation.

**OverlapRate**: Implements proper oriented bounding box intersection using the Sutherland-Hodgman clipping algorithm for accurate IoU calculation.

**mAP**: Classifies trajectories into 8 types (stationary, straight, left/right turns, U-turns) and computes precision-recall curves per type, aggregated across all scenarios.

**Speed-Based Weighting**: Metrics are weighted based on agent speed using linear interpolation. Agents outside the speed range [speed_lower_bound, speed_upper_bound] are filtered out. Within the range, weights scale linearly from speed_scale_lower to speed_scale_upper (default: 0.5 to 1.0 for 1.4-11.0 m/s).

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from womd_torch_metrics import compute_motion_metrics, MotionMetricsConfig, StepConfig

# Create config - using official Waymo challenge format
config = MotionMetricsConfig(
    track_steps_per_second=10,  # Ground truth at 10Hz
    prediction_steps_per_second=2,  # Predictions at 2Hz
    track_history_samples=10,
    track_future_samples=80,
    step_configurations=[
        StepConfig(measurement_step=5, lateral_miss_threshold=1.0, longitudinal_miss_threshold=2.0),  # 3s
        StepConfig(measurement_step=9, lateral_miss_threshold=1.8, longitudinal_miss_threshold=3.6),  # 5s
        StepConfig(measurement_step=15, lateral_miss_threshold=3.0, longitudinal_miss_threshold=6.0),  # 8s
    ],
    max_predictions=6,
)

# Prepare your data
# prediction_trajectory: [B, M, K, N, T, 2]
# prediction_score: [B, M, K]
# ground_truth_trajectory: [B, A, T_gt, 7] - [x, y, length, width, heading, velocity_x, velocity_y]
#   Note: Can also use [B, A, T_gt, 2] for just [x, y], but miss rate will be less accurate
# ground_truth_is_valid: [B, A, T_gt]
# prediction_ground_truth_indices: [B, M, N]
# prediction_ground_truth_indices_mask: [B, M, N]
# object_type: [B, A]

metrics = compute_motion_metrics(
    prediction_trajectory=pred_traj,
    prediction_score=pred_scores,
    ground_truth_trajectory=gt_traj,
    ground_truth_is_valid=gt_valid,
    prediction_ground_truth_indices=gt_indices,
    prediction_ground_truth_indices_mask=gt_mask,
    object_type=obj_types,
    config=config,
)

# Metrics are broken down by object type and measurement step
# e.g., "VEHICLE_5/minADE", "PEDESTRIAN_9/minFDE", etc.
# where 5, 9, 15 are measurement steps at 2Hz (3s, 5s, 8s)
print(metrics)
```

## Tensor Shapes

- **B**: Batch size (number of scenarios)
- **M**: Number of joint prediction groups per scenario
- **K**: Top-K predictions per joint prediction
- **N**: Number of agents in a joint prediction (usually 1)
- **T**: Number of prediction timesteps
- **A**: Number of agents in ground truth
- **T_gt**: Number of ground truth timesteps

### Ground Truth Format

For accurate miss rate calculation with lateral/longitudinal decomposition, use 7D format:
- `ground_truth_trajectory: [B, A, T_gt, 7]`
- Dimensions: `[x, y, length, width, heading, velocity_x, velocity_y]`

Simplified 2D format is also supported but less accurate:
- `ground_truth_trajectory: [B, A, T_gt, 2]`  
- Dimensions: `[x, y]` only

### Trajectory Type Classification

The mAP metric uses trajectory type classification based on heading changes and displacement:
- **STATIONARY**: Total displacement < 2.0m
- **STRAIGHT**: Heading change < 15°
- **STRAIGHT_LEFT/RIGHT**: 15° < heading < 45°
- **LEFT/RIGHT_TURN**: 45° < heading < 135°
- **LEFT/RIGHT_U_TURN**: heading > 135°

Precision-recall curves are computed per trajectory type and averaged for final mAP.

## Project Structure

```
womd-torch-metrics/
├── womd_torch_metrics/      # Main package
│   ├── __init__.py
│   └── motion_metrics.py    # Core metrics implementation
├── examples/                # Example usage scripts
│   └── example_usage.py
├── tests/                   # Unit tests
│   └── test_motion_metrics.py
├── setup.py
└── README.md
```

## Running Examples

```bash
python examples/example_usage.py
```

## Running Tests

```bash
python tests/test_motion_metrics.py
```

## Notes

This is a PyTorch-native implementation that doesn't require TensorFlow. The mAP computation is simplified compared to the full Waymo implementation which includes trajectory classification. For production use, you may want to enhance the trajectory classification and matching logic.

