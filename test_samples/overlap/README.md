# Overlap Test Samples

This directory contains test samples for overlap rate computation in `.pth` format.

## File Format

Each `.pth` file should be a dictionary containing:

### Required fields:
- `prediction_trajectories`: `[K, T, 2]` - K predicted trajectories, each with T timesteps and (x, y) coordinates
- `ground_truth_trajectory`: `[T, 2]` - Ground truth trajectory with T timesteps and (x, y) coordinates
- `ground_truth_boxes`: `[T, 4]` - Ground truth boxes with format `[length, width, heading, velocity_heading]` for each timestep

### Optional fields:
- `valid_mask`: `[T]` - Boolean mask indicating valid timesteps (default: all True)
- `measurement_step`: `int` - Measurement step to evaluate at (default: None, uses all steps)
- `threshold`: `float` - Overlap threshold (default: 0.5)
- `expected_overlap_rate`: `float` - Expected overlap rate value for validation
- `tolerance`: `float` - Tolerance for comparison (default: 1e-5)

## Example: Creating a test sample

```python
import torch

# Create test data
K, T = 6, 80
pred_trajectories = torch.randn(K, T, 2) * 0.5
gt_trajectory = torch.randn(T, 2) * 0.3
gt_boxes = torch.ones(T, 4)
gt_boxes[:, 0] = 4.5  # length
gt_boxes[:, 1] = 2.0  # width

# Compute expected value (or set manually)
# expected_overlap_rate = ...

# Save test sample
test_data = {
    "prediction_trajectories": pred_trajectories,
    "ground_truth_trajectory": gt_trajectory,
    "ground_truth_boxes": gt_boxes,
    "threshold": 0.5,
    "expected_overlap_rate": expected_overlap_rate,
    "tolerance": 1e-5,
}

torch.save(test_data, "test_samples/overlap/sample_001.pth")
```

## Batch Test Format

For batch tests (files named `batch_*.pth`):

- `prediction_trajectories`: `[B, K, T, 2]` - Batch of predictions
- `ground_truth_trajectory`: `[B, T, 2]` - Batch of ground truth
- `ground_truth_boxes`: `[B, T, 4]` - Batch of boxes
- `expected_overlap_rates`: `[B]` - Expected overlap rates per batch item

## Running Tests

Tests will automatically discover and run all `.pth` files in this directory:

```bash
pytest tests/test_overlap_from_files.py -v
```

