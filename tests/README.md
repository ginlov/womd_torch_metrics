# Test Suite

Comprehensive test suite for Waymo Open Motion Dataset metrics for PyTorch.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_motion_metrics.py::TestIndividualMetrics -v

# Run specific test
pytest tests/test_motion_metrics.py::TestIndividualMetrics::test_min_ade_perfect_prediction -v
```

## Test Coverage

### TestIndividualMetrics (10 tests)
Tests for individual metric computation functions:
- `test_min_ade_perfect_prediction`: Perfect predictions should yield minADE ≈ 0
- `test_min_ade_with_offset`: minADE with known offset
- `test_min_ade_with_measurement_step`: minADE with step cutoff
- `test_min_ade_with_valid_mask`: minADE with validity mask
- `test_min_fde_perfect_prediction`: Perfect final prediction should yield minFDE ≈ 0
- `test_min_fde_with_known_error`: minFDE with known final error
- `test_miss_rate_below_threshold`: Miss rate when within threshold
- `test_miss_rate_above_threshold`: Miss rate when above threshold
- `test_overlap_rate_simple`: Basic overlap rate computation

### TestFullMetrics (7 tests)
Tests for full motion metrics computation:
- `test_basic_computation`: Basic metric computation with all metrics
- `test_multiple_object_types`: Metrics with VEHICLE, PEDESTRIAN, CYCLIST
- `test_batch_processing`: Processing multiple scenarios in batch
- `test_with_ground_truth_boxes`: Metrics with ground truth boxes for overlap
- `test_invalid_mask_handling`: Handling of invalid prediction-GT mappings
- `test_partial_validity_mask`: Partial validity in ground truth
- `test_different_measurement_steps`: Different measurement step configurations

### TestEdgeCases (4 tests)
Edge cases and boundary conditions:
- `test_single_prediction`: Only one prediction (K=1)
- `test_very_short_trajectory`: Very short trajectories
- `test_zero_predictions`: Zero-valued predictions
- `test_out_of_bounds_indices`: Out-of-bounds ground truth indices

### TestNumericalCorrectness (2 tests)
Numerical correctness with known values:
- `test_min_ade_manual_calculation`: Manually verify minADE calculation
- `test_min_fde_manual_calculation`: Manually verify minFDE calculation

### TestConfig (2 tests)
Configuration handling:
- `test_default_config`: Default configuration values
- `test_custom_config`: Custom configuration values

## Total: 24 tests

All tests pass and cover:
- ✅ Individual metric functions
- ✅ Full pipeline computation
- ✅ Edge cases and error handling
- ✅ Numerical correctness
- ✅ Configuration options
- ✅ Different object types
- ✅ Batch processing
- ✅ Mask handling
- ✅ Boundary conditions

