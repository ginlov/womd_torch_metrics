# WOMD Motion Metrics - Corrected Implementation

This document summarizes the corrections made to match the official Waymo Open Motion Dataset metric specifications.

## Summary of Changes

### 1. New Module: `displacement_utils.py`
**Purpose**: Compute lateral and longitudinal displacements in trajectory-aligned coordinate frame

**Key Functions**:
- `compute_lateral_longitudinal_displacement()`: Decomposes displacement into trajectory-aligned components
  - Lateral: perpendicular to heading direction
  - Longitudinal: along heading direction
- `extract_heading_from_trajectory()`: Extracts or estimates heading from 2D or 7D trajectories
- `check_within_thresholds()`: Checks if prediction is within lateral AND longitudinal thresholds

### 2. New Module: `trajectory_classification.py`
**Purpose**: Classify trajectories into 8 types for mAP bucketing

**Classification Logic** (matching official WOMD):
1. If displacement < 2.0m: **STATIONARY**
2. Otherwise, based on heading change:
   - < 15°: **STRAIGHT**
   - 15-45°: **STRAIGHT_LEFT** or **STRAIGHT_RIGHT**
   - 45-135°: **LEFT_TURN** or **RIGHT_TURN**
   - > 135°: **LEFT_U_TURN** or **RIGHT_U_TURN**

**Key Functions**:
- `classify_trajectory_type()`: Main classification function
- `compute_trajectory_displacement()`: Total start-to-end displacement
- `compute_trajectory_heading_change()`: Total heading change

### 3. Updated: `motion_metrics.py`

#### Miss Rate Metric (FIXED)
**Before**: Used Euclidean FDE
**After**: Uses lateral AND longitudinal thresholds

```python
# A prediction misses if EITHER:
# - Lateral displacement > lateral_threshold, OR
# - Longitudinal displacement > longitudinal_threshold

# Official thresholds:
# - At 3s (step 30): lateral=1.0m, longitudinal=2.0m
# - At 5s (step 50): lateral=1.8m, longitudinal=3.6m  
# - At 8s (step 80): lateral=3.0m, longitudinal=6.0m
```

#### mAP Metric (FIXED)
**Before**: Matched on trajectory type equality + FDE ≤ 2.0m
**After**: Matches on lateral/longitudinal thresholds only

```python
# A prediction matches GT if BOTH:
# - Lateral displacement ≤ 1.0m (default)
# - Longitudinal displacement ≤ 2.0m (default)

# Trajectory type is used ONLY for bucketing (organizing results)
# NOT for matching!
```

## Key Differences from Previous Implementation

| Metric | Previous | Corrected (Official WOMD) |
|--------|----------|---------------------------|
| **Miss Rate** | FDE ≤ threshold | Lateral ≤ 1.0m AND Longitudinal ≤ 2.0m |
| **mAP Matching** | Type match + FDE ≤ 2.0m | Lateral ≤ 1.0m AND Longitudinal ≤ 2.0m |
| **Displacement** | Euclidean distance | Trajectory-aligned (lateral + longitudinal) |
| **Type Usage** | Required for matching | Only for bucketing results |

## References

1. **Official C++ Implementation**:
   - https://github.com/waymo-research/waymo-open-dataset/blob/main/src/waymo_open_dataset/metrics/motion_metrics.cc
   - Lines 487-497: `IsMatch()` function shows lateral/longitudinal checks
   - Lines 635-687: Trajectory type classification and bucketing

2. **Paper**:
   - Ettinger et al., "Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset", ICCV 2021
   - arXiv: 2104.10133

3. **Challenge Website**:
   - https://waymo.com/open/challenges/2024/motion-prediction/

## Testing

To verify the implementation matches official metrics:
1. Run predictions through official WOMD evaluation code
2. Run same predictions through this implementation
3. Compare mAP, Miss Rate values - should match closely

## Function Signature Compatibility

The main API `compute_motion_metrics()` signature remains **unchanged** - all fixes are internal to maintain backward compatibility.
