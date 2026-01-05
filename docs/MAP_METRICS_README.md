# Map-Based Metrics (Off-Road Detection)

Map-based metrics evaluate whether predicted or ground truth trajectories stay on valid road surfaces using HD map data from the Waymo Open Motion Dataset.

## Metrics

### OffRoadRate

Measures the percentage of timesteps where an agent's bounding box extends beyond road boundaries.

- **Range**: 0.0 (always on-road) to 1.0 (always off-road)
- **Computation**: For each timestep, checks if any corner of the agent's bounding box is outside the road edges
- **Use case**: Validate that predictions are physically plausible (vehicles should stay on roads)

### DistanceToRoadEdge

Signed distance from the agent's bounding box to the nearest road edge.

- **Negative values**: Agent is on-road (inside road boundaries)
- **Positive values**: Agent is off-road (outside road boundaries)
- **Zero**: Agent is exactly at the road edge
- **Computation**: Uses the most off-road corner (maximum positive distance)

## Input Requirements

Map-based metrics require **additional inputs** beyond standard motion metrics:

### Required Inputs

1. **Agent Dimensions**: Bounding box dimensions at each timestep
   - `length`: [B, A, T] or included in trajectory as [..., 3]
   - `width`: [B, A, T] or included in trajectory as [..., 4]
   - `heading`: [B, A, T] or included in trajectory as [..., 6]

2. **HD Map Data**: Road edge polylines from scenario
   - List of road boundary polylines
   - Each polyline: List of 3D points (x, y, z)
   - Format: `List[List[Tuple[float, float, float]]]`

### Extracting from Waymo Scenarios

```python
# Extract road edges from Waymo scenario proto
road_edges = []
for map_feature in scenario.map_features:
    if map_feature.HasField('road_edge'):
        polyline = [(pt.x, pt.y, pt.z) for pt in map_feature.road_edge.polyline]
        road_edges.append(polyline)
```

## Usage

### Basic Example

```python
import torch
from womd_torch_metrics.map_metrics import compute_map_metrics

# Ground truth trajectories with dimensions
# Format: [x, y, z, length, width, height, heading]
gt_traj = torch.zeros(B, A, T, 7)
gt_traj[..., 0] = x_coordinates
gt_traj[..., 1] = y_coordinates
gt_traj[..., 3] = 4.5  # length
gt_traj[..., 4] = 2.0  # width
gt_traj[..., 6] = headings

gt_valid = torch.ones(B, A, T, dtype=torch.bool)
object_type = torch.ones(B, A, dtype=torch.long)  # 1=VEHICLE

# Road edges from HD map
road_edges = [[
    [(0.0, -5.0, 0.0), (100.0, -5.0, 0.0)],  # Left boundary
    [(0.0, 5.0, 0.0), (100.0, 5.0, 0.0)],    # Right boundary
]]

# Compute metrics
metrics = compute_map_metrics(
    gt_traj, gt_valid, object_type, road_edges
)

print(f"Off-road rate: {metrics['VEHICLE/OffRoadRate'].item():.2%}")
```

### Computing Distance Only

```python
from womd_torch_metrics.map_metrics import compute_distance_to_road_edge

# Road edge polylines as tensors
road_edges = [
    torch.tensor([[0.0, -5.0, 0.0], [100.0, -5.0, 0.0]]),
    torch.tensor([[0.0, 5.0, 0.0], [100.0, 5.0, 0.0]]),
]

distances = compute_distance_to_road_edge(
    center_x, center_y, center_z,
    length, width, heading,
    road_edges,
    valid_mask
)

# Negative = on-road, Positive = off-road
is_offroad = distances > 0.0
```

### Computing Off-Road Rate

```python
from womd_torch_metrics.map_metrics import compute_offroad_rate

offroad_rate = compute_offroad_rate(
    center_x, center_y, center_z,
    length, width, heading,
    road_edges,
    valid_mask,
    threshold=0.0  # Distance threshold for off-road
)

# Returns [B, A] off-road rates (fraction of timesteps off-road)
```

## How It Works

### 1. Bounding Box Computation

For each agent at each timestep, computes the 4 bottom corners of the oriented bounding box:

```
       Front
    FL ---- FR
    |        |
    |   +    |  (+ = center)
    |        |
    BL ---- BR
       Back
```

Corners are computed using:
- Center position (x, y)
- Dimensions (length, width)
- Orientation (heading in radians)

### 2. Distance Computation

For each corner, computes the signed 2D distance to road boundaries:

**Two-boundary model** (typical for roads):
- If signs to both boundaries are opposite → point is between boundaries (on-road, negative distance)
- If signs to both boundaries are same → point is outside both (off-road, positive distance)

**Distance calculation**:
- Project corner onto each road edge segment
- Compute 2D Euclidean distance in xy-plane
- Use z-coordinate with stretch factor to handle overpasses/underpasses

### 3. Off-Road Detection

An agent is considered off-road at a timestep if:
```
max(distance_of_all_corners) > threshold
```

Default threshold is `0.0` meters (any corner outside road boundary).

## Supported Trajectory Formats

### 7D Format (Recommended)
```python
trajectory = [..., T, 7]  # [x, y, z, length, width, height, heading]
```

Full format with all required dimensions at each timestep.

### 2D Format (Fallback)
```python
trajectory = [..., T, 2]  # [x, y]
```

Uses default dimensions:
- Length: 4.5m (typical vehicle)
- Width: 2.0m
- Heading: 0.0 (pointing in +x direction)
- Z: 0.0

**Note**: 2D format is less accurate for off-road detection.

## Metrics Breakdown

Results are broken down by object type:

```python
{
    "VEHICLE/OffRoadRate": tensor(0.05),    # 5% off-road
    "PEDESTRIAN/OffRoadRate": tensor(0.12), # 12% off-road
    "CYCLIST/OffRoadRate": tensor(0.08),    # 8% off-road
}
```

## Implementation Details

### Computational Complexity

- **Per agent per timestep**: O(C × S) where:
  - C = 4 corners per bounding box
  - S = total number of road edge segments

### GPU Acceleration

All computations are fully vectorized for GPU:
- Batch processing of multiple scenarios
- Parallel corner distance computation
- Efficient tensor operations

### Altitude Handling

Uses z-stretch factor (3.0×) for altitude matching to properly handle:
- Overpasses and underpasses
- Multi-level road structures
- Elevation changes

## Limitations

1. **Map Data Required**: Cannot compute without HD map road edges
2. **Road Boundary Model**: Assumes roads are defined by two parallel boundaries
3. **2D Projection**: Distance computed in xy-plane only
4. **Segment-Based**: Accuracy depends on polyline segment density

## Comparison with Waymo Official Implementation

This implementation follows the Waymo official approach from `sim_agents_metrics`:

- ✅ Oriented bounding box corners
- ✅ Signed distance to polylines
- ✅ Two-boundary road model
- ✅ Z-stretch for altitude matching
- ✅ Maximum corner distance

**Differences**:
- Pure PyTorch (no TensorFlow dependency)
- Simplified for motion forecasting evaluation
- Does not include full sim agents features (traffic lights, etc.)

## See Also

- [Motion Metrics README](../README.md) - Standard motion forecasting metrics
- [Example Usage](../examples/example_map_metrics.py) - Complete examples
- [Waymo Open Dataset](https://waymo.com/open/) - Official dataset documentation
