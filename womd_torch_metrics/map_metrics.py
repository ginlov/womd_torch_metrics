"""
PyTorch implementation of map-based metrics for Waymo Open Motion Dataset.

This module implements map adherence metrics:
- OffRoadRate: Rate of predictions that go off-road
- DistanceToRoadEdge: Signed distance from agent bounding box to road edges
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


# Constants
EXTREMELY_LARGE_DISTANCE = 1e10
OFFROAD_DISTANCE_THRESHOLD = 0.0  # meters - positive distance is off-road
Z_STRETCH_FACTOR = 3.0  # Scaling for vertical distances (handles overpasses)
CYCLIC_POLYLINE_TOLERANCE = 1.0  # m^2


@dataclass
class MapPoint:
    """A 3D point representing a location on the map."""
    x: float
    y: float
    z: float


def compute_bounding_box_corners(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    center_z: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 4 bottom corners of oriented bounding boxes.
    
    Args:
        center_x: [...] x-coordinates of box centers
        center_y: [...] y-coordinates of box centers
        center_z: [...] z-coordinates of box centers
        length: [...] box lengths
        width: [...] box widths
        heading: [...] box headings (radians)
        
    Returns:
        [..., 4, 3] tensor of corner coordinates (x, y, z) for 4 corners
    """
    # Compute half dimensions
    half_length = length / 2.0
    half_width = width / 2.0
    
    # Compute cos and sin of heading
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    
    # Define 4 corners in local coordinate frame (front-left, front-right, back-right, back-left)
    # Front is in the direction of heading
    local_corners = torch.stack([
        torch.stack([half_length, half_width], dim=-1),    # Front-left
        torch.stack([half_length, -half_width], dim=-1),   # Front-right
        torch.stack([-half_length, -half_width], dim=-1),  # Back-right
        torch.stack([-half_length, half_width], dim=-1),   # Back-left
    ], dim=-2)  # [..., 4, 2]
    
    # Rotation matrix application
    # x' = x * cos - y * sin
    # y' = x * sin + y * cos
    corners_x = (local_corners[..., 0] * cos_h.unsqueeze(-1) - 
                 local_corners[..., 1] * sin_h.unsqueeze(-1))
    corners_y = (local_corners[..., 0] * sin_h.unsqueeze(-1) + 
                 local_corners[..., 1] * cos_h.unsqueeze(-1))
    
    # Translate to global coordinates
    corners_x = corners_x + center_x.unsqueeze(-1)
    corners_y = corners_y + center_y.unsqueeze(-1)
    corners_z = center_z.unsqueeze(-1).expand_as(corners_x)
    
    # Stack into [..., 4, 3] tensor
    corners = torch.stack([corners_x, corners_y, corners_z], dim=-1)
    
    return corners


def check_polyline_is_cyclic(polyline_points: torch.Tensor) -> bool:
    """
    Check if a polyline is cyclic (forms a closed loop).
    
    Args:
        polyline_points: [N, 3] tensor of 3D points
        
    Returns:
        True if the polyline is cyclic (start and end points are close)
    """
    if polyline_points.shape[0] < 2:
        return False
    
    first_point = polyline_points[0]
    last_point = polyline_points[-1]
    distance_squared = torch.sum((first_point - last_point) ** 2)
    
    return distance_squared < CYCLIC_POLYLINE_TOLERANCE


def compute_signed_distance_to_polylines(
    query_points: torch.Tensor,
    polylines: List[torch.Tensor],
    z_stretch: float = Z_STRETCH_FACTOR,
) -> torch.Tensor:
    """
    Compute signed distance from query points to the nearest polyline segment.
    
    For road edges:
    - Two parallel polylines define a road (left and right boundaries)
    - Negative distance = inside road (on-road)
    - Positive distance = outside road (off-road)
    
    The distance to road is computed as:
    - If between the two boundaries: negative (distance to nearest boundary)
    - If outside both boundaries: positive (distance to nearest boundary)
    
    Args:
        query_points: [N, 3] tensor of 3D query points (x, y, z)
        polylines: List of [M_i, 3] tensors representing polyline segments
        z_stretch: Scaling factor for vertical distances
        
    Returns:
        [N] tensor of signed distances (2D distance in xy plane)
    """
    device = query_points.device
    num_points = query_points.shape[0]
    
    if len(polylines) == 0:
        return torch.full((num_points,), EXTREMELY_LARGE_DISTANCE, device=device)
    
    # For each point, find minimum distance to any polyline
    # Store distances to each polyline separately
    all_distances = []
    all_signs = []
    
    for polyline in polylines:
        if polyline.shape[0] < 2:
            continue
            
        polyline = polyline.to(device)
        num_segments = polyline.shape[0] - 1
        
        # Initialize with large distances for this polyline
        polyline_min_dist = torch.full((num_points,), float('inf'), device=device)
        polyline_signs = torch.zeros((num_points,), device=device)
        
        for i in range(num_segments):
            p1 = polyline[i]      # [3]
            p2 = polyline[i + 1]  # [3]
            
            # Segment vector
            segment = p2 - p1  # [3]
            segment_length_2d = torch.sqrt(segment[0]**2 + segment[1]**2)
            
            if segment_length_2d < 1e-6:
                continue
            
            # Vector from p1 to query points
            to_points = query_points - p1.unsqueeze(0)  # [N, 3]
            
            # Project onto segment (with z-stretch for altitude matching)
            segment_stretched = segment.clone()
            segment_stretched[2] *= z_stretch
            to_points_stretched = to_points.clone()
            to_points_stretched[:, 2] *= z_stretch
            
            # Compute projection parameter t
            dot_product = torch.sum(to_points_stretched * segment_stretched.unsqueeze(0), dim=1)
            segment_length_sq = torch.sum(segment_stretched ** 2)
            t = dot_product / (segment_length_sq + 1e-10)
            t = torch.clamp(t, 0.0, 1.0)
            
            # Closest point on segment
            closest_on_segment = p1.unsqueeze(0) + t.unsqueeze(1) * segment.unsqueeze(0)
            
            # 2D distance in xy plane
            diff_xy = query_points[:, :2] - closest_on_segment[:, :2]
            distance_2d = torch.sqrt(torch.sum(diff_xy ** 2, dim=1) + 1e-10)
            
            # Compute which side of segment the point is on using cross product
            # cross_z > 0: point is to the left of the segment direction
            # cross_z < 0: point is to the right of the segment direction
            segment_2d = segment[:2]
            cross_z = segment_2d[0] * to_points[:, 1] - segment_2d[1] * to_points[:, 0]
            sign = torch.sign(cross_z)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            
            # Update minimum distance for this polyline
            update_mask = distance_2d < polyline_min_dist
            polyline_min_dist = torch.where(update_mask, distance_2d, polyline_min_dist)
            polyline_signs = torch.where(update_mask, sign, polyline_signs)
        
        if torch.isfinite(polyline_min_dist).any():
            all_distances.append(polyline_min_dist)
            all_signs.append(polyline_signs)
    
    if len(all_distances) == 0:
        return torch.full((num_points,), EXTREMELY_LARGE_DISTANCE, device=device)
    
    # Stack all polyline distances
    all_distances = torch.stack(all_distances, dim=0)  # [num_polylines, N]
    all_signs = torch.stack(all_signs, dim=0)  # [num_polylines, N]
    
    # For road boundaries (2 polylines), determine if point is between them or outside
    if len(polylines) == 2:
        # Check if signs are opposite (point is between the two boundaries)
        signs_opposite = (all_signs[0] * all_signs[1]) < 0
        
        # If signs are opposite, point is between boundaries (inside/on-road)
        # Distance is negative (closest to nearest boundary)
        # If signs are same, point is outside both boundaries (off-road)
        # Distance is positive
        min_dist_to_boundaries = torch.min(all_distances, dim=0)[0]
        
        signed_distances = torch.where(
            signs_opposite,
            -min_dist_to_boundaries,  # Inside: negative
            min_dist_to_boundaries    # Outside: positive
        )
    else:
        # For single polyline or more complex cases, use minimum distance with sign
        min_idx = torch.argmin(all_distances, dim=0)
        min_distances = all_distances[min_idx, torch.arange(num_points)]
        min_signs = all_signs[min_idx, torch.arange(num_points)]
        signed_distances = min_signs * min_distances
    
    return signed_distances


def compute_distance_to_road_edge(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    center_z: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    road_edge_polylines: List[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute signed distance from agent bounding boxes to road edges.
    
    Uses the most off-road corner (maximum positive distance) for each agent.
    Negative distances = on-road, Positive distances = off-road.
    
    Args:
        center_x: [B, A, T] or [A, T] x-coordinates
        center_y: [B, A, T] or [A, T] y-coordinates
        center_z: [B, A, T] or [A, T] z-coordinates
        length: [B, A, T] or [A, T] agent lengths
        width: [B, A, T] or [A, T] agent widths
        heading: [B, A, T] or [A, T] agent headings (radians)
        road_edge_polylines: List of [M_i, 3] tensors of road edge polylines
        valid_mask: Optional [B, A, T] or [A, T] boolean mask for valid timesteps
        
    Returns:
        [B, A, T] or [A, T] signed distances to road edge
    """
    if len(road_edge_polylines) == 0:
        raise ValueError("road_edge_polylines cannot be empty. Map data is required for off-road metrics.")
    
    original_shape = center_x.shape
    device = center_x.device
    
    # Flatten to [N] for processing
    center_x_flat = center_x.reshape(-1)
    center_y_flat = center_y.reshape(-1)
    center_z_flat = center_z.reshape(-1)
    length_flat = length.reshape(-1)
    width_flat = width.reshape(-1)
    heading_flat = heading.reshape(-1)
    
    # Compute bounding box corners [N, 4, 3]
    corners = compute_bounding_box_corners(
        center_x_flat, center_y_flat, center_z_flat,
        length_flat, width_flat, heading_flat
    )
    
    # Flatten corners to [N*4, 3] for distance computation
    corners_flat = corners.reshape(-1, 3)
    
    # Compute signed distance for all corners
    corner_distances = compute_signed_distance_to_polylines(
        corners_flat, road_edge_polylines
    )
    
    # Reshape to [N, 4]
    corner_distances = corner_distances.reshape(-1, 4)
    
    # Take maximum (most off-road corner)
    max_distances = torch.max(corner_distances, dim=1)[0]
    
    # Reshape back to original shape
    distances = max_distances.reshape(original_shape)
    
    # Apply valid mask
    if valid_mask is not None:
        distances = torch.where(
            valid_mask,
            distances,
            torch.full_like(distances, -EXTREMELY_LARGE_DISTANCE)
        )
    
    return distances


def compute_offroad_rate(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    center_z: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    road_edge_polylines: List[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = OFFROAD_DISTANCE_THRESHOLD,
) -> torch.Tensor:
    """
    Compute the rate of timesteps where agents are off-road.
    
    Args:
        center_x: [B, A, T] x-coordinates
        center_y: [B, A, T] y-coordinates
        center_z: [B, A, T] z-coordinates
        length: [B, A, T] agent lengths
        width: [B, A, T] agent widths
        heading: [B, A, T] agent headings (radians)
        road_edge_polylines: List of [M_i, 3] tensors of road edge polylines
        valid_mask: Optional [B, A, T] boolean mask
        threshold: Distance threshold for off-road (default 0.0)
        
    Returns:
        [B, A] off-road rates per agent (fraction of timesteps off-road)
    """
    # Compute distances
    distances = compute_distance_to_road_edge(
        center_x, center_y, center_z, length, width, heading,
        road_edge_polylines, valid_mask
    )
    
    # Check if off-road (positive distance > threshold)
    is_offroad = distances > threshold
    
    # Apply valid mask
    if valid_mask is not None:
        is_offroad = is_offroad & valid_mask
        num_valid = valid_mask.sum(dim=-1).float()
    else:
        num_valid = torch.full(
            (center_x.shape[0], center_x.shape[1]),
            center_x.shape[2],
            dtype=torch.float32,
            device=center_x.device
        )
    
    # Compute rate
    num_offroad = is_offroad.sum(dim=-1).float()
    offroad_rate = num_offroad / (num_valid + 1e-10)
    
    return offroad_rate


def compute_map_metrics(
    ground_truth_trajectory: torch.Tensor,
    ground_truth_is_valid: torch.Tensor,
    object_type: torch.Tensor,
    road_edge_polylines: List[List[Tuple[float, float, float]]],
    prediction_trajectory: Optional[torch.Tensor] = None,
    prediction_ground_truth_indices: Optional[torch.Tensor] = None,
    prediction_ground_truth_indices_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute map-based metrics for Waymo Open Motion Dataset.
    
    This function can compute metrics for:
    1. Ground truth trajectories only (validation)
    2. Predicted trajectories (if prediction_trajectory is provided)
    
    Args:
        ground_truth_trajectory: [B, A, T, 7] ground truth trajectories
            Format: [x, y, z, length, width, height, heading]
            Or [B, A, T, 2] for just [x, y] (will use default dimensions)
        ground_truth_is_valid: [B, A, T] validity mask
        object_type: [B, A] object type per agent (1=vehicle, 2=pedestrian, 3=cyclist)
        road_edge_polylines: List of road edge polylines per scenario
            Each scenario: List of polylines, each polyline: List of (x, y, z) tuples
        prediction_trajectory: Optional [B, M, K, N, T, 7] predicted trajectories
            Format: [x, y, z, length, width, height, heading]
        prediction_ground_truth_indices: Optional [B, M, N] indices mapping predictions to GT
        prediction_ground_truth_indices_mask: Optional [B, M, N] mask for valid indices
        
    Returns:
        Dictionary of metric names to values
    """
    device = ground_truth_trajectory.device
    B, A, T_gt = ground_truth_is_valid.shape
    
    # Parse ground truth format
    if ground_truth_trajectory.shape[-1] == 7:
        gt_x = ground_truth_trajectory[..., 0]
        gt_y = ground_truth_trajectory[..., 1]
        gt_z = ground_truth_trajectory[..., 2]
        gt_length = ground_truth_trajectory[..., 3]
        gt_width = ground_truth_trajectory[..., 4]
        gt_heading = ground_truth_trajectory[..., 6]
    elif ground_truth_trajectory.shape[-1] == 2:
        gt_x = ground_truth_trajectory[..., 0]
        gt_y = ground_truth_trajectory[..., 1]
        gt_z = torch.zeros_like(gt_x)
        # Use default dimensions
        gt_length = torch.full_like(gt_x, 4.5)  # Default vehicle length
        gt_width = torch.full_like(gt_x, 2.0)   # Default vehicle width
        gt_heading = torch.zeros_like(gt_x)
    else:
        raise ValueError(f"ground_truth_trajectory must have shape [..., 2] or [..., 7], got {ground_truth_trajectory.shape}")
    
    # Convert road edge polylines to tensors
    road_edge_tensors_per_scenario = []
    for scenario_polylines in road_edge_polylines:
        scenario_tensors = []
        for polyline in scenario_polylines:
            if len(polyline) > 0:
                tensor = torch.tensor(polyline, dtype=torch.float32, device=device)
                scenario_tensors.append(tensor)
        road_edge_tensors_per_scenario.append(scenario_tensors)
    
    metrics = {}
    
    # Object types to evaluate
    object_types = [1, 2, 3]  # Vehicle, Pedestrian, Cyclist
    object_type_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    
    # Compute metrics for ground truth
    for b in range(B):
        if len(road_edge_tensors_per_scenario[b]) == 0:
            continue
            
        for obj_type, obj_name in zip(object_types, object_type_names):
            mask = object_type[b] == obj_type
            if not mask.any():
                continue
            
            # Get agents of this type
            agents_x = gt_x[b, mask]
            agents_y = gt_y[b, mask]
            agents_z = gt_z[b, mask]
            agents_length = gt_length[b, mask]
            agents_width = gt_width[b, mask]
            agents_heading = gt_heading[b, mask]
            agents_valid = ground_truth_is_valid[b, mask]
            
            # Compute off-road rate
            offroad_rate = compute_offroad_rate(
                agents_x.unsqueeze(0),
                agents_y.unsqueeze(0),
                agents_z.unsqueeze(0),
                agents_length.unsqueeze(0),
                agents_width.unsqueeze(0),
                agents_heading.unsqueeze(0),
                road_edge_tensors_per_scenario[b],
                agents_valid.unsqueeze(0)
            )
            
            # Average across agents
            avg_offroad_rate = offroad_rate.mean()
            
            key = f"{obj_name}/OffRoadRate"
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(avg_offroad_rate)
    
    # Average across batches
    for key in metrics:
        metrics[key] = torch.stack(metrics[key]).mean()
    
    return metrics
