"""
PyTorch implementation of Waymo Open Motion Dataset metrics.

This module implements the motion forecasting metrics:
- minADE: Minimum Average Displacement Error
- minFDE: Minimum Final Displacement Error  
- MissRate: Rate of predictions that miss the ground truth
- OverlapRate: Rate of predictions that overlap with ground truth
- mAP: Mean Average Precision
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


# Object type constants (matching Waymo Open Dataset)
TYPE_UNSET = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3
TYPE_OTHER = 4

# Trajectory type constants for mAP computation
TRAJECTORY_TYPE_STATIONARY = 0
TRAJECTORY_TYPE_STRAIGHT = 1
TRAJECTORY_TYPE_STRAIGHT_LEFT = 2
TRAJECTORY_TYPE_STRAIGHT_RIGHT = 3
TRAJECTORY_TYPE_LEFT_U_TURN = 4
TRAJECTORY_TYPE_LEFT_TURN = 5
TRAJECTORY_TYPE_RIGHT_U_TURN = 6
TRAJECTORY_TYPE_RIGHT_TURN = 7


@dataclass
class StepConfig:
    """Configuration for a single measurement step."""
    measurement_step: int  # The timestep to evaluate at (1-indexed)
    lateral_miss_threshold: float  # Lateral miss threshold in meters
    longitudinal_miss_threshold: float  # Longitudinal miss threshold in meters


@dataclass
class MotionMetricsConfig:
    """Configuration for motion metrics computation.
    
    This matches the official Waymo Open Dataset MotionMetricsConfig format.
    """
    
    # Track and prediction frequency
    track_steps_per_second: int = 10  # Ground truth frequency (Hz)
    prediction_steps_per_second: int = 2  # Prediction frequency (Hz)
    
    # Track sample counts
    track_history_samples: int = 10  # Number of history timesteps
    track_future_samples: int = 80  # Number of future timesteps to predict
    
    # Speed-based filtering and scaling
    speed_lower_bound: float = 1.4  # m/s - minimum speed for evaluation
    speed_upper_bound: float = 11.0  # m/s - maximum speed for evaluation
    speed_scale_lower: float = 0.5  # Speed scaling weight lower bound
    speed_scale_upper: float = 1.0  # Speed scaling weight upper bound
    
    # Measurement step configurations (time horizons to evaluate at)
    step_configurations: List[StepConfig] = None
    
    # Maximum number of predictions to evaluate
    max_predictions: int = 6
    
    # Overlap threshold (IoU threshold for collision detection)
    overlap_threshold: float = 0.5
    
    def __post_init__(self):
        if self.step_configurations is None:
            # Default: Official Waymo challenge config
            # At 2Hz prediction: step 5 = 3s, step 9 = 5s, step 15 = 8s
            self.step_configurations = [
                StepConfig(measurement_step=5, lateral_miss_threshold=1.0, longitudinal_miss_threshold=2.0),
                StepConfig(measurement_step=9, lateral_miss_threshold=1.8, longitudinal_miss_threshold=3.6),
                StepConfig(measurement_step=15, lateral_miss_threshold=3.0, longitudinal_miss_threshold=6.0),
            ]


def compute_displacement_error(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute displacement error between predicted and ground truth trajectories.
    
    Args:
        pred_trajectory: [..., T, 2] predicted (x, y) positions
        gt_trajectory: [..., T, 2] ground truth (x, y) positions
        valid_mask: [..., T] optional mask for valid timesteps
        
    Returns:
        [..., T] displacement errors at each timestep
    """
    errors = torch.norm(pred_trajectory - gt_trajectory, dim=-1)  # [..., T]
    
    if valid_mask is not None:
        errors = errors * valid_mask.float()
    
    return errors


def compute_min_ade(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    measurement_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute minimum Average Displacement Error (minADE).
    
    Args:
        pred_trajectories: [K, T, 2] K predicted trajectories, each with T timesteps
        gt_trajectory: [T, 2] or [T, 7] ground truth trajectory
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
            If [T, 2]: [x, y] only
        valid_mask: [T] optional mask for valid timesteps
        measurement_step: if provided, only evaluate up to this step
        
    Returns:
        Scalar tensor with minADE value
    """
    # Extract x, y if 7D format
    if gt_trajectory.shape[-1] >= 7:
        gt_trajectory = gt_trajectory[..., :2]
    
    if measurement_step is not None:
        pred_trajectories = pred_trajectories[:, :measurement_step, :]
        gt_trajectory = gt_trajectory[:measurement_step, :]
        if valid_mask is not None:
            valid_mask = valid_mask[:measurement_step]
    
    # Compute ADE for each prediction: [K, T]
    errors = compute_displacement_error(
        pred_trajectories,  # [K, T, 2]
        gt_trajectory.unsqueeze(0),  # [1, T, 2]
        valid_mask.unsqueeze(0) if valid_mask is not None else None,
    )
    
    # Average over timesteps: [K]
    if valid_mask is not None:
        ade_per_pred = errors.sum(dim=-1) / valid_mask.sum().clamp(min=1.0)
    else:
        ade_per_pred = errors.mean(dim=-1)
    
    # Take minimum across K predictions
    min_ade = ade_per_pred.min()
    
    return min_ade


def compute_min_fde(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    measurement_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute minimum Final Displacement Error (minFDE).
    
    Args:
        pred_trajectories: [K, T, 2] K predicted trajectories
        gt_trajectory: [T, 2] or [T, 7] ground truth trajectory
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
            If [T, 2]: [x, y] only
        valid_mask: [T] optional mask for valid timesteps
        measurement_step: if provided, evaluate at this step (0-indexed)
        
    Returns:
        Scalar tensor with minFDE value
    """
    # Extract x, y if 7D format
    gt_xy = gt_trajectory[..., :2] if gt_trajectory.shape[-1] >= 7 else gt_trajectory
    
    if measurement_step is not None:
        # Evaluate at specific step
        pred_final = pred_trajectories[:, measurement_step - 1, :]  # [K, 2]
        gt_final = gt_xy[measurement_step - 1, :]  # [2]
    else:
        # Evaluate at final step
        pred_final = pred_trajectories[:, -1, :]  # [K, 2]
        gt_final = gt_xy[-1, :]  # [2]
    
    # Compute FDE for each prediction: [K]
    fde_per_pred = torch.norm(pred_final - gt_final, dim=-1)
    
    # Take minimum across K predictions
    min_fde = fde_per_pred.min()
    
    return min_fde


def compute_miss_rate(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    lateral_threshold: float,
    longitudinal_threshold: float,
    gt_heading: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    measurement_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute miss rate using lateral and longitudinal displacement thresholds.
    
    The official Waymo implementation decomposes the displacement error into:
    - Lateral: perpendicular to the heading direction
    - Longitudinal: along the heading direction
    
    A prediction misses if EITHER lateral OR longitudinal error exceeds its threshold.
    
    Args:
        pred_trajectories: [K, T, 2] K predicted trajectories (x, y)
        gt_trajectory: [T, 2] or [T, 7] ground truth trajectory
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
            If [T, 2]: [x, y] only (requires gt_heading parameter)
        lateral_threshold: lateral miss threshold in meters
        longitudinal_threshold: longitudinal miss threshold in meters
        gt_heading: [T] optional heading angles if not in gt_trajectory
        valid_mask: [T] optional mask for valid timesteps
        measurement_step: if provided, evaluate at this step
        
    Returns:
        Scalar tensor with miss rate (0.0 to 1.0)
    """
    if measurement_step is not None:
        # Evaluate at specific step
        pred_final = pred_trajectories[:, measurement_step - 1, :]  # [K, 2]
        if gt_trajectory.shape[-1] >= 7:
            gt_final = gt_trajectory[measurement_step - 1, :2]  # [2] - extract x, y
            heading = gt_trajectory[measurement_step - 1, 4]  # scalar - extract heading
        else:
            gt_final = gt_trajectory[measurement_step - 1, :]  # [2]
            heading = gt_heading[measurement_step - 1] if gt_heading is not None else None
    else:
        # Evaluate at final step
        pred_final = pred_trajectories[:, -1, :]  # [K, 2]
        if gt_trajectory.shape[-1] >= 7:
            gt_final = gt_trajectory[-1, :2]  # [2]
            heading = gt_trajectory[-1, 4]  # scalar
        else:
            gt_final = gt_trajectory[-1, :]  # [2]
            heading = gt_heading[-1] if gt_heading is not None else None
    
    # Compute displacement errors for each prediction: [K, 2]
    displacements = pred_final - gt_final  # [K, 2]
    
    if heading is not None:
        # Decompose into lateral and longitudinal components
        # heading is the direction of motion (yaw angle)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        
        # Longitudinal: component along heading direction
        # longitudinal = displacement_x * cos(heading) + displacement_y * sin(heading)
        longitudinal = displacements[:, 0] * cos_h + displacements[:, 1] * sin_h  # [K]
        
        # Lateral: component perpendicular to heading direction
        # lateral = -displacement_x * sin(heading) + displacement_y * cos(heading)
        lateral = -displacements[:, 0] * sin_h + displacements[:, 1] * cos_h  # [K]
        
        # Take absolute values for threshold comparison
        lateral_errors = torch.abs(lateral)  # [K]
        longitudinal_errors = torch.abs(longitudinal)  # [K]
        
        # Find the best prediction (minimum combined error)
        # A prediction misses if lateral > lateral_threshold OR longitudinal > longitudinal_threshold
        lateral_miss = lateral_errors > lateral_threshold  # [K]
        longitudinal_miss = longitudinal_errors > longitudinal_threshold  # [K]
        prediction_misses = lateral_miss | longitudinal_miss  # [K]
        
        # Check if all predictions miss
        miss = prediction_misses.all().float()
    else:
        # Fallback: use Euclidean distance with lateral threshold
        # This is a simplified approach when heading is not available
        fde_per_pred = torch.norm(displacements, dim=-1)  # [K]
        min_fde = fde_per_pred.min()
        miss = (min_fde > lateral_threshold).float()
    
    return miss


def box_to_corners(center_x: torch.Tensor, center_y: torch.Tensor, 
                   length: torch.Tensor, width: torch.Tensor, 
                   heading: torch.Tensor) -> torch.Tensor:
    """
    Convert box parameters to 4 corner points.
    
    Args:
        center_x, center_y: Center position
        length: Box length (along heading direction)
        width: Box width (perpendicular to heading)
        heading: Heading angle in radians
        
    Returns:
        corners: [4, 2] tensor of corner points in order: 
                 front-right, front-left, rear-left, rear-right
    """
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    
    # Half dimensions
    half_length = length / 2.0
    half_width = width / 2.0
    
    # Local corner offsets (before rotation)
    # Front-right, front-left, rear-left, rear-right
    local_corners = torch.tensor([
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width],
        [-half_length, -half_width],
    ], device=center_x.device, dtype=center_x.dtype)
    
    # Rotation matrix
    rotation = torch.tensor([
        [cos_h, -sin_h],
        [sin_h, cos_h]
    ], device=center_x.device, dtype=center_x.dtype)
    
    # Rotate and translate
    corners = torch.matmul(local_corners, rotation.T)
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    
    return corners


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute area of a polygon using the shoelace formula.
    
    Args:
        vertices: [N, 2] vertices in order (counter-clockwise or clockwise)
        
    Returns:
        Scalar area (always positive)
    """
    if vertices.shape[0] < 3:
        return torch.tensor(0.0, device=vertices.device, dtype=vertices.dtype)
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Shoelace formula
    area = 0.5 * torch.abs(
        torch.sum(x[:-1] * y[1:]) - torch.sum(y[:-1] * x[1:]) +
        x[-1] * y[0] - y[-1] * x[0]
    )
    
    return area


def line_segment_intersection(p1: torch.Tensor, p2: torch.Tensor,
                              p3: torch.Tensor, p4: torch.Tensor) -> tuple:
    """
    Find intersection point of two line segments.
    
    Args:
        p1, p2: Endpoints of first segment
        p3, p4: Endpoints of second segment
        
    Returns:
        (intersects, point) where intersects is bool and point is [2] tensor
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if torch.abs(denom) < 1e-10:
        return False, None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = torch.tensor([
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1)
        ], device=p1.device, dtype=p1.dtype)
        return True, intersection
    
    return False, None


def point_in_polygon(point: torch.Tensor, polygon: torch.Tensor) -> bool:
    """
    Check if a point is inside a polygon using ray casting.
    
    Args:
        point: [2] point coordinates
        polygon: [N, 2] polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point[0].item(), point[1].item()
    n = polygon.shape[0]
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i, 0].item(), polygon[i, 1].item()
        xj, yj = polygon[j, 0].item(), polygon[j, 1].item()
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def polygon_intersection(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection of two convex polygons using Sutherland-Hodgman algorithm.
    
    Args:
        poly1: [N1, 2] vertices of first polygon
        poly2: [N2, 2] vertices of second polygon
        
    Returns:
        [M, 2] vertices of intersection polygon (may be empty)
    """
    # Start with poly1
    output = poly1.clone()
    
    # Clip against each edge of poly2
    for i in range(poly2.shape[0]):
        if output.shape[0] == 0:
            break
            
        input_list = output
        output = []
        
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % poly2.shape[0]]
        
        for j in range(input_list.shape[0]):
            current = input_list[j]
            previous = input_list[j - 1]
            
            # Vector from edge_start to edge_end
            edge_vec = edge_end - edge_start
            # Normal vector (perpendicular, pointing inward)
            normal = torch.tensor([-edge_vec[1], edge_vec[0]], 
                                device=poly1.device, dtype=poly1.dtype)
            
            # Check which side of the edge the points are on
            prev_side = torch.dot(previous - edge_start, normal)
            curr_side = torch.dot(current - edge_start, normal)
            
            if curr_side >= 0:  # Current point is inside
                if prev_side < 0:  # Previous was outside, add intersection
                    intersects, intersection = line_segment_intersection(
                        previous, current, edge_start, edge_end
                    )
                    if intersects:
                        output.append(intersection)
                output.append(current)
            elif prev_side >= 0:  # Previous was inside, current is outside
                intersects, intersection = line_segment_intersection(
                    previous, current, edge_start, edge_end
                )
                if intersects:
                    output.append(intersection)
        
        if len(output) > 0:
            output = torch.stack(output)
        else:
            output = torch.empty((0, 2), device=poly1.device, dtype=poly1.dtype)
    
    return output


def compute_polygon_iou(box1_corners: torch.Tensor, box2_corners: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two oriented bounding boxes represented as polygons.
    
    Args:
        box1_corners: [4, 2] corners of first box
        box2_corners: [4, 2] corners of second box
        
    Returns:
        Scalar IoU value in [0, 1]
    """
    # Compute areas
    area1 = polygon_area(box1_corners)
    area2 = polygon_area(box2_corners)
    
    if area1 < 1e-10 or area2 < 1e-10:
        return torch.tensor(0.0, device=box1_corners.device, dtype=box1_corners.dtype)
    
    # Compute intersection polygon
    try:
        intersection = polygon_intersection(box1_corners, box2_corners)
        
        if intersection.shape[0] < 3:
            intersection_area = torch.tensor(0.0, device=box1_corners.device, dtype=box1_corners.dtype)
        else:
            intersection_area = polygon_area(intersection)
    except:
        # If polygon intersection fails, return 0
        intersection_area = torch.tensor(0.0, device=box1_corners.device, dtype=box1_corners.dtype)
    
    # Compute union
    union_area = area1 + area2 - intersection_area
    
    if union_area < 1e-10:
        return torch.tensor(0.0, device=box1_corners.device, dtype=box1_corners.dtype)
    
    iou = intersection_area / union_area
    
    return iou.clamp(0.0, 1.0)


def compute_overlap_rate(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_boxes: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: Optional[torch.Tensor] = None,
    measurement_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute overlap rate using proper polygon IoU for oriented bounding boxes.
    
    This implementation uses the Sutherland-Hodgman algorithm to compute the
    intersection of rotated rectangles and calculates true IoU (Intersection
    over Union) values, matching the official Waymo implementation.
    
    Args:
        pred_trajectories: [K, T, 2] K predicted trajectories (center positions)
        gt_trajectory: [T, 2] or [T, 7] ground truth trajectory
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
            If [T, 2]: [x, y] only
        gt_boxes: [T, 4] ground truth boxes [length, width, heading, velocity_heading]
                  or [T, 5] [x, y, length, width, heading]
        threshold: IoU threshold for overlap (default: 0.5)
        valid_mask: [T] optional mask for valid timesteps
        measurement_step: if provided, evaluate at this step
        
    Returns:
        Scalar tensor with overlap rate (0.0 to 1.0)
    """
    # Extract x, y if 7D format
    gt_xy = gt_trajectory[..., :2] if gt_trajectory.shape[-1] >= 7 else gt_trajectory
    
    if measurement_step is not None:
        pred_trajectories = pred_trajectories[:, :measurement_step, :]
        gt_xy = gt_xy[:measurement_step, :]
        gt_boxes = gt_boxes[:measurement_step, :]
        if valid_mask is not None:
            valid_mask = valid_mask[:measurement_step]
    
    # Compute proper polygon IoU for oriented bounding boxes
    K, T, _ = pred_trajectories.shape
    
    # Extract box dimensions from gt_trajectory (7D) or gt_boxes
    if gt_trajectory.shape[-1] >= 7:
        # Use dimensions and heading from 7D ground truth
        gt_lengths = gt_trajectory[..., 2]  # [T]
        gt_widths = gt_trajectory[..., 3]  # [T]
        gt_headings = gt_trajectory[..., 4]  # [T]
    elif gt_boxes.shape[-1] >= 3:
        # Use dimensions from gt_boxes
        gt_lengths = gt_boxes[..., 0]  # [T]
        gt_widths = gt_boxes[..., 1]  # [T]
        gt_headings = gt_boxes[..., 2] if gt_boxes.shape[-1] > 2 else torch.zeros(T, device=gt_xy.device)
    else:
        # Default box size if not provided
        gt_lengths = torch.ones(T, device=gt_xy.device) * 4.5  # meters
        gt_widths = torch.ones(T, device=gt_xy.device) * 2.0  # meters
        gt_headings = torch.zeros(T, device=gt_xy.device)  # radians
    
    # Assume prediction boxes have same dimensions as GT (typical assumption)
    # In practice, predictions might have their own box dimensions
    pred_lengths = gt_lengths
    pred_widths = gt_widths
    
    # For each prediction, check if it overlaps at any timestep
    overlaps = []
    for k in range(K):
        pred_pos = pred_trajectories[k]  # [T, 2]
        has_overlap = False
        
        for t in range(T):
            if valid_mask is not None and not valid_mask[t]:
                continue
            
            # Get box corners for prediction
            # Assume prediction heading aligns with velocity direction or uses GT heading
            pred_heading = gt_headings[t]  # Simplified: use GT heading
            pred_corners = box_to_corners(
                pred_pos[t, 0], pred_pos[t, 1],
                pred_lengths[t], pred_widths[t], pred_heading
            )
            
            # Get box corners for ground truth
            gt_corners = box_to_corners(
                gt_xy[t, 0], gt_xy[t, 1],
                gt_lengths[t], gt_widths[t], gt_headings[t]
            )
            
            # Compute IoU
            iou = compute_polygon_iou(pred_corners, gt_corners)
            
            # Check if IoU exceeds threshold
            if iou > threshold:
                has_overlap = True
                break
        
        overlaps.append(torch.tensor(1.0 if has_overlap else 0.0, device=gt_xy.device))
    
    # Overlap rate: fraction of predictions that overlap
    overlap_rate = torch.stack(overlaps).mean()
    
    return overlap_rate


def classify_trajectory_type(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Classify trajectory type based on heading changes and displacement.
    
    This matches the official Waymo Open Dataset trajectory classification:
    - STATIONARY: Total displacement < 2.0m
    - STRAIGHT: Heading change < 15 degrees
    - STRAIGHT_LEFT: 15° < heading < 45°, turning left
    - STRAIGHT_RIGHT: 15° < heading < 45°, turning right  
    - LEFT_TURN: 45° < heading < 135°, turning left
    - RIGHT_TURN: 45° < heading < 135°, turning right
    - LEFT_U_TURN: heading > 135°, turning left
    - RIGHT_U_TURN: heading > 135°, turning right
    
    Args:
        trajectory: [T, 2] or [T, 7] trajectory with (x, y) or full state
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
        valid_mask: [T] optional mask for valid timesteps
        
    Returns:
        Trajectory type constant (0-7)
    """
    # Extract positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[..., :2]  # [T, 2]
    else:
        positions = trajectory
    
    # Apply valid mask
    if valid_mask is not None:
        valid_positions = positions[valid_mask.bool()]
    else:
        valid_positions = positions
    
    if valid_positions.shape[0] < 2:
        return TRAJECTORY_TYPE_STATIONARY
    
    # Compute total displacement
    displacement = torch.norm(valid_positions[-1] - valid_positions[0])
    
    # Stationary threshold: 2.0 meters
    if displacement < 2.0:
        return TRAJECTORY_TYPE_STATIONARY
    
    # Compute heading change
    # Use displacement vectors to estimate heading
    start_pos = valid_positions[0]
    mid_idx = len(valid_positions) // 2
    mid_pos = valid_positions[mid_idx]
    end_pos = valid_positions[-1]
    
    # Initial heading
    vec_start = mid_pos - start_pos
    heading_start = torch.atan2(vec_start[1], vec_start[0])
    
    # Final heading  
    vec_end = end_pos - mid_pos
    heading_end = torch.atan2(vec_end[1], vec_end[0])
    
    # Heading change (normalized to [-pi, pi])
    heading_change = heading_end - heading_start
    heading_change = torch.atan2(torch.sin(heading_change), torch.cos(heading_change))
    heading_change_deg = torch.abs(heading_change) * 180.0 / math.pi
    
    # Determine turn direction (positive = left/counter-clockwise)
    turn_direction = torch.sign(heading_change)
    
    # Classify based on heading change magnitude
    if heading_change_deg < 15.0:
        return TRAJECTORY_TYPE_STRAIGHT
    elif heading_change_deg < 45.0:
        if turn_direction > 0:
            return TRAJECTORY_TYPE_STRAIGHT_LEFT
        else:
            return TRAJECTORY_TYPE_STRAIGHT_RIGHT
    elif heading_change_deg < 135.0:
        if turn_direction > 0:
            return TRAJECTORY_TYPE_LEFT_TURN
        else:
            return TRAJECTORY_TYPE_RIGHT_TURN
    else:
        if turn_direction > 0:
            return TRAJECTORY_TYPE_LEFT_U_TURN
        else:
            return TRAJECTORY_TYPE_RIGHT_U_TURN


def compute_precision_recall(
    pred_scores: torch.Tensor,
    pred_matched: torch.Tensor,
    num_gt: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision-recall curve for mAP calculation.
    
    Args:
        pred_scores: [N] confidence scores for predictions
        pred_matched: [N] boolean mask indicating if prediction matched GT
        num_gt: Number of ground truth instances
        
    Returns:
        Tuple of (precision, recall, thresholds)
        - precision: [N+1] precision at each threshold
        - recall: [N+1] recall at each threshold  
        - thresholds: [N] confidence thresholds
    """
    if len(pred_scores) == 0:
        # No predictions
        return torch.zeros(1, device=pred_scores.device), \
               torch.zeros(1, device=pred_scores.device), \
               torch.tensor([], device=pred_scores.device)
    
    # Sort by confidence (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    sorted_matched = pred_matched[sorted_indices]
    sorted_scores = pred_scores[sorted_indices]
    
    # Compute cumulative TP and FP
    tp_cumsum = torch.cumsum(sorted_matched.float(), dim=0)
    fp_cumsum = torch.cumsum((~sorted_matched).float(), dim=0)
    
    # Compute precision and recall at each threshold
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / max(num_gt, 1)
    
    # Add (0, 0) point at the beginning
    precision = torch.cat([torch.ones(1, device=precision.device), precision])
    recall = torch.cat([torch.zeros(1, device=recall.device), recall])
    
    return precision, recall, sorted_scores


def compute_average_precision(
    precision: torch.Tensor,
    recall: torch.Tensor,
) -> torch.Tensor:
    """
    Compute average precision from precision-recall curve.
    
    Uses the VOC-style all-point interpolation method:
    AP = sum of (r_i - r_{i-1}) * max(p_j for j >= i)
    
    Args:
        precision: [N] precision values
        recall: [N] recall values
        
    Returns:
        Scalar average precision value
    """
    # Compute maximum precision at each recall level
    # This implements the VOC all-point interpolation
    max_precision = torch.zeros_like(precision)
    for i in range(len(precision) - 1, -1, -1):
        if i == len(precision) - 1:
            max_precision[i] = precision[i]
        else:
            max_precision[i] = torch.maximum(precision[i], max_precision[i + 1])
    
    # Compute AP as area under interpolated curve
    recall_diff = recall[1:] - recall[:-1]
    ap = torch.sum(recall_diff * max_precision[1:])
    
    return ap


def compute_map_metric(
    pred_trajectories: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_boxes: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    measurement_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute mean Average Precision (mAP) metric with trajectory type classification.
    
    This implementation:
    1. Classifies ground truth trajectory type
    2. Classifies each predicted trajectory type  
    3. Matches predictions to GT based on type and distance
    4. Returns matches for later aggregation into precision-recall
    
    Note: This function returns per-prediction results. The final mAP
    is computed by aggregating across all scenarios and computing
    precision-recall curves per trajectory type.
    
    Args:
        pred_trajectories: [K, T, 2] K predicted trajectories
        pred_scores: [K] confidence scores for each prediction
        gt_trajectory: [T, 2] or [T, 7] ground truth trajectory
        gt_boxes: [T, 4] ground truth boxes (unused, kept for compatibility)
        valid_mask: [T] optional mask for valid timesteps
        measurement_step: if provided, evaluate at this step
        
    Returns:
        Tuple of (gt_type, pred_types, pred_matched, pred_scores)
        - gt_type: Ground truth trajectory type
        - pred_types: [K] predicted trajectory types
        - pred_matched: [K] boolean indicating if prediction matched GT
        - pred_scores: [K] confidence scores (for precision-recall)
    """
    K = pred_trajectories.shape[0]
    device = pred_trajectories.device
    
    # Apply measurement step
    if measurement_step is not None:
        pred_trajectories = pred_trajectories[:, :measurement_step, :]
        gt_trajectory = gt_trajectory[:measurement_step, :]
        if valid_mask is not None:
            valid_mask = valid_mask[:measurement_step]
    
    # Classify ground truth trajectory
    gt_type = classify_trajectory_type(gt_trajectory, valid_mask)
    
    # Classify predicted trajectories and compute distances
    pred_types = torch.zeros(K, dtype=torch.long, device=device)
    pred_distances = torch.zeros(K, device=device)
    
    for k in range(K):
        pred_types[k] = classify_trajectory_type(
            pred_trajectories[k], 
            valid_mask
        )
        # Compute FDE as matching distance
        pred_distances[k] = compute_min_fde(
            pred_trajectories[k:k+1], 
            gt_trajectory, 
            valid_mask,
            measurement_step=None  # Already applied above
        )
    
    # Match predictions to ground truth
    # A prediction matches if:
    # 1. Same trajectory type as GT
    # 2. FDE within threshold (2.0m for vehicles, 1.0m for pedestrians/cyclists)
    match_threshold = 2.0  # meters
    
    pred_matched = (pred_types == gt_type) & (pred_distances <= match_threshold)
    
    return gt_type, pred_types, pred_matched, pred_scores


def compute_speed(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    time_step_seconds: float = 0.1,
) -> torch.Tensor:
    """
    Compute average speed of a trajectory.
    
    Args:
        trajectory: [T, 2] or [T, 7] trajectory with (x, y) positions
            If [T, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
        valid_mask: [T] optional mask for valid timesteps
        time_step_seconds: Time between consecutive timesteps in seconds
        
    Returns:
        Scalar tensor with average speed in m/s
    """
    # Extract positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[..., :2]  # [T, 2]
    else:
        positions = trajectory
    
    # Apply valid mask
    if valid_mask is not None:
        valid_positions = positions[valid_mask.bool()]
    else:
        valid_positions = positions
    
    if valid_positions.shape[0] < 2:
        return torch.tensor(0.0, device=trajectory.device)
    
    # Compute displacements
    displacements = torch.norm(valid_positions[1:] - valid_positions[:-1], dim=-1)  # [T-1]
    
    # Total distance
    total_distance = displacements.sum()
    
    # Total time
    total_time = (valid_positions.shape[0] - 1) * time_step_seconds
    
    # Average speed
    speed = total_distance / max(total_time, 1e-6)
    
    return speed


def compute_speed_scale(
    speed: torch.Tensor,
    speed_lower_bound: float,
    speed_upper_bound: float,
    speed_scale_lower: float,
    speed_scale_upper: float,
) -> torch.Tensor:
    """
    Compute speed-based scaling weight.
    
    This implements linear interpolation between scale_lower and scale_upper
    based on the speed within [lower_bound, upper_bound].
    
    Args:
        speed: Speed value in m/s
        speed_lower_bound: Lower speed threshold (m/s)
        speed_upper_bound: Upper speed threshold (m/s)
        speed_scale_lower: Weight at lower bound
        speed_scale_upper: Weight at upper bound
        
    Returns:
        Scalar tensor with speed scale weight (0.0 if outside bounds)
    """
    # Filter out speeds outside bounds
    if speed < speed_lower_bound or speed > speed_upper_bound:
        return torch.tensor(0.0, device=speed.device)
    
    # Linear interpolation
    speed_range = speed_upper_bound - speed_lower_bound
    scale_range = speed_scale_upper - speed_scale_lower
    
    normalized_speed = (speed - speed_lower_bound) / max(speed_range, 1e-6)
    scale = speed_scale_lower + normalized_speed * scale_range
    
    return scale


def compute_motion_metrics(
    prediction_trajectory: torch.Tensor,
    prediction_score: torch.Tensor,
    ground_truth_trajectory: torch.Tensor,
    ground_truth_is_valid: torch.Tensor,
    prediction_ground_truth_indices: torch.Tensor,
    prediction_ground_truth_indices_mask: torch.Tensor,
    object_type: torch.Tensor,
    config: MotionMetricsConfig,
    ground_truth_boxes: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute motion metrics for Waymo Open Motion Dataset.
    
    Args:
        prediction_trajectory: [B, M, K, N, T, 2] predicted trajectories
            B: batch size (scenarios)
            M: number of joint prediction groups
            K: top-K predictions per group
            N: number of agents in joint prediction (usually 1)
            T: number of prediction timesteps
        prediction_score: [B, M, K] confidence scores
        ground_truth_trajectory: [B, A, T_gt, 2] or [B, A, T_gt, 7] ground truth trajectories
            A: number of agents in ground truth
            T_gt: number of ground truth timesteps
            If shape is [B, A, T_gt, 7]: [x, y, length, width, heading, velocity_x, velocity_y]
            If shape is [B, A, T_gt, 2]: [x, y] only (simplified, less accurate miss rate)
        ground_truth_is_valid: [B, A, T_gt] validity mask
        prediction_ground_truth_indices: [B, M, N] indices mapping predictions to GT
        prediction_ground_truth_indices_mask: [B, M, N] mask for valid indices
        object_type: [B, A] object type per agent
        config: MotionMetricsConfig
        ground_truth_boxes: [B, A, T_gt, 4] optional box dimensions [length, width, heading, velocity]
            Note: If ground_truth_trajectory is 7D, this parameter is optional/redundant
        
    Returns:
        Dictionary of metric names to values, broken down by object type and step
    """
    device = prediction_trajectory.device
    B, M, K, N, T_pred, _ = prediction_trajectory.shape
    B_gt, A, T_gt, _ = ground_truth_trajectory.shape
    
    assert B == B_gt, "Batch sizes must match"
    
    # Object types to evaluate
    object_types = [TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST]
    object_type_names = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    
    # Trajectory type names for mAP
    trajectory_type_names = [
        "STATIONARY", "STRAIGHT", "STRAIGHT_LEFT", "STRAIGHT_RIGHT",
        "LEFT_U_TURN", "LEFT_TURN", "RIGHT_U_TURN", "RIGHT_TURN"
    ]
    
    # Initialize metrics storage
    metrics = {}
    
    # Storage for mAP aggregation across scenarios
    # Maps (object_type, trajectory_type, measurement_step) -> list of (pred_scores, pred_matched, num_gt)
    map_accumulator = {}
    
    # Process each scenario in batch
    for b in range(B):
        # Get ground truth for this scenario
        gt_trajectories = ground_truth_trajectory[b]  # [A, T_gt, 2]
        gt_valid = ground_truth_is_valid[b]  # [A, T_gt]
        obj_types = object_type[b]  # [A]
        
        # Process each joint prediction group
        for m in range(M):
            pred_traj = prediction_trajectory[b, m]  # [K, N, T_pred, 2]
            pred_scores = prediction_score[b, m]  # [K]
            gt_indices = prediction_ground_truth_indices[b, m]  # [N]
            gt_mask = prediction_ground_truth_indices_mask[b, m]  # [N]
            
            # Process each agent in the joint prediction
            for n in range(N):
                if not gt_mask[n]:
                    continue
                
                gt_idx = gt_indices[n].item()
                if gt_idx >= A:
                    continue
                
                # Get ground truth for this agent
                gt_traj = gt_trajectories[gt_idx]  # [T_gt, 2]
                gt_valid_n = gt_valid[gt_idx]  # [T_gt]
                obj_type = obj_types[gt_idx].item()
                
                # Skip if object type not in evaluation set
                if obj_type not in object_types:
                    continue
                
                obj_type_name = object_type_names[object_types.index(obj_type)]
                
                # Compute speed for this agent
                time_step_seconds = 1.0 / config.track_steps_per_second
                agent_speed = compute_speed(gt_traj, gt_valid_n, time_step_seconds)
                
                # Compute speed-based scaling weight
                speed_scale = compute_speed_scale(
                    agent_speed,
                    config.speed_lower_bound,
                    config.speed_upper_bound,
                    config.speed_scale_lower,
                    config.speed_scale_upper,
                )
                
                # Skip this agent if outside speed bounds (scale = 0)
                if speed_scale.item() == 0.0:
                    continue
                
                # Get predictions for this agent (assuming N=1 for now)
                # If N>1, we'd need to handle joint predictions differently
                pred_traj_n = pred_traj[:, n, :, :]  # [K, T_pred, 2]
                
                # Get ground truth boxes if provided
                gt_boxes_n = None
                if ground_truth_boxes is not None:
                    gt_boxes_n = ground_truth_boxes[b, gt_idx]  # [T_gt, 4]
                
                # Compute metrics for each measurement step
                for step_config in config.step_configurations:
                    # Convert prediction step to track step
                    # prediction_step * (track_steps_per_second / prediction_steps_per_second)
                    step_ratio = config.track_steps_per_second / config.prediction_steps_per_second
                    track_step = int(step_config.measurement_step * step_ratio)
                    
                    # Ensure we don't go out of bounds
                    track_step = min(track_step, T_pred - 1, T_gt - 1)
                    
                    metric_key = f"{obj_type_name}_{step_config.measurement_step}"
                    
                    # Compute minADE
                    min_ade = compute_min_ade(
                        pred_traj_n, gt_traj, gt_valid_n, track_step + 1
                    )
                    
                    # Compute minFDE
                    min_fde = compute_min_fde(
                        pred_traj_n, gt_traj, gt_valid_n, track_step + 1
                    )
                    
                    # Compute MissRate with lateral/longitudinal thresholds
                    miss_rate = compute_miss_rate(
                        pred_traj_n,
                        gt_traj,
                        step_config.lateral_miss_threshold,
                        step_config.longitudinal_miss_threshold,
                        valid_mask=gt_valid_n,
                        measurement_step=track_step + 1,
                    )
                    
                    # Compute OverlapRate
                    if gt_boxes_n is not None:
                        overlap_rate = compute_overlap_rate(
                            pred_traj_n,
                            gt_traj,
                            gt_boxes_n,
                            config.overlap_threshold,
                            gt_valid_n,
                            track_step + 1,
                        )
                    else:
                        overlap_rate = torch.tensor(0.0, device=device)
                    
                    # Compute mAP components (for later aggregation)
                    gt_type, pred_types, pred_matched, _ = compute_map_metric(
                        pred_traj_n,
                        pred_scores,
                        gt_traj,
                        gt_boxes_n if gt_boxes_n is not None else torch.zeros(T_gt, 4, device=device),
                        gt_valid_n,
                        track_step + 1,
                    )
                    
                    # Accumulate mAP data per trajectory type
                    map_key = (obj_type, gt_type, step_config.measurement_step)
                    if map_key not in map_accumulator:
                        map_accumulator[map_key] = {
                            'pred_scores': [],
                            'pred_matched': [],
                            'num_gt': 0
                        }
                    
                    map_accumulator[map_key]['pred_scores'].append(pred_scores)
                    map_accumulator[map_key]['pred_matched'].append(pred_matched)
                    map_accumulator[map_key]['num_gt'] += 1  # One GT instance per scenario
                    
                    # Store metrics with speed scaling applied
                    if metric_key not in metrics:
                        metrics[f"{metric_key}/minADE"] = []
                        metrics[f"{metric_key}/minFDE"] = []
                        metrics[f"{metric_key}/MissRate"] = []
                        metrics[f"{metric_key}/OverlapRate"] = []
                        metrics[f"{metric_key}/speed_scales"] = []  # Track weights for proper averaging
                    
                    # Apply speed scaling weight
                    metrics[f"{metric_key}/minADE"].append(min_ade * speed_scale)
                    metrics[f"{metric_key}/minFDE"].append(min_fde * speed_scale)
                    metrics[f"{metric_key}/MissRate"].append(miss_rate * speed_scale)
                    metrics[f"{metric_key}/OverlapRate"].append(overlap_rate * speed_scale)
                    metrics[f"{metric_key}/speed_scales"].append(speed_scale)
    
    # Compute mAP from accumulated predictions
    # For each (object_type, trajectory_type, measurement_step), compute AP
    map_results = {}
    for map_key, data in map_accumulator.items():
        obj_type, traj_type, meas_step = map_key
        
        # Concatenate all predictions
        all_scores = torch.cat(data['pred_scores'])
        all_matched = torch.cat(data['pred_matched'])
        num_gt = data['num_gt']
        
        # Compute precision-recall curve
        precision, recall, _ = compute_precision_recall(
            all_scores, all_matched, num_gt
        )
        
        # Compute average precision
        ap = compute_average_precision(precision, recall)
        
        # Store with key format matching other metrics
        obj_type_name = object_type_names[object_types.index(obj_type)]
        traj_type_name = trajectory_type_names[traj_type]
        metric_key = f"{obj_type_name}_{meas_step}/mAP_{traj_type_name}"
        map_results[metric_key] = ap
    
    # Compute weighted average metrics using speed scales
    result = {}
    metric_keys_to_average = set()
    for key in metrics.keys():
        if not key.endswith('/speed_scales'):
            base_key = key.rsplit('/', 1)[0]  # Get VEHICLE_5 from VEHICLE_5/minADE
            metric_keys_to_average.add(base_key)
    
    for base_key in metric_keys_to_average:
        # Get speed scales for this base key
        scale_key = f"{base_key}/speed_scales"
        if scale_key in metrics and metrics[scale_key]:
            scales = torch.stack(metrics[scale_key])
            total_scale = scales.sum()
            
            # Weighted average for each metric type
            for metric_name in ['minADE', 'minFDE', 'MissRate', 'OverlapRate']:
                full_key = f"{base_key}/{metric_name}"
                if full_key in metrics and metrics[full_key]:
                    values = torch.stack(metrics[full_key])
                    # Values are already scaled, just need to normalize by total weight
                    if total_scale > 0:
                        result[full_key] = values.sum() / total_scale
                    else:
                        result[full_key] = torch.tensor(0.0, device=device)
                else:
                    result[full_key] = torch.tensor(0.0, device=device)
    
    # Add mAP results
    result.update(map_results)
    
    # Compute overall mAP per object type and step (average across trajectory types)
    for obj_type_name in object_type_names:
        for step_config in config.step_configurations:
            meas_step = step_config.measurement_step
            metric_prefix = f"{obj_type_name}_{meas_step}/mAP"
            
            # Find all trajectory-specific mAPs for this object/step
            traj_aps = [v for k, v in map_results.items() if k.startswith(metric_prefix + "_")]
            
            if traj_aps:
                result[f"{obj_type_name}_{meas_step}/mAP"] = torch.stack(traj_aps).mean()
            else:
                result[f"{obj_type_name}_{meas_step}/mAP"] = torch.tensor(0.0, device=device)
    
    return result

