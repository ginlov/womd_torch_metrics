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

# Import trajectory-aligned displacement utilities
from .displacement_utils import (
    compute_lateral_longitudinal_displacement,
    compute_displacement_at_step,
    extract_heading_from_trajectory,
    check_within_thresholds,
)

# Import trajectory classification
from .trajectory_classification import (
    classify_trajectory_type,
    get_trajectory_type_name,
    TRAJECTORY_TYPE_STATIONARY,
    TRAJECTORY_TYPE_STRAIGHT,
    TRAJECTORY_TYPE_STRAIGHT_LEFT,
    TRAJECTORY_TYPE_STRAIGHT_RIGHT,
    TRAJECTORY_TYPE_LEFT_U_TURN,
    TRAJECTORY_TYPE_LEFT_TURN,
    TRAJECTORY_TYPE_RIGHT_U_TURN,
    TRAJECTORY_TYPE_RIGHT_TURN,
)


# Object type constants (matching Waymo Open Dataset)
TYPE_UNSET = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3
TYPE_OTHER = 4


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
    prediction_steps_per_second: int = 10  # Prediction frequency (Hz)
    
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
            # At 10Hz prediction: step 30 = 3s, step 50 = 5s, step 80 = 8s
            self.step_configurations = [
                StepConfig(measurement_step=30, lateral_miss_threshold=1.0, longitudinal_miss_threshold=2.0),
                StepConfig(measurement_step=50, lateral_miss_threshold=1.8, longitudinal_miss_threshold=3.6),
                StepConfig(measurement_step=80, lateral_miss_threshold=3.0, longitudinal_miss_threshold=6.0),
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


def compute_precision_recall(
    pred_scores: torch.Tensor,
    pred_matched: torch.Tensor,
    num_gt: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision-recall curve for mAP calculation.
    
    Implements greedy matching: each GT can only be matched once by the
    highest-scoring prediction that matches it. This prevents recall > 1.0.
    
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
    
    # Greedy matching: each GT can only be matched once
    # After num_gt TPs, all remaining "matched" predictions become FPs
    tp_count = 0
    greedy_matched = torch.zeros_like(sorted_matched, dtype=torch.bool)
    
    for i in range(len(sorted_matched)):
        if sorted_matched[i] and tp_count < num_gt:
            greedy_matched[i] = True
            tp_count += 1
        # else: remains False (either didn't match or GT already taken)
    
    # Compute cumulative TP and FP
    tp_cumsum = torch.cumsum(greedy_matched.float(), dim=0)
    fp_cumsum = torch.cumsum((~greedy_matched).float(), dim=0)
    
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
    # Area = sum of (recall_i+1 - recall_i) * precision_i
    recall_diff = recall[1:] - recall[:-1]
    ap = torch.sum(recall_diff * max_precision[:-1])
    
    return ap


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
    if trajectory.shape[-1] == 7:
        velo = trajectory[-1, -2:] # velo_x, velo_y
        speed = torch.norm(velo)
        return speed
    else:
        positions = trajectory[..., :2]
    
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
    if speed < speed_lower_bound:
        return torch.tensor(speed_scale_lower, device=speed.device)
    
    if speed > speed_upper_bound:
        return torch.tensor(speed_scale_upper, device=speed.device)
    
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
        gt_trajectories = ground_truth_trajectory[b]  # [A, T_gt, 2] or [A, T_gt, 7]
        gt_valid = ground_truth_is_valid[b]  # [A, T_gt]
        obj_types = object_type[b]  # [A]
        
        # Process each joint prediction group
        for m in range(M):
            pred_traj = prediction_trajectory[b, m]  # [K, N, T_pred, 2]
            pred_scores = prediction_score[b, m]  # [K]
            gt_indices = prediction_ground_truth_indices[b, m]  # [N]
            gt_mask = prediction_ground_truth_indices_mask[b, m]  # [N]
            
            # Collect all agents in this joint group
            joint_group_agents = []
            for n in range(N):
                if not gt_mask[n]:
                    continue
                
                gt_idx = gt_indices[n].item()
                if gt_idx >= A:
                    raise ValueError(f"GT index {gt_idx} out of bounds for scenario {b}, group {m}, agent {n}")
                
                # Get ground truth for this agent
                gt_traj = gt_trajectories[gt_idx]  # [T_gt, 2] or [T_gt, 7]
                gt_valid_n = gt_valid[gt_idx]  # [T_gt]
                agent_type = obj_types[gt_idx].item()
                
                # Skip if object type not in evaluation set
                if agent_type not in object_types:
                    continue
                
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
                
                # Get ground truth boxes if provided
                gt_boxes_n = None
                if ground_truth_boxes is not None:
                    gt_boxes_n = ground_truth_boxes[b, gt_idx]  # [T_gt, 4]
                
                joint_group_agents.append({
                    'n': n,
                    'gt_idx': gt_idx,
                    'gt_traj': gt_traj,
                    'gt_valid': gt_valid_n,
                    'agent_type': agent_type,
                    'speed_scale': speed_scale,
                    'gt_boxes': gt_boxes_n,
                })
            
            # Skip if no valid agents in this joint group
            if len(joint_group_agents) == 0:
                continue
            
            # For joint predictions: use the least common (rarest) object type for bucketing
            # Frequency: vehicle > pedestrian > cyclist
            type_frequency = {TYPE_VEHICLE: 3, TYPE_PEDESTRIAN: 2, TYPE_CYCLIST: 1, TYPE_OTHER: 0}
            least_common_type = min(joint_group_agents, key=lambda a: type_frequency.get(a['agent_type'], 0))['agent_type']
            obj_type_name = object_type_names[object_types.index(least_common_type)]
            
            # Compute metrics for each measurement step
            for step_config in config.step_configurations:
                # Convert prediction step to track step
                step_ratio = config.track_steps_per_second / config.prediction_steps_per_second
                track_step = int(step_config.measurement_step * step_ratio)
                
                # Ensure we don't go out of bounds
                track_step = min(track_step, T_pred - 1, T_gt - 1)
                
                metric_key = f"{obj_type_name}_{step_config.measurement_step}"
                
                # Compute minADE and minFDE for joint prediction
                # Formula: (1/M) * min_i sum_{j=1}^M ADE_j^i
                # where M = number of agents, i = joint prediction index, j = agent index
                
                ade_per_joint_pred = []  # [K] - sum of ADEs for each joint prediction
                fde_per_joint_pred = []  # [K] - sum of FDEs for each joint prediction
                
                # For each of K joint predictions
                for k in range(K):
                    ade_sum = 0.0
                    fde_sum = 0.0
                    
                    # Sum ADE/FDE across all agents in this joint prediction
                    for agent_info in joint_group_agents:
                        n = agent_info['n']
                        gt_traj = agent_info['gt_traj']
                        gt_valid_n = agent_info['gt_valid']
                        
                        # Get prediction for this agent in this joint prediction
                        pred_traj_kn = pred_traj[k:k+1, n, :, :]  # [1, T_pred, 2]
                        
                        # Compute ADE for this agent in this joint prediction
                        errors = compute_displacement_error(
                            pred_traj_kn[:, :track_step+1, :],  # [1, T, 2]
                            gt_traj[:track_step+1, :2].unsqueeze(0),  # [1, T, 2]
                            gt_valid_n[:track_step+1].unsqueeze(0) if gt_valid_n is not None else None,
                        )
                        
                        if gt_valid_n is not None:
                            ade = errors.sum() / gt_valid_n[:track_step+1].sum().clamp(min=1.0)
                        else:
                            ade = errors.mean()
                        
                        ade_sum += ade.item()
                        
                        # Compute FDE for this agent in this joint prediction
                        gt_xy = gt_traj[:, :2] if gt_traj.shape[-1] >= 7 else gt_traj
                        pred_final = pred_traj_kn[0, track_step, :]  # [2]
                        gt_final = gt_xy[track_step, :]  # [2]
                        fde = torch.norm(pred_final - gt_final)
                        
                        fde_sum += fde.item()
                    
                    ade_per_joint_pred.append(ade_sum)
                    fde_per_joint_pred.append(fde_sum)
                
                # Take minimum across K joint predictions and divide by M (number of agents)
                M_agents = len(joint_group_agents)
                min_ade = torch.tensor(min(ade_per_joint_pred) / M_agents, device=device)
                min_fde = torch.tensor(min(fde_per_joint_pred) / M_agents, device=device)
                
                # Compute Miss Rate for joint group
                # A miss is when NONE of the K joint predictions have ALL M agents within thresholds
                is_miss = True
                for k in range(K):
                    # Check if all agents in this joint prediction k are within thresholds
                    all_agents_within = True
                    for agent_info in joint_group_agents:
                        n = agent_info['n']
                        gt_traj = agent_info['gt_traj']
                        gt_valid_n = agent_info['gt_valid']
                        speed_scale = agent_info['speed_scale']
                        
                        # Get prediction for this agent in joint prediction k
                        pred_traj_kn = pred_traj[k:k+1, n, :, :]  # [1, T_pred, 2]
                        
                        # Check if this agent is within thresholds
                        scaled_lateral_threshold = step_config.lateral_miss_threshold * speed_scale
                        scaled_longitudinal_threshold = step_config.longitudinal_miss_threshold * speed_scale
                        
                        within = check_within_thresholds(
                            pred_traj_kn,
                            gt_traj,
                            scaled_lateral_threshold,
                            scaled_longitudinal_threshold,
                            track_step,
                            gt_valid_n
                        )
                        
                        if not within.item():
                            all_agents_within = False
                            break
                    
                    # If all agents in this joint prediction are within thresholds, not a miss
                    if all_agents_within:
                        is_miss = False
                        break
                
                miss_rate = torch.tensor(1.0 if is_miss else 0.0, device=device)
                
                # Compute Overlap Rate for joint group
                # For now, simplified: compute per agent and check if any overlap
                # TODO: Implement full overlap checking with other objects
                overlap_rate = torch.tensor(0.0, device=device)
                
                # Compute mAP for joint group
                # Classification uses first agent's trajectory type (arbitrary selection)
                first_agent = joint_group_agents[0]
                gt_type = classify_trajectory_type(first_agent['gt_traj'], first_agent['gt_valid'])
                
                # Check if each joint prediction k is a miss (same logic as miss rate)
                pred_matched = torch.zeros(K, dtype=torch.bool, device=device)
                for k in range(K):
                    all_agents_within = True
                    for agent_info in joint_group_agents:
                        n = agent_info['n']
                        gt_traj = agent_info['gt_traj']
                        gt_valid_n = agent_info['gt_valid']
                        speed_scale = agent_info['speed_scale']
                        
                        pred_traj_kn = pred_traj[k:k+1, n, :, :]
                        
                        scaled_lateral_threshold = step_config.lateral_miss_threshold * speed_scale
                        scaled_longitudinal_threshold = step_config.longitudinal_miss_threshold * speed_scale
                        
                        within = check_within_thresholds(
                            pred_traj_kn,
                            gt_traj,
                            scaled_lateral_threshold,
                            scaled_longitudinal_threshold,
                            track_step,
                            gt_valid_n
                        )
                        
                        if not within.item():
                            all_agents_within = False
                            break
                    
                    pred_matched[k] = all_agents_within
                
                # At most 1 prediction can match, in case of multiple matches, keep highest score
                if pred_matched.sum() > 1:
                    matched_indices = torch.nonzero(pred_matched).squeeze(1)
                    matched_scores = pred_scores[matched_indices]
                    best_idx = matched_indices[torch.argmax(matched_scores)]
                    pred_matched[:] = False
                    pred_matched[best_idx] = True
                
                # Accumulate mAP data per trajectory type
                map_key = (least_common_type, gt_type, step_config.measurement_step)
                if map_key not in map_accumulator:
                    map_accumulator[map_key] = {
                        'pred_scores': [],
                        'pred_matched': [],
                        'num_gt': 0
                    }
                
                map_accumulator[map_key]['pred_scores'].append(pred_scores)
                map_accumulator[map_key]['pred_matched'].append(pred_matched)
                map_accumulator[map_key]['num_gt'] += 1
                
                # Store metrics once per joint group
                if metric_key not in metrics:
                    metrics[f"{metric_key}/minADE"] = []
                    metrics[f"{metric_key}/minFDE"] = []
                    metrics[f"{metric_key}/MissRate"] = []
                    metrics[f"{metric_key}/OverlapRate"] = []
                
                # Store all metrics for the joint group
                metrics[f"{metric_key}/minADE"].append(min_ade)
                metrics[f"{metric_key}/minFDE"].append(min_fde)
                metrics[f"{metric_key}/MissRate"].append(miss_rate)
                metrics[f"{metric_key}/OverlapRate"].append(overlap_rate)
    
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
    
    # Compute simple average of metrics
    result = {}
    metric_keys_to_average = set()
    for key in metrics.keys():
        base_key = key.rsplit('/', 1)[0]  # Get VEHICLE_5 from VEHICLE_5/minADE
        metric_keys_to_average.add(base_key)
    
    for base_key in metric_keys_to_average:
        # Simple average for each metric type
        for metric_name in ['minADE', 'minFDE', 'MissRate', 'OverlapRate']:
            full_key = f"{base_key}/{metric_name}"
            if full_key in metrics and metrics[full_key]:
                values = torch.stack(metrics[full_key])
                result[full_key] = values.mean()
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

