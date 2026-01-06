"""
Trajectory classification for WOMD metrics.

This module implements trajectory type classification into 8 categories
based on displacement and heading change, matching the official Waymo
Open Dataset implementation.
"""

import torch
from typing import Optional
import math


# Trajectory type constants (matching official WOMD)
TRAJECTORY_TYPE_STATIONARY = 0
TRAJECTORY_TYPE_STRAIGHT = 1
TRAJECTORY_TYPE_STRAIGHT_LEFT = 2
TRAJECTORY_TYPE_STRAIGHT_RIGHT = 3
TRAJECTORY_TYPE_LEFT_U_TURN = 4
TRAJECTORY_TYPE_LEFT_TURN = 5
TRAJECTORY_TYPE_RIGHT_U_TURN = 6
TRAJECTORY_TYPE_RIGHT_TURN = 7

# Classification thresholds
STATIONARY_THRESHOLD = 2.0  # meters - displacement below this is stationary
HEADING_CHANGE_THRESHOLD_SMALL = math.radians(15)  # 15 degrees
HEADING_CHANGE_THRESHOLD_MEDIUM = math.radians(45)  # 45 degrees
HEADING_CHANGE_THRESHOLD_LARGE = math.radians(135)  # 135 degrees


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_trajectory_displacement(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute total displacement of a trajectory (start to end Euclidean distance).
    
    Args:
        trajectory: [T, 2] or [T, 7] trajectory with (x, y) positions
        valid_mask: [T] optional mask for valid timesteps
        
    Returns:
        Scalar displacement in meters
    """
    positions = trajectory[..., :2]
    
    if valid_mask is not None:
        # Find first and last valid indices
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=trajectory.device)
        start_idx = valid_indices[0]
        end_idx = valid_indices[-1]
    else:
        start_idx = 0
        end_idx = len(positions) - 1
    
    start_pos = positions[start_idx]
    end_pos = positions[end_idx]
    
    displacement = torch.norm(end_pos - start_pos)
    return displacement


def compute_trajectory_heading_change(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute total heading change of a trajectory.
    
    For trajectories with 7D format, uses the heading field directly.
    For 2D trajectories, estimates heading from velocity direction.
    
    Args:
        trajectory: [T, 2] or [T, 7] trajectory
            If 7D: [x, y, length, width, heading, velocity_x, velocity_y]
        valid_mask: [T] optional mask for valid timesteps
        
    Returns:
        Total heading change in radians (absolute value)
    """
    D = trajectory.shape[-1]
    
    if valid_mask is not None:
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=trajectory.device)
    else:
        valid_indices = torch.arange(len(trajectory), device=trajectory.device)
    
    if D >= 5:
        # Extract heading directly
        headings = trajectory[valid_indices, 4]
    else:
        # Estimate heading from position changes
        positions = trajectory[valid_indices, :2]
        
        if len(positions) < 2:
            return torch.tensor(0.0, device=trajectory.device)
        
        # Compute velocities
        velocities = positions[1:] - positions[:-1]
        
        # Compute headings from velocities
        headings = torch.atan2(velocities[:, 1], velocities[:, 0])
    
    if len(headings) < 2:
        return torch.tensor(0.0, device=trajectory.device)
    
    # Compute total heading change
    start_heading = headings[0]
    end_heading = headings[-1]
    
    heading_change = normalize_angle(end_heading - start_heading)
    
    return torch.abs(heading_change)


def classify_trajectory_type(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Classify trajectory into one of 8 types based on displacement and heading change.
    
    Classification logic (matching official WOMD):
    1. If displacement < 2.0m: STATIONARY
    2. Otherwise, based on heading change:
       - < 15°: STRAIGHT
       - 15° to 45°: STRAIGHT_LEFT or STRAIGHT_RIGHT
       - 45° to 135°: LEFT_TURN or RIGHT_TURN  
       - > 135°: LEFT_U_TURN or RIGHT_U_TURN
       
    Direction (left vs right) is determined by sign of heading change.
    
    Args:
        trajectory: [T, 2] or [T, 7] trajectory
        valid_mask: [T] optional mask for valid timesteps
        
    Returns:
        Trajectory type (integer 0-7)
    """
    device = trajectory.device
    
    # Compute displacement
    displacement = compute_trajectory_displacement(trajectory, valid_mask)
    
    # Check if stationary
    if displacement < STATIONARY_THRESHOLD:
        return TRAJECTORY_TYPE_STATIONARY
    
    # Compute heading change
    D = trajectory.shape[-1]
    
    if valid_mask is not None:
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return TRAJECTORY_TYPE_STATIONARY
    else:
        valid_indices = torch.arange(len(trajectory), device=device)
    
    if D >= 5:
        # Extract heading directly
        headings = trajectory[valid_indices, 4]
    else:
        # Estimate heading from position changes
        positions = trajectory[valid_indices, :2]
        
        if len(positions) < 2:
            return TRAJECTORY_TYPE_STATIONARY
        
        # Compute velocities
        velocities = positions[1:] - positions[:-1]
        
        # Compute headings from velocities
        headings = torch.atan2(velocities[:, 1], velocities[:, 0])
    
    if len(headings) < 2:
        return TRAJECTORY_TYPE_STATIONARY
    
    # Compute signed heading change (positive = left, negative = right)
    start_heading = headings[0]
    end_heading = headings[-1]
    heading_change = normalize_angle(end_heading - start_heading)
    
    abs_heading_change = torch.abs(heading_change)
    
    # Classify based on heading change magnitude
    if abs_heading_change < HEADING_CHANGE_THRESHOLD_SMALL:
        return TRAJECTORY_TYPE_STRAIGHT
    elif abs_heading_change < HEADING_CHANGE_THRESHOLD_MEDIUM:
        # STRAIGHT_LEFT or STRAIGHT_RIGHT
        if heading_change > 0:
            return TRAJECTORY_TYPE_STRAIGHT_LEFT
        else:
            return TRAJECTORY_TYPE_STRAIGHT_RIGHT
    elif abs_heading_change < HEADING_CHANGE_THRESHOLD_LARGE:
        # LEFT_TURN or RIGHT_TURN
        if heading_change > 0:
            return TRAJECTORY_TYPE_LEFT_TURN
        else:
            return TRAJECTORY_TYPE_RIGHT_TURN
    else:
        # LEFT_U_TURN or RIGHT_U_TURN
        if heading_change > 0:
            return TRAJECTORY_TYPE_LEFT_U_TURN
        else:
            return TRAJECTORY_TYPE_RIGHT_U_TURN


def get_trajectory_type_name(trajectory_type: int) -> str:
    """Get human-readable name for trajectory type."""
    names = [
        "STATIONARY",
        "STRAIGHT", 
        "STRAIGHT_LEFT",
        "STRAIGHT_RIGHT",
        "LEFT_U_TURN",
        "LEFT_TURN",
        "RIGHT_U_TURN",
        "RIGHT_TURN"
    ]
    return names[trajectory_type] if 0 <= trajectory_type < len(names) else "UNKNOWN"
