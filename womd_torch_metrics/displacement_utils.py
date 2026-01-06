"""
Displacement computation utilities for WOMD metrics.

This module implements lateral and longitudinal displacement calculations
in a trajectory-aligned coordinate frame, matching the official Waymo
Open Dataset metric implementation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_lateral_longitudinal_displacement(
    pred_point: torch.Tensor,
    gt_point: torch.Tensor,
    gt_heading: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute lateral and longitudinal displacement between predicted and ground truth points.
    
    Lateral displacement is perpendicular to the GT heading direction.
    Longitudinal displacement is along the GT heading direction.
    
    Args:
        pred_point: [..., 2] predicted (x, y) position
        gt_point: [..., 2] ground truth (x, y) position
        gt_heading: [...] ground truth heading angle in radians
        
    Returns:
        Tuple of (lateral_displacement, longitudinal_displacement)
        Both have shape [...]
    """
    # Displacement vector from GT to prediction
    delta = pred_point - gt_point  # [..., 2]
    dx, dy = delta[..., 0], delta[..., 1]
    
    # Heading direction (along longitudinal axis)
    cos_heading = torch.cos(gt_heading)
    sin_heading = torch.sin(gt_heading)
    
    # Project displacement onto heading direction (longitudinal)
    longitudinal = dx * cos_heading + dy * sin_heading
    
    # Project displacement onto perpendicular direction (lateral)
    # Perpendicular to heading is (-sin, cos) rotated 90 degrees
    lateral = -dx * sin_heading + dy * cos_heading
    
    return lateral, longitudinal


def compute_displacement_at_step(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_heading: torch.Tensor,
    step: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute lateral and longitudinal displacement at a specific timestep.
    
    Args:
        pred_trajectory: [..., T, 2] predicted trajectory
        gt_trajectory: [..., T, 2] ground truth trajectory  
        gt_heading: [..., T] ground truth heading angles in radians
        step: Timestep index to evaluate (0-indexed)
        valid_mask: [..., T] optional mask for valid timesteps
        
    Returns:
        Tuple of (lateral, longitudinal) displacements at the step,
        or None if the step is invalid
    """
    # Check validity
    if valid_mask is not None:
        if not valid_mask[..., step].all():
            return None
    
    # Extract points at the specified step
    pred_point = pred_trajectory[..., step, :]
    gt_point = gt_trajectory[..., step, :]
    heading = gt_heading[..., step]
    
    return compute_lateral_longitudinal_displacement(pred_point, gt_point, heading)


def compute_average_displacement_lateral_longitudinal(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_heading: torch.Tensor,
    max_step: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute average lateral and longitudinal displacements up to a maximum step.
    
    Args:
        pred_trajectory: [..., T, 2] predicted trajectory
        gt_trajectory: [..., T, 2] ground truth trajectory
        gt_heading: [..., T] ground truth heading angles in radians
        max_step: Maximum timestep to include (exclusive, so max_step means steps 0 to max_step-1)
        valid_mask: [..., T] optional mask for valid timesteps
        
    Returns:
        Tuple of (avg_lateral, avg_longitudinal) average displacements
    """
    # Compute displacements at each timestep
    lateral_sum = torch.zeros_like(gt_trajectory[..., 0, 0])
    longitudinal_sum = torch.zeros_like(gt_trajectory[..., 0, 0])
    count = torch.zeros_like(gt_trajectory[..., 0, 0])
    
    for t in range(min(max_step, gt_trajectory.shape[-2])):
        # Check validity
        if valid_mask is not None:
            step_valid = valid_mask[..., t]
        else:
            step_valid = torch.ones_like(gt_trajectory[..., t, 0], dtype=torch.bool)
        
        if step_valid.any():
            lateral, longitudinal = compute_lateral_longitudinal_displacement(
                pred_trajectory[..., t, :],
                gt_trajectory[..., t, :],
                gt_heading[..., t]
            )
            
            # Accumulate only for valid timesteps
            lateral_sum = lateral_sum + torch.where(step_valid, torch.abs(lateral), torch.zeros_like(lateral))
            longitudinal_sum = longitudinal_sum + torch.where(step_valid, torch.abs(longitudinal), torch.zeros_like(longitudinal))
            count = count + step_valid.float()
    
    # Compute averages
    avg_lateral = torch.where(count > 0, lateral_sum / count, torch.zeros_like(lateral_sum))
    avg_longitudinal = torch.where(count > 0, longitudinal_sum / count, torch.zeros_like(longitudinal_sum))
    
    return avg_lateral, avg_longitudinal


def extract_heading_from_trajectory(
    trajectory: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract or estimate heading angles from trajectory.
    
    If trajectory has 7 dimensions [x, y, length, width, heading, vx, vy],
    extracts the heading directly. Otherwise, estimates heading from
    velocity direction (finite differences of positions).
    
    Args:
        trajectory: [..., T, D] trajectory where D is 2, 5, or 7
            If D=2: [x, y]
            If D=7: [x, y, length, width, heading, velocity_x, velocity_y]
        valid_mask: [..., T] optional mask for valid timesteps
        
    Returns:
        [..., T] heading angles in radians
    """
    D = trajectory.shape[-1]
    
    if D >= 5:
        # Heading is at index 4: [x, y, length, width, heading, ...]
        heading = trajectory[..., 4]
    else:
        # Estimate heading from position changes
        positions = trajectory[..., :2]  # [..., T, 2]
        
        # Compute velocity by finite differences
        velocity = torch.zeros_like(positions)
        velocity[..., 1:, :] = positions[..., 1:, :] - positions[..., :-1, :]
        
        # For first timestep, use the same as second
        velocity[..., 0, :] = velocity[..., 1, :]
        
        # Compute heading from velocity
        heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        
        # Handle invalid timesteps
        if valid_mask is not None:
            # For invalid timesteps, use previous valid heading
            for t in range(1, heading.shape[-1]):
                invalid = ~valid_mask[..., t]
                heading[..., t] = torch.where(invalid, heading[..., t-1], heading[..., t])
    
    return heading


def check_within_thresholds(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    lateral_threshold: float,
    longitudinal_threshold: float,
    measurement_step: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Check if prediction is within lateral and longitudinal thresholds at measurement step.
    
    This implements the official WOMD matching criterion: a prediction matches
    if BOTH lateral <= lateral_threshold AND longitudinal <= longitudinal_threshold.
    
    Args:
        pred_trajectory: [..., T, 2] predicted trajectory
        gt_trajectory: [..., T, 2] or [..., T, 7] ground truth trajectory
        lateral_threshold: Maximum lateral displacement in meters
        longitudinal_threshold: Maximum longitudinal displacement in meters
        measurement_step: Timestep to evaluate (0-indexed)
        valid_mask: [..., T] optional mask for valid timesteps
        
    Returns:
        [...] boolean tensor indicating if within thresholds
    """
    # Extract heading from GT
    gt_heading = extract_heading_from_trajectory(gt_trajectory, valid_mask)
    
    # Compute displacement at measurement step
    result = compute_displacement_at_step(
        pred_trajectory,
        gt_trajectory[..., :2] if gt_trajectory.shape[-1] > 2 else gt_trajectory,
        gt_heading,
        measurement_step,
        valid_mask
    )
    
    if result is None:
        # Invalid step - return False
        return torch.zeros_like(pred_trajectory[..., 0, 0], dtype=torch.bool)
    
    lateral, longitudinal = result
    
    # Check both thresholds
    within_lateral = torch.abs(lateral) <= lateral_threshold
    within_longitudinal = torch.abs(longitudinal) <= longitudinal_threshold
    
    return within_lateral & within_longitudinal
