"""
Example usage of map-based off-road metrics for Waymo Open Motion Dataset.
"""

import torch
from womd_torch_metrics.map_metrics import (
    compute_map_metrics,
    compute_distance_to_road_edge,
    compute_offroad_rate,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example 1: Basic off-road detection
    print("\n=== Example 1: Basic Off-Road Detection ===")
    
    # Define a road: 10m wide strip from y=-5 to y=5
    road_edges = [
        torch.tensor([
            [0.0, -5.0, 0.0],
            [100.0, -5.0, 0.0],
        ], device=device),
        torch.tensor([
            [0.0, 5.0, 0.0],
            [100.0, 5.0, 0.0],
        ], device=device),
    ]
    
    # Agent driving straight on road
    B, A, T = 1, 1, 10
    center_x = torch.linspace(0, 100, T, device=device).reshape(B, A, T)
    center_y = torch.zeros(B, A, T, device=device)  # Stays at y=0 (center)
    center_z = torch.zeros(B, A, T, device=device)
    length = torch.ones(B, A, T, device=device) * 4.5
    width = torch.ones(B, A, T, device=device) * 2.0
    heading = torch.zeros(B, A, T, device=device)
    valid_mask = torch.ones(B, A, T, dtype=torch.bool, device=device)
    
    # Compute distance to road edge
    distances = compute_distance_to_road_edge(
        center_x, center_y, center_z, length, width, heading, road_edges, valid_mask
    )
    
    print(f"Distance to road edge (on-road): {distances[0, 0, 0].item():.2f}m (negative = on-road)")
    
    # Now move agent off-road
    center_y_offroad = torch.ones(B, A, T, device=device) * 10.0  # Move to y=10
    distances_offroad = compute_distance_to_road_edge(
        center_x, center_y_offroad, center_z, length, width, heading, road_edges, valid_mask
    )
    
    print(f"Distance to road edge (off-road): {distances_offroad[0, 0, 0].item():.2f}m (positive = off-road)")
    
    # Example 2: Off-road rate computation
    print("\n=== Example 2: Off-Road Rate Computation ===")
    
    # Agent that gradually moves off road
    center_y_mixed = torch.cat([
        torch.zeros(5, device=device),      # On-road for first 5 steps
        torch.ones(5, device=device) * 8.0  # Off-road for last 5 steps
    ]).reshape(B, A, T)
    
    offroad_rate = compute_offroad_rate(
        center_x, center_y_mixed, center_z, length, width, heading,
        road_edges, valid_mask
    )
    
    print(f"Off-road rate: {offroad_rate[0, 0].item()*100:.1f}% (5 out of 10 timesteps)")
    
    # Example 3: Full metrics with ground truth trajectories
    print("\n=== Example 3: Complete Map Metrics ===")
    
    # Create ground truth data
    gt_traj = torch.zeros(B, A, T, 7, device=device)
    gt_traj[..., 0] = center_x  # x
    gt_traj[..., 1] = center_y_mixed  # y (mixed on/off road)
    gt_traj[..., 2] = center_z  # z
    gt_traj[..., 3] = 4.5  # length
    gt_traj[..., 4] = 2.0  # width
    gt_traj[..., 5] = 1.5  # height
    gt_traj[..., 6] = 0.0  # heading
    
    gt_valid = torch.ones(B, A, T, dtype=torch.bool, device=device)
    object_type = torch.ones(B, A, dtype=torch.long, device=device)  # 1 = VEHICLE
    
    # Convert road edges to list format
    road_edges_list = [[
        [(0.0, -5.0, 0.0), (100.0, -5.0, 0.0)],
        [(0.0, 5.0, 0.0), (100.0, 5.0, 0.0)],
    ]]
    
    metrics = compute_map_metrics(
        gt_traj, gt_valid, object_type, road_edges_list
    )
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value.item():.4f}")
    
    # Example 4: Multiple agents with different behaviors
    print("\n=== Example 4: Multiple Agents ===")
    
    B, A, T = 1, 3, 10
    
    # Agent 0: Always on road
    # Agent 1: Always off road
    # Agent 2: Sometimes off road
    gt_traj_multi = torch.zeros(B, A, T, 7, device=device)
    
    # X coordinates (all agents move forward)
    gt_traj_multi[:, :, :, 0] = torch.linspace(0, 100, T).repeat(B, A, 1)
    
    # Y coordinates (different behaviors)
    gt_traj_multi[0, 0, :, 1] = 0.0  # Agent 0: center of road
    gt_traj_multi[0, 1, :, 1] = 10.0  # Agent 1: way off road
    gt_traj_multi[0, 2, :5, 1] = 0.0  # Agent 2: on-road first half
    gt_traj_multi[0, 2, 5:, 1] = 8.0  # Agent 2: off-road second half
    
    # Dimensions
    gt_traj_multi[..., 3] = 4.5  # length
    gt_traj_multi[..., 4] = 2.0  # width
    gt_traj_multi[..., 5] = 1.5  # height
    
    gt_valid_multi = torch.ones(B, A, T, dtype=torch.bool, device=device)
    object_type_multi = torch.ones(B, A, dtype=torch.long, device=device)
    
    metrics_multi = compute_map_metrics(
        gt_traj_multi, gt_valid_multi, object_type_multi, road_edges_list
    )
    
    print(f"Average off-road rate across 3 agents: {metrics_multi['VEHICLE/OffRoadRate'].item()*100:.1f}%")
    print("  (Expected: ~50% = (0% + 100% + 50%) / 3)")


if __name__ == "__main__":
    main()
