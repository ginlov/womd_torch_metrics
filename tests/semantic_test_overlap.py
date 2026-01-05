"""
Semantic tests for overlap computation using test samples from .pth files.

This test will automatically discover and test all .pth files in test_samples/overlap/
"""

import torch
import pytest
from pathlib import Path
from womd_torch_metrics import compute_overlap_rate


@pytest.fixture
def test_samples_dir():
    """Get the test samples directory."""
    return Path(__file__).parent.parent / "test_samples" / "overlap"


def test_overlap_from_files(test_samples_dir):
    """
    Test overlap computation by loading test cases from .pth files.
    
    Each .pth file should contain a dictionary with:
    - 'prediction_trajectories': [K, T, 2] predicted trajectories
    - 'ground_truth_trajectory': [T, 2] ground truth trajectory
    - 'ground_truth_boxes': [T, 4] ground truth boxes [length, width, heading, velocity_heading]
    - 'valid_mask': [T] optional validity mask
    - 'measurement_step': optional measurement step
    - 'threshold': overlap threshold
    - 'expected_overlap_rate': expected overlap rate value
    """
    if not test_samples_dir.exists():
        pytest.skip(f"Test samples directory not found: {test_samples_dir}")
    
    # Find all .pth files in the directory (excluding batch files)
    all_pth_files = list(test_samples_dir.glob("*.pth"))
    pth_files = [f for f in all_pth_files if not f.name.startswith("batch_")]
    
    if not pth_files:
        pytest.skip(f"No .pth files found in {test_samples_dir}")
    
    print(f"\nFound {len(pth_files)} test sample(s)")
    
    for pth_file in sorted(pth_files):
        print(f"\nTesting: {pth_file.name}")
        
        # Load test data
        test_data = torch.load(pth_file, map_location="cpu")
        
        # Extract inputs
        pred_trajectories = test_data["prediction_trajectories"]
        gt_trajectory = test_data["ground_truth_trajectory"]
        gt_boxes = test_data["ground_truth_boxes"]
        
        # Optional inputs
        valid_mask = test_data.get("valid_mask", None)
        measurement_step = test_data.get("measurement_step", None)
        threshold = test_data.get("threshold", 0.5)
        expected_overlap_rate = test_data.get("expected_overlap_rate", None)
        
        # Compute overlap rate
        computed_overlap_rate = compute_overlap_rate(
            pred_trajectories,
            gt_trajectory,
            gt_boxes,
            threshold=threshold,
            valid_mask=valid_mask,
            measurement_step=measurement_step,
        )
        
        print(f"  Computed overlap rate: {computed_overlap_rate.item():.6f}")
        
        if expected_overlap_rate is not None:
            print(f"  Expected overlap rate: {expected_overlap_rate:.6f}")
            print(f"  Difference: {abs(computed_overlap_rate.item() - expected_overlap_rate):.6f}")
            
            # Check if values match (with tolerance for floating point)
            tolerance = test_data.get("tolerance", 1e-5)
            assert abs(computed_overlap_rate.item() - expected_overlap_rate) < tolerance, (
                f"Overlap rate mismatch for {pth_file.name}:\n"
                f"  Expected: {expected_overlap_rate}\n"
                f"  Computed: {computed_overlap_rate.item()}\n"
                f"  Difference: {abs(computed_overlap_rate.item() - expected_overlap_rate)}"
            )
            print(f"  ✓ Passed (tolerance: {tolerance})")
        else:
            print(f"  ⚠ No expected value provided, only checking computation succeeds")
            assert 0.0 <= computed_overlap_rate.item() <= 1.0, \
                f"Overlap rate should be in [0, 1], got {computed_overlap_rate.item()}"


def test_overlap_batch_from_files(test_samples_dir):
    """
    Test batch overlap computation from files.
    
    Files should contain:
    - 'prediction_trajectories': [B, K, T, 2] batch of predictions
    - 'ground_truth_trajectory': [B, T, 2] batch of ground truth
    - 'ground_truth_boxes': [B, T, 4] batch of boxes
    - 'expected_overlap_rates': [B] expected overlap rates per batch
    """
    if not test_samples_dir.exists():
        pytest.skip(f"Test samples directory not found: {test_samples_dir}")
    
    # Look for batch test files (e.g., batch_*.pth)
    batch_files = list(test_samples_dir.glob("batch_*.pth"))
    
    if not batch_files:
        pytest.skip(f"No batch test files found in {test_samples_dir}")
    
    print(f"\nFound {len(batch_files)} batch test sample(s)")
    
    for pth_file in sorted(batch_files):
        print(f"\nTesting batch: {pth_file.name}")
        
        test_data = torch.load(pth_file, map_location="cpu")
        
        pred_trajectories = test_data["prediction_trajectories"]  # [B, K, T, 2]
        gt_trajectory = test_data["ground_truth_trajectory"]  # [B, T, 2]
        gt_boxes = test_data["ground_truth_boxes"]  # [B, T, 4]
        
        valid_mask = test_data.get("valid_mask", None)
        threshold = test_data.get("threshold", 0.5)
        expected_overlap_rates = test_data.get("expected_overlap_rates", None)
        tolerance = test_data.get("tolerance", 1e-5)
        
        B = pred_trajectories.shape[0]
        computed_rates = []
        
        # Process each batch item
        for b in range(B):
            overlap_rate = compute_overlap_rate(
                pred_trajectories[b],
                gt_trajectory[b],
                gt_boxes[b],
                threshold=threshold,
                valid_mask=valid_mask[b] if valid_mask is not None else None,
            )
            computed_rates.append(overlap_rate.item())
        
        computed_rates = torch.tensor(computed_rates)
        print(f"  Computed overlap rates: {computed_rates.tolist()}")
        
        if expected_overlap_rates is not None:
            expected_rates = torch.tensor(expected_overlap_rates)
            print(f"  Expected overlap rates: {expected_rates.tolist()}")
            
            differences = torch.abs(computed_rates - expected_rates)
            print(f"  Differences: {differences.tolist()}")
            
            assert torch.all(differences < tolerance), (
                f"Overlap rate mismatch for {pth_file.name}:\n"
                f"  Expected: {expected_rates.tolist()}\n"
                f"  Computed: {computed_rates.tolist()}\n"
                f"  Max difference: {differences.max().item()}"
            )
            print(f"  ✓ Passed (tolerance: {tolerance})")
        else:
            # Just check values are in valid range
            assert torch.all((computed_rates >= 0.0) & (computed_rates <= 1.0)), \
                f"Overlap rates should be in [0, 1], got {computed_rates.tolist()}"


if __name__ == "__main__":
    # Run tests
    test_samples_dir = Path(__file__).parent.parent / "test_samples" / "overlap"
    
    if test_samples_dir.exists():
        print(f"Running tests from: {test_samples_dir}")
        pytest.main([__file__, "-v", "-s"])
    else:
        print(f"Test samples directory not found: {test_samples_dir}")
        print("Create test_samples/overlap/ directory and add .pth files to test")
