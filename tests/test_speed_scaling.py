"""Test speed scaling functionality."""

import torch
import pytest
from womd_torch_metrics.motion_metrics import (
    compute_speed,
    compute_speed_scale,
)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSpeedScaling:
    """Test speed computation and speed-based scaling."""
    
    def test_compute_speed(self, device):
        """Test speed computation from trajectory."""
        # Trajectory moving at 2 m/s for 5 seconds
        trajectory = torch.tensor([
            [0.0, 0.0],
            [0.2, 0.0],  # 0.2m in 0.1s = 2 m/s
            [0.4, 0.0],
            [0.6, 0.0],
            [0.8, 0.0],
            [1.0, 0.0],
        ], device=device)
        
        speed = compute_speed(trajectory, time_step_seconds=0.1)
        assert abs(speed.item() - 2.0) < 0.01
    
    def test_compute_speed_with_mask(self, device):
        """Test speed computation with validity mask."""
        trajectory = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],  # Invalid point
        ], device=device)
        
        valid_mask = torch.tensor([True, True, True, False], device=device)
        
        speed = compute_speed(trajectory, valid_mask, time_step_seconds=1.0)
        # Should compute speed from first 3 points: 2m / 2s = 1.0 m/s
        assert abs(speed.item() - 1.0) < 0.01
    
    def test_speed_scale_within_bounds(self, device):
        """Test speed scaling within bounds."""
        # Speed of 5.0 m/s (midpoint of [1.4, 11.0])
        speed = torch.tensor(5.0, device=device)
        
        scale = compute_speed_scale(
            speed,
            speed_lower_bound=1.4,
            speed_upper_bound=11.0,
            speed_scale_lower=0.5,
            speed_scale_upper=1.0,
        )
        
        # At midpoint, should be between 0.5 and 1.0
        assert 0.5 <= scale.item() <= 1.0
    
    def test_speed_scale_below_bounds(self, device):
        """Test speed scaling below lower bound."""
        speed = torch.tensor(0.5, device=device)  # Below 1.4
        
        scale = compute_speed_scale(
            speed,
            speed_lower_bound=1.4,
            speed_upper_bound=11.0,
            speed_scale_lower=0.5,
            speed_scale_upper=1.0,
        )
        
        assert scale.item() == 0.0
    
    def test_speed_scale_above_bounds(self, device):
        """Test speed scaling above upper bound."""
        speed = torch.tensor(15.0, device=device)  # Above 11.0
        
        scale = compute_speed_scale(
            speed,
            speed_lower_bound=1.4,
            speed_upper_bound=11.0,
            speed_scale_lower=0.5,
            speed_scale_upper=1.0,
        )
        
        assert scale.item() == 0.0
    
    def test_speed_scale_linear_interpolation(self, device):
        """Test linear interpolation of speed scaling."""
        # At lower bound: should get scale_lower
        speed_low = torch.tensor(1.4, device=device)
        scale_low = compute_speed_scale(speed_low, 1.4, 11.0, 0.5, 1.0)
        assert abs(scale_low.item() - 0.5) < 0.01
        
        # At upper bound: should get scale_upper
        speed_high = torch.tensor(11.0, device=device)
        scale_high = compute_speed_scale(speed_high, 1.4, 11.0, 0.5, 1.0)
        assert abs(scale_high.item() - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
