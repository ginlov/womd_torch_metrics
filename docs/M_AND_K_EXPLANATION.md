# Understanding M and K in Waymo Open Motion Dataset Metrics

## Overview

In the Waymo Open Motion Dataset metrics, **M** and **K** are key dimensions that define the structure of motion predictions.

## Definitions

### M: Number of Joint Prediction Groups

**M** represents the number of **joint prediction groups** per scenario.

- **Joint Prediction Group**: A set of predictions for multiple agents that may interact with each other (e.g., two vehicles at an intersection, a pedestrian and a vehicle).

- **Marginal Predictions** (N=1): When predicting agents independently (mutual independence assumed), M typically equals the number of agents you want to predict. Each joint prediction group contains predictions for a single agent.

- **Joint/Interaction Predictions** (N>1): When predicting interacting agents together, M is typically 1 (one joint prediction for the interacting pair), and N=2 (two agents in the joint prediction).

**Example:**
- Scenario with 3 agents, marginal predictions: M=3, N=1, K=6
  - Group 0: 6 predictions for agent 0
  - Group 1: 6 predictions for agent 1  
  - Group 2: 6 predictions for agent 2

- Scenario with 2 interacting agents: M=1, N=2, K=6
  - Group 0: 6 joint predictions, each containing trajectories for both agents

### K: Top-K Predictions per Joint Prediction

**K** represents the number of trajectory hypotheses (modes) predicted for each joint prediction group.

- Models typically output multiple possible future trajectories (multimodal predictions)
- K is the number of these trajectory modes (e.g., K=6 means 6 different possible futures)
- Each mode has a confidence score
- Metrics like **minADE** and **minFDE** compute the **minimum** error across these K predictions
- Common values: K=6 or K=8

**Example:**
- K=6 means for each joint prediction group, the model predicts 6 different possible trajectories
- The metrics select the best one (minimum error) among these 6

## Tensor Shape Notation

From Waymo's official documentation:

```
prediction_trajectory: [B, M, K, N, T, 2]
```

Where:
- **B**: Batch size (number of scenarios). Each batch should contain exactly 1 scenario.
- **M**: Number of joint prediction groups to predict per scenario
- **K**: Top-K predictions per joint prediction
- **N**: Number of agents in a joint prediction. 1 if mutual independence is assumed.
- **T**: Number of prediction timesteps (e.g., 80 for 8 seconds at 10Hz)
- **2**: (x, y) coordinates

## Official Waymo References

1. **GitHub Repository**: 
   - https://github.com/waymo-research/waymo-open-dataset
   - File: `waymo_open_dataset/metrics/python/motion_metrics.py`
   - Function: `get_motion_metric_ops()`

2. **Code Documentation** (from waymo-open-dataset):
   ```python
   # From waymo_open_dataset/metrics/python/motion_metrics.py
   # - Notations:
   #   - B: batch size. Each batch should contain 1 scenario.
   #   - M: Number of joint prediction groups to predict per scenario.
   #   - N: number of agents in a joint prediction. 1 if mutual independence is
   #       assumed between agents.
   #   - K: top_K predictions per joint prediction.
   #   - A: number of agents in the groundtruth.
   #   - TP: number of steps to evaluate on.
   #   - TG: number of steps in the groundtruth track.
   ```

3. **Research Paper**:
   - "Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset"
   - Ettinger et al., ICCV 2021
   - https://arxiv.org/abs/2104.10133

4. **Motion Metrics Documentation**:
   - Waymo Open Dataset website: https://waymo.com/open/
   - Motion Prediction Challenge documentation

## Common Use Cases

### Case 1: Marginal Predictions (Most Common)
```python
# Predict 3 agents independently, 6 modes each
B, M, K, N, T = 1, 3, 6, 1, 80
# M=3: 3 joint prediction groups (one per agent)
# N=1: 1 agent per group (independent)
# K=6: 6 trajectory modes per agent
```

### Case 2: Joint Predictions (Interaction)
```python
# Predict 2 interacting agents jointly, 6 modes
B, M, K, N, T = 1, 1, 6, 2, 80
# M=1: 1 joint prediction group
# N=2: 2 agents in the joint prediction (interacting)
# K=6: 6 joint trajectory modes
```

### Case 3: Mixed (Some Independent, Some Joint)
```python
# 2 independent agents + 1 joint prediction for 2 interacting agents
# This would require separate metric computations or restructuring
```

## How Metrics Use M and K

1. **minADE/minFDE**: For each joint prediction group (M), compute the minimum error across all K predictions
2. **MissRate**: Check if the best (minimum FDE) among K predictions misses the threshold
3. **OverlapRate**: Check if any of the K predictions overlap with ground truth
4. **mAP**: Uses confidence scores across K predictions to compute precision-recall curves

## References in Code

The original Waymo implementation can be found at:
- `waymo_open_dataset/metrics/python/motion_metrics.py::get_motion_metric_ops()`
- `waymo_open_dataset/metrics/ops/motion_metrics_ops.cc`

The docstring in the original code explicitly defines these dimensions as shown above.

