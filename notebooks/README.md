# Notebooks

This directory contains Jupyter notebooks for generating test cases and visualizations.

## generate_overlap_tests.ipynb

Generates test cases for overlap computation with visualizations.

### Features:
- Creates 8 different test scenarios covering various collision cases
- Each scene has 6 joint predictions for 3 modeled agents
- Generates animated GIF visualizations
- Saves test data as `.pth` files for use in `tests/semantic_test_overlap.py`

### Test Cases:
1. **All collisions**: All predictions lead to collision
2. **Partial collisions**: Part of predictions lead to collision
3. **No collisions**: No collisions at all
4. **Highest confident collision**: Highest confident prediction leads to collision
5. **Lowest confident collision**: Lowest confident prediction leads to collision
6. **Middle confident collision**: Middle confident prediction leads to collision
7. **Early collision**: Collision happens early in trajectory
8. **Late collision**: Collision happens late in trajectory

### Usage:

1. Install dependencies:
```bash
pip install matplotlib pillow
```

2. Open the notebook:
```bash
jupyter notebook generate_overlap_tests.ipynb
```

3. Run all cells to generate test cases and visualizations

4. Test cases will be saved to `../test_samples/overlap/`
5. GIF visualizations will be saved alongside the test files

### Output:
- `.pth` files: Test data for semantic tests
- `.gif` files: Animated visualizations of each test case

