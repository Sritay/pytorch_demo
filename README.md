# PyTorch Quickstart Demo

This repo contains two minimal PyTorch examples that show:
- Creating tensors and synthetic datasets
- Defining models (`nn.Linear`, `nn.Sequential`)
- Training loops: forward → loss → backward → optimizer step
- Visualizing results with matplotlib

## Files
- `linear_regression_demo.py` — fits a straight line `y = 2.5x - 1` from noisy data
- `nonlinear_sine_demo.py` — fits a noisy `sin(x)` curve with a small MLP

## Setup
```bash
pip install -r requirements.txt

