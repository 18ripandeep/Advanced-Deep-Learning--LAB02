# Making Training Work

Diagnosing and fixing a network that would not train, one change at a time.

## Setup

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run

python diagnose.py   # reproduces the zero-gradient failure mode
python fix_stages.py  # trains all stages and saves loss_comparison.png

## Example

Baseline (Stage 0): final loss 0.6931
Stage 1 (+ Kaiming Init): final loss 0.0210
Stage 2 (+ BatchNorm1d Only): final loss 0.0001
Stage 3 (+ Init & BatchNorm SGD): final loss 0.0000
Stage 4 (+ Init, BatchNorm AdamW): final loss 0.0000

## What changed and why

Running diagnose.py confirmed that weight gradients across all hidden layers were 0.0000e+00 due to dead ReLUs caused by the hardcoded -2.0 bias initialization. Stage 1 contributed the primary breakthrough by replacing the constant -2.0 bias with 0.0 and applying Kaiming normal initialization. This prevented ReLU saturation, restored backward gradient flow across all 8 hidden layers, and dropped the final loss from 0.6931 down to 0.0210.

## Known limitations

The toy dataset is small and linearly separable, causing losses to reach zero rapidly. Additionally, evaluation was conducted on a single fixed random seed (seed=0) rather than averaged across multiple runs.
