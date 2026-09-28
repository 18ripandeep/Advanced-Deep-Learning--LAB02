
## Diagnosis
Gradients across all hidden layers are exactly 0.0 because of the -2.0 bias initialization, causing all ReLU units to turn off ("Dead ReLU").

## Conclusion
Stage 1 (Fixing Initialization) made the biggest impact by restoring gradient flow.
