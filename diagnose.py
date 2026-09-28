#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install -r requirements.txt')


# In[4]:


import torch
import torch.nn as nn
from lab02_starter import BrokenNet, make_toy_classification

def diagnose_gradients():
    torch.manual_seed(0)
    model = BrokenNet()
    x, y = make_toy_classification(seed=0)

    preds = model(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
    loss.backward()

    print("=== BrokenNet Initial Gradient Check ===")
    print(f"Loss after 1 pass: {loss.item():.4f}\n")
    print(f"{'Layer':>7} | {'Mean Absolute Weight Gradient':>30}")
    print("-" * 42)

    grad_norms = []
    for i, layer in enumerate(model.hidden):
        grad_mean = layer.weight.grad.abs().mean().item()
        grad_norms.append(grad_mean)
        print(f"Layer {i:>2} | {grad_mean:>30.4e}")

if __name__ == "__main__":
    diagnose_gradients()


# In[ ]:




