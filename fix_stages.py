#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lab02_starter import make_toy_classification

# Stage 0: Original Broken Baseline
class Stage0_BrokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(20 if i == 0 else 32, 32) for i in range(8)])
        self.out = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0.0, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)

# Stage 1: Fix Initialization Only (Kaiming Normal + Zero Bias)
class Stage1_FixInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(20 if i == 0 else 32, 32) for i in range(8)])
        self.out = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)

# Stage 2: Fix Normalization Only (BatchNorm1d added to broken init)
class Stage2_FixNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(20 if i == 0 else 32, 32) for i in range(8)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(8)])
        self.out = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0.0, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = torch.relu(self.norms[i](layer(x)))
        return self.out(x)

# Stage 3: Fix Init + Normalization Combined
class Stage3_FixInitNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(20 if i == 0 else 32, 32) for i in range(8)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(8)])
        self.out = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = torch.relu(self.norms[i](layer(x)))
        return self.out(x)

def train(model, optimizer_type="sgd", lr=0.1, epochs=50):
    torch.manual_seed(0)
    x, y = make_toy_classification(seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr) if optimizer_type == "sgd" else torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def main():
    h0 = train(Stage0_BrokenNet(), "sgd", 0.1)
    h1 = train(Stage1_FixInit(), "sgd", 0.1)
    h2 = train(Stage2_FixNorm(), "sgd", 0.1)
    h3 = train(Stage3_FixInitNorm(), "sgd", 0.1)
    h4 = train(Stage3_FixInitNorm(), "adamw", 0.01)

    print("=== Final Losses per Stage ===")
    print(f"Stage 0 (Baseline):          {h0[-1]:.4f}")
    print(f"Stage 1 (+ Kaiming Init):     {h1[-1]:.4f}")
    print(f"Stage 2 (+ BatchNorm Only):   {h2[-1]:.4f}")
    print(f"Stage 3 (+ Init & BatchNorm): {h3[-1]:.4f}")
    print(f"Stage 4 (+ AdamW):            {h4[-1]:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(h0, label="Stage 0: Baseline")
    plt.plot(h1, label="Stage 1: + Kaiming Init")
    plt.plot(h2, label="Stage 2: + BatchNorm Only")
    plt.plot(h3, label="Stage 3: + Init & BatchNorm (SGD)")
    plt.plot(h4, label="Stage 4: + Init & BatchNorm (AdamW)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AIGC 5500 Lab 02: Stages Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("loss_comparison.png")
    print("Saved updated loss_comparison.png!")

if __name__ == "__main__":
    main()


# In[ ]:




