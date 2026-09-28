#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lab02_starter import make_toy_classification


# In[5]:


class Stage0_BrokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(20 if i == 0 else 32, 32) for i in range(8)])
        self.out = nn.Linear(32, 1)
        for layer in self.hidden:
            nn.init.normal_(layer.weight, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)


# In[10]:


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


# In[16]:


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


# In[20]:


def main():
    h0 = train(Stage0_BrokenNet(), "sgd", 0.1)
    h1 = train(Stage1_FixInit(), "sgd", 0.1)

    plt.plot(h0, label="Stage 0: Baseline")
    plt.plot(h1, label="Stage 1: Fixed Init")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_comparison.png")
    print("Saved loss_comparison.png!")

if __name__ == "__main__":
    main()


# In[ ]:




