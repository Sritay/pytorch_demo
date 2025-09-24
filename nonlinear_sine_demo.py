# nonlinear_sine_demo.py
# ------------------------------------------------------------
# Fit a noisy sine curve with a tiny MLP (nn.Sequential with ReLU)
# Shows: non-linear function approximation with a simple training loop
# ------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Synthetic data: y = sin(x) + noise
torch.manual_seed(0)
x = torch.linspace(-2*math.pi, 2*math.pi, steps=200).unsqueeze(1)  # (200,1)
y_clean = torch.sin(x)
y = y_clean + 0.1 * torch.randn_like(x)                            # noisy targets

# 2) Small MLP: 1 -> 32 -> 32 -> 1 with ReLU (enables non-linear fits)
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 3) Loss + optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 4) Training loop with simple progress prints
for step in range(2000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step:4d} | loss={loss.item():.4f}")

# 5) Plot: data, true sine, and model fit
with torch.no_grad():
    y_fit = model(x)

plt.figure(figsize=(8,4))
plt.scatter(x, y, s=10, alpha=0.5, label="noisy data")
plt.plot(x, y_clean, linewidth=2, label="true sin(x)")
plt.plot(x, y_fit, linewidth=2, label="NN fit")
plt.title("Nonlinear Regression with a Tiny MLP")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

