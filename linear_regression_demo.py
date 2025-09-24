# linear_regression_demo.py
# ------------------------------------------------------------
# Minimal PyTorch example: learn slope + intercept from noisy data
# Shows: tensors, nn.Linear, loss, backward(), optimizer.step(), plotting
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Make synthetic dataset: y = 2.5x - 1 + noise
torch.manual_seed(0)                          # reproducibility
x = torch.linspace(-2, 2, 50).unsqueeze(1)    # shape (50,1): column vector
noise = 0.2 * torch.randn_like(x)             # Gaussian noise
y = 2.5 * x - 1.0 + noise                     # targets

# 2) Define simple linear model (1 input -> 1 output)
#    nn.Linear internally stores weight and bias as trainable parameters
model = nn.Linear(1, 1)

# 3) Loss + optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4) Training loop
for step in range(201):
    optimizer.zero_grad()         # clear old gradients (PyTorch accumulates)
    y_pred = model(x)             # forward pass (calls model.__call__ -> forward)
    loss = loss_fn(y_pred, y)     # mean squared error
    loss.backward()               # compute d(loss)/d(parameters)
    optimizer.step()              # update parameters using gradients

    if step % 50 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"step {step:3d} | loss={loss.item():.4f} | w={w:.3f}, b={b:.3f}")

# 5) Plot fit
with torch.no_grad():
    y_fit = model(x)              # predictions with learned parameters

plt.figure(figsize=(7,4))
plt.scatter(x, y, s=20, label="data", alpha=0.7)
plt.plot(x, y_fit, linewidth=2, label="fit")
plt.title("Linear Regression (PyTorch)")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

