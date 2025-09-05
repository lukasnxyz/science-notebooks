import torch
import matplotlib.pyplot as plt
from tqdm import trange

#device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = "cpu"

alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1
params = torch.tensor([alpha, beta, gamma, delta], device=device)

# lotka-volterra predator-prey system
#   dx/dt = alpha x - beta x y
#   dy/dt = delta x y - gamma y

x = torch.tensor(10.0, device=device) # prey
y = torch.tensor(5.0, device=device) # predator

dt = 0.001
steps = 10_000

xs, ys = [], []

for _ in trange(steps):
    alpa, beta, gamma, delta = params

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    x = x + dx * dt
    y = y + dy * dt

    xs.append((x.sum() / len(x)).item())
    ys.append((y.sum() / len(x)).item())

xs = torch.tensor(xs)
ys = torch.tensor(ys)

plt.plot(xs, label="prey")
plt.plot(ys, label="predator")
plt.legend()
plt.show()

