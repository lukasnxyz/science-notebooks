import torch

torch.manual_seed(42)

class CustomAdam(): pass

# applies an affine linear transformation to the incoming data:
#   y = xA.T + b
model = torch.nn.Linear(10, 1)

for param in model.parameters(): print(param)
print()

#optimizer = CustomAdam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# dummy data
x = torch.randn(16, 10)
y = torch.randn(16, 1)

loss_fn = torch.nn.MSELoss()

for _ in range(200):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

print("final loss:", loss.item())
