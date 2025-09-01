import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # this is a single layer
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model.forward(x)

if __name__ == "__main__":
    xor = torch.Tensor([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]).to(device)

    model = MLP().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    epochs = 200
    for epoch in range(epochs):
        out = model(xor[:, :2])
        optimizer.zero_grad()
        loss = loss_fn(out, xor[:, -1].unsqueeze(1))

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"predicted: {out.tolist()}, expected: {xor[:, -1].tolist()}")
            print(f"epoch {epoch}, loss: {loss.item()}")

    with torch.no_grad():
        predicted = model(xor[:, :2])
        predicted = (predicted > 0.9).float()
        print("\npredicted:", predicted.tolist())
        print("actual:", xor[:, -1].unsqueeze(1).tolist())

