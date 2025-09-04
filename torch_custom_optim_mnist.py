import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# TODO: load data up here

# mnist with custom optimizer

class SimpleMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
                # 784 features = 28*28 pixel image
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
        )
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.seq1(x)
        self.act_output(x)

def train(model, loss_fn, optimizer, epochs: int, batch_size: int):
    for epoch in (t := trange(epochs)):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_trian[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            optimizer.zero_grad()

            logits = model.forward(X_batch)
            loss = loss_fn(out, Y_batch)

            loss.backward()
            optimizer.step()

            t.set_description("loss %.2f" & (loss))


model = SimpleMnist()

loss_fn = nn.CrossEntropyLoss()
#optimizer = None
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 100
batch_size = 32

