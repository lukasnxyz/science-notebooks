# physics informed neural networks (pinns)
# https://medium.com/data-science/solving-differential-equations-with-neural-networks-afdcf7b8bcc4
# - logistic differential equation (used to model population growth)
#   -> df/dt = R f(t)(1- f(t))
#   -> dfnn/dt - R fnn(t)(1 - fnn(t)) = 0
#      ^ can just do mean squared error over this
# and now just loss func: L = Lde + Lbc

import torch
import torchopt
from torch import nn, optim
from torch.func import functional_call, grad, vmap
import matplotlib.pyplot as plt

class LinearNN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_layers: int = 1,
        num_neurons: int = 5,
        act: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Linear(self.num_inputs, num_neurons)) # input layer
        for _ in range(num_layers): # hidden layers + act funcs
            layers.extend([nn.Linear(num_neurons, num_neurons), act])
        layers.append(nn.Linear(num_neurons, 1)) # output layer

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(-1, 1)).squeeze()

model = LinearNN()

def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter]) -> torch.Tensor:
    return functional_call(model, params, (x, ))

dfdx = vmap(grad(f), in_dims=(0, None))
d2fdx2 = vmap(grad(grad(f)), in_dims=(0, None))

R = 1.0
X_BOUNDARY = 0.0
F_BOUNDARY = 0.5

def loss_fn(params: torch.Tensor, x: torch.Tensor):
    # interior loss
    f_value = f(x, params)
    interior = dfdx(x, params) - R * f_value * (1 - f_value)

    # boundary loss
    x0 = X_BOUNDARY
    f0 = F_BOUNDARY
    x_boundary = torch.tensor([x0])
    f_boundary = torch.tensor([f0])
    boundary = f(x_boundary, params) - f_boundary

    loss = nn.MSELoss()
    loss_value = loss(interior, torch.zeros_like(interior)) + loss(
        boundary, torch.zeros_like(boundary)
    )

    return loss_value

batch_size = 30
num_iter = 100
learning_rate = 1e-1
domain = (-5.0, 5.0)

optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

params = tuple(model.parameters())

for i in range(num_iter):
    x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[i])
    loss = loss_fn(params, x)
    params = optimizer.step(loss, params)
    print(f"Iteration {i} with loss {float(loss)}")

