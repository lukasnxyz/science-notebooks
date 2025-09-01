import torch
from torch.optim import Optimizer

class MyAdam(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {'lr': lr}
        super(MyAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        return loss

