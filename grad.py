# gradient descent for linear regression
# yhat = wx + b
# loss_func = (y-yhat)**2 / N

import numpy as np
from typing import Tuple

# params
x = np.random.randn(10, 1) # 10 examples each in it's own row
y = 2*x + np.random.rand()
w = 0.0 # should be 2
b = 0.0 # should be 2*x

print(f"x: {x}\ny: {y}")

# hyper-params
learning_rate = 0.01
epochs = 400

# gradient descent func
def gd(x: np.ndarray, y: np.ndarray, w: float, b: float, lr: float) -> Tuple[float, float]:
    dl_dw = 0.0 # derivative of loss func with respect to w
    dl_db = 0.0 # derivative of loss func with respect to b

    N = x.shape[0] # number of rows (num of samples)

    # loss = (y-(wx+b))**2
    for xi, yi in zip(x, y):
        dl_dw += -2*xi*(yi-(w*xi+b)) # derive partial derivative via chain rule
        dl_db += -2*(yi-(w*xi+b)) # derive partial derivative via chain rule

    w = w - learning_rate*(1/N)*dl_dw # update w param
    b = b - learning_rate*(1/N)*dl_db # update b param

    return w, b

# iter updates
for epoch in range(epochs):
    w, b = gd(x, y, w, b, learning_rate)
    yhat = w*x+b
    loss = np.squeeze(np.divide(np.sum((y-yhat)**2, axis = 0), x.shape[0]), axis=0)
    print(f"{epoch} loss: {loss}, w: {w}, b: {b}")
