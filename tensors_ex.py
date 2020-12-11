# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")

#create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)
e = torch.randn((), device=device, dtype=dtype)
learning_rate = 1e-6

for t in range(2000):
    y_pred = a + (b * x) + (c * x ** 2) + (d * x ** 3)

    #compute and print loss
    loss = (y_pred - y).pow(2).sum().item()

    if t % 100 == 99:
        print(t, loss)

    #backpropogation to compute gradients of a, b, c, d, e with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    #update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} * x + {c.item()} * x ** 2 + {d.item()} * x ** 3')

def test_results():
    pred_vals, actual_vals = [], []
    input = np.arange(-math.pi, math.pi, math.pi/20)
    for x in input:
        pred = a.item() + b.item() * x + c.item() * (x ** 2) + d.item() * (x ** 3)
        actual = math.sin(x)

        print("Iteration: %s, Prediction %s, Expected: %s " % (x, pred, actual))
        pred_vals.append(pred)
        actual_vals.append(actual)

    fig, ax = plt.subplots()
    ax.plot(input, pred_vals, label='Predicted')
    ax.plot(input, actual_vals,  label='Actual')
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Predicted/Actual Outputs')
    plt.show()

test_results()