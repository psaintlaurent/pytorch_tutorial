# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6

for t in range(2000):
    y_pred = (a + b * x) + (c * x ** 2) + (d * x ** 3)
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    #back propogation
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

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