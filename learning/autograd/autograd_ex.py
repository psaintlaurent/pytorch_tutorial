import torch
import math
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 5000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(5000):
    y_pred = a + (b * x) + (c * x ** 2) + (d * x ** 3)

    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

    a.grad, b.grad, c.grad, d.grad = None, None, None, None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()}')

def test_results():

    pred_vals, actual_vals = [], []
    input = np.arange(-math.pi, math.pi, math.pi/20)
    for x in input:
        pred = (a.item() + (b.item() * x ** 1) + (c.item() * x ** 2) + (d.item() * x ** 3))
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
