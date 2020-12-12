# -*- coding: utf-8 -*-

import random
import torch
import math
import numpy as np
import  matplotlib.pyplot as plt

class DynamicNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 + {self.e.item()} x^5'

    def plot(self):
        pred_vals, actual_vals = [], []
        input = np.arange(-math.pi, math.pi, math.pi / 20)
        for x in input:
            pred = (self.a.item() + (self.b.item() * x ** 1) + (self.c.item() * x ** 2) + (self.d.item() * x ** 3)
                    + (self.e.item() * x ** 4) + (self.e.item() * x ** 5))
            actual = math.sin(x)

            print("Iteration: %s, Prediction %s, Expected: %s " % (x, pred, actual))
            pred_vals.append(pred)
            actual_vals.append(actual)

        fig, ax = plt.subplots()
        ax.plot(input, pred_vals, label='Predicted')
        ax.plot(input, actual_vals, label='Actual')
        ax.set_xlabel('Inputs')
        ax.set_ylabel('Predicted/Actual Outputs')
        plt.show()

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

for t in range(30000):
    y_pred = model(x) #forward pass through the model

    loss = criterion(y_pred, y) #calculate and print loss
    if t % 2000 == 1999:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')