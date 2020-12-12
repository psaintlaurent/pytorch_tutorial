import torch
import math
import numpy as np
import matplotlib.pyplot as plt

class Polynomial3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))


    def forward(self, x):
        return self.a + self.b * x + self.c * x **2 + self.d * x ** 3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

    def result(self):
        return

    def plot(self):
        pred_vals, actual_vals = [], []
        input = np.arange(-math.pi, math.pi, math.pi / 20)
        for x in input:
            pred = (self.a.item() + (self.b.item() * x ** 1) + (self.c.item() * x ** 2) + (self.d.item() * x ** 3))
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

model = Polynomial3()
criterion = torch.nn.MSELoss(reduction='sum') #mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6) #stochiastic gradient descent

for t in range(2000):
    #forward pass
    y_pred = model(x)

    #compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad() #zero the gradient
    loss.backward() #backward pass through the network
    optimizer.step() #iterate one through the parameter optimization function

print(f'Result: {model.string()}')

