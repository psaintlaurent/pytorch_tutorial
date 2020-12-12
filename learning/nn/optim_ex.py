import torch
import math
import matplotlib.pyplot as plt
import numpy as np


#create tensors for inputs and outputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

#prepare input tensors
p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)

#use the NN package
model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

#set the loss function, learning rate and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    #zero out the gradients before the backward pass
    optimizer.zero_grad()

    #backward pass
    loss.backward()
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

def test_result():

    pred_vals, actual_vals = [], []
    input = np.arange(-math.pi, math.pi, math.pi/20)
    for x in input:
        pred = linear_layer.bias.item() + linear_layer.weight[:, 0].item() * x + linear_layer.weight[:, 1].item() * x ** 2 + linear_layer.weight[:, 2].item() * x ** 3
        actual = math.sin(x)

        print('Input: %s, Predicted: %s, Actual: %s' % (x, pred, actual))
        pred_vals.append(pred)
        actual_vals.append(actual)

    fig, ax = plt.subplots()
    ax.plot(input, pred_vals, label='Predicted')
    ax.plot(input, actual_vals,  label='Actual')
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Predicted/Actual Outputs')
    plt.show()

test_result()