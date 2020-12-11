import torch
import math


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

for t in range(2000):

    #forward pass through the NN
    y_pred = model(xx)
    #compute loss
    loss = loss_fn(y_pred, y)
    learning_rate = 1e-6

    if t % 100 == 99:
        print(t, loss.item())

    #zero the gradients before the bakward pass
    model.zero_grad()
    #backward pass through the NN
    loss.backward()

    #update the weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} + x^3')
