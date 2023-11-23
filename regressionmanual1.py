import torch

x = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

def forward(x):
    return w*x

# loss MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'prediction before training:f(5)={forward(5):.3f}')

learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    y_pred = forward(x)
    l = loss(y,y_pred)
    #gradient = backward pass
    l.backward() #dl/dw
    #update weights
    with torch.no_grad():
        w -= learning_rate*w.grad
    w.grad.zero_()
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'prediction after training:f(5)={forward(5):.3f}') 