import torch


x = torch.tensor(4, dtype=torch.float32, requires_grad=True)

def fun(x):
	if x < -1:
		return 2 * x + 1
	elif x < 1:
		return 3 * x**3
	else: 
		return -x**2

f = fun(x)
f.backward()
print(x.grad) # df/dx



x = torch.tensor([1,2,3], dtype=torch.float32, requires_grad=True)
y = x**2
v = torch.tensor([0.1, 1.0, -1.3], dtype=torch.float32)
y.backward(v)
print(x.grad) # dy/dx


A = torch.tensor([[1,-1], [2,-3]])
print(torch.abs(A).sum())
