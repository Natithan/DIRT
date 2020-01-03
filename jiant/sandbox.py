import torch

d_in = 2
d_hidden = 2
d_out = 1
layer_1 = torch.tensor([[0.5,0],[.5,.5]],requires_grad=True)
layer_2 = torch.rand(d_hidden,d_out,requires_grad=True)
layer_1.requires_grad_(True)
input = torch.tensor([1.,1.],requires_grad=True)
act_1 = torch.matmul(layer_1,input)
act_2 = torch.matmul(layer_2.transpose(0,1),act_1)
correct = -2
loss = .5*(correct - act_2)**2
print(5)


