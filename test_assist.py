#%%
import torch
import torchvision as tv

#%%
mnist_data=tv.datasets.MNIST('/home/hh/dataset/mnist',True,None,None,True)

#%%
l=[1,2,3,5]
l[1:]

#%%
l[1:-1]

#%%
a=torch.randn((2,3,2))
b=torch.randn((3,3,2))
c=torch.cat((a,b))
c.shape

#%%
import numpy as np
period=np.arange(3)
a,b=np.meshgrid(period,period)
print(a,'\n\n',b)


#%%
np.vstack((a.flatten(),b.flatten()))


#%%
a.repeat(2,axis=1)

#%%
t=np.resize(a,(2,3,3))
t.reshape(3,-1)
#%%
(13*13+26*26+52*52)*3

#%%
print(a,'\t',b)

#%%
torch.stack((a,b))

#%%
x=torch.tensor([2,3],dtype=torch.float,requires_grad=True)
y=torch.as_tensor(x)
print(y.requires_grad)
print(y.grad_fn)
z=(y**2).sum()
z.backward()
print(x.grad)
print(y.grad)

#%%
y.grad

#%%
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1=nn.Linear(2,3)
        self.layer2=nn.Linear(3,1)
    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)

net = Net()
for param in net.modules():
    print(param)

#%%
layer=nn.Linear(2,3)
layer.weight.shape

#%%
layer.bias.shape

#%%
