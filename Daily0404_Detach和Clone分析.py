
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
#一.detach
##1.原始数据require_grad为False
###1.1 不修改detach的数据
x=torch.tensor([2,4],dtype=torch.float)
y=x.detach()
print(x.is_leaf,y.is_leaf)#True True
y.requires_grad_()#修改y的req_grad
print(x.requires_grad,y.requires_grad)#False True
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)


#%%
### 1.2修改detach数据
x=torch.tensor([2,4],dtype=torch.float)
y=x.detach()
y[0]=1
print(x.is_leaf,y.is_leaf)#True True
y.requires_grad_()#修改y的req_grad
##在y更改requires_grad之后，
##不能再对y进行修改，将导致求导失败.
##这是autograd的机制决定的，与.detach无关
print(x.requires_grad,y.requires_grad)#False True
z=y**2
z=z.sum()
z.backward()
print(y,x)
print(y.grad,x.grad)


#%%
##2. 原始数据require_grad为True
###2.1 不修改detach的数据
x=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
y=x.detach()
print(x.is_leaf,y.is_leaf)#True False
y.requires_grad_()#修改y的req_grad
print(x.requires_grad,y.requires_grad)#True False
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)


#%%
###2.2 修改detach的数据
x=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
y=x.detach()
y[0]=1
print(x.is_leaf,y.is_leaf)#True True
y.requires_grad_()#修改y的req_grad
print(x.requires_grad,y.requires_grad)#True True
z=y**2
z=z.sum()
z.backward()
print(y,x)
print(y.grad,x.grad)


#%%
#一.clone
##1.原始数据require_grad为False
###1.1 不修改clone的数据
x=torch.tensor([2,4],dtype=torch.float)
y=x.clone()
print(x.is_leaf,y.is_leaf)#True True
print(x.requires_grad,y.requires_grad)
print(x.grad_fn,y.grad_fn)
y.requires_grad_()#修改y的req_grad
print(x.requires_grad,y.requires_grad)#False True
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)


#%%
###1.2 修改clone的数据
x=torch.tensor([2,4],dtype=torch.float)
y=x.clone()
y[0]=1
print(x.is_leaf,y.is_leaf)#True True
y.requires_grad_()#修改y的req_grad
print(x.requires_grad,y.requires_grad)#False True
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)


#%%
##2. 原始数据require_grad为True
###2.1 不修改clone的数据
x=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
y=x.clone()#y不是叶子节点：因为是x操作产生的节点，且require_grad为True
print(x.is_leaf,y.is_leaf)#True False
print(x.requires_grad,y.requires_grad)
print(x.grad_fn,y.grad_fn)
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)#对y求导，反应到x上


#%%
###2.2 修改clone的数据
x=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
y=x.clone()#y不是叶子节点：因为是x操作产生的节点，且require_grad为True
y[0]=2
y.retain_grad()
print(x.is_leaf,y.is_leaf)#True False
print(x.requires_grad,y.requires_grad)
print(x.grad_fn,y.grad_fn)
z=y**2
z=z.sum()
z.backward()
print(y.grad,x.grad)


#%%



