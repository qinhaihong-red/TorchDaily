
#%%
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim


#%%
X=torch.arange(1,7,dtype=torch.float).reshape(3,2)
b=torch.tensor([7,8,9],dtype=torch.float).reshape(3,-1)
l=nn.Linear(2,1,bias=False)
E=nn.MSELoss()
opt=optim.SGD(l.parameters(),lr=0.04,momentum=0.9)
sheduler=optim.lr_scheduler.MultiStepLR(opt,[10,30,60],0.5)
print(X,'\n',b)


#%%
print(l.weight.data,id(l.weight),id(l.weight.data),id(l.weight.grad))
loss=E(l(X),b)
loss.backward()
opt.step()
print(l.weight.data,id(l.weight),id(l.weight.data),id(l.weight.grad))
loss=E(l(X),b)
loss.backward()
opt.step()
print(l.weight.data,id(l.weight),id(l.weight.data),id(l.weight.grad))
loss=E(l(X),b)
loss.backward()
opt.step()
print(l.weight.data,id(l.weight),id(l.weight.data),id(l.weight.grad))


#%%
id((opt.param_groups[0]['params'][0]))


#%%
l.zero_grad()
print(l.weight,l.weight.is_leaf)


#%%
l.weight.grad_fn


#%%
l._parameters


#%%
id(l.weight.data[0,0])


#%%
E=nn.MSELoss()
opt=optim.SGD(l.parameters(),lr=0.04,momentum=0.9)
sheduler=optim.lr_scheduler.MultiStepLR(opt,[10,30,60],0.5)


#%%
def train():
    for i in range(2):
        opt.zero_grad()
        print('round:',i)
        print(id(l.weight),id(l.weight.data))
        l.weight.data[0,0]=1
        loss=E(l(X),b)
        
        

        loss.backward()
        opt.step()
        sheduler.step()
        print(id(l.weight),id(l.weight.data))

        print()

train()


#%%
a=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
print(id(a))

y=a**2
a.data[0]=1

z=y.sum()

#a.detach_()
print(id(a))
#a[0]=3
#a.requires_grad_()
print(y.requires_grad,a.requires_grad,y.is_leaf)
z.backward()

print(a.grad,a.is_leaf,a.grad_fn,y.grad)
a.grad.is_leaf


#%%
a=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
print(id(a),a.is_leaf,a.grad_fn)
a[0]=1
print(id(a),a.is_leaf,a.grad_fn)


#%%
a=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
print(id(a),a.is_leaf,a.grad_fn)
a.data[0]=1
print(id(a),a.is_leaf,a.grad_fn)


#%%
print(type(a),type(a.data),type(a.grad))
print(id(a),id(a.data),id(a.grad))


#%%
a.data.requires_grad_()
a.data


#%%
a.grad.requires_grad_()
a.grad


#%%
a.data.requires_grad_


#%%
a.data.requires_grad_(True)
a.data.is_leaf
a.data


#%%
a


#%%
a[0]=1
id(a)


#%%
a=torch.tensor([2,4],dtype=torch.float,requires_grad=True)
b=a**2
z=b.sum()
a.data[0]=1

b=torch.rand_like(a)


#%%
a


#%%
#b.data[0]=1
a


#%%
c=b**2
d=c.sum()
d.backward()


#%%
a.grad


#%%
get_ipython().run_line_magic('pinfo', 'torch._C._TensorBase')


#%%



