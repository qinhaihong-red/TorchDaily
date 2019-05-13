
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1. 参数类型Parameter

#%%
## Paramter是tensor的子类
## 默认添加了 requires_grad 为 True
p=nn.Parameter(torch.randn(3))
p


#%%
p.is_leaf


#%%
p.requires_grad


#%%
p.data


#%%
p.grad is None


#%%
p.grad_fn is None


#%%
z=(p**2).sum()
z.backward()
p.grad

#%% [markdown]
# ## 2.Parameter容器（一）：ParameterList

#%%
##初始化或者为None,或者为一个可推断的迭代器
pl=nn.ParameterList([nn.Parameter(torch.randn(i)) for i in range(1,3)])
pl


#%%
pl[0]


#%%
pl[1]


#%%
##添加与扩展
pl.append(nn.Parameter(torch.randn(3)))
pl.extend([nn.Parameter(torch.randn(i)) for i in range(4,6)])
pl

#%% [markdown]
# ## 3.Parameter容器（二）：ParameterDict

#%%
##初始化或者为None，或者为可迭代的推断
pd=nn.ParameterDict({a:nn.Parameter(torch.randn(i)) for i,a in enumerate('abc')})
pd


#%%
##添加
pd['d']=nn.Parameter(torch.randn(ord('d')-ord('a')))


#%%
pd


#%%
pd.keys()


#%%
pd.values()


#%%
##更新
pd.update({a:nn.Parameter(torch.randn(i)) for i,a in enumerate('efg')})


#%%
pd

#%% [markdown]
# ## 4.Module
#%% [markdown]
# 所有神经网络的模块和层的基类，可以通过它嵌套构成树状结构。
# 
# #### 重要成员：
# -  _parameters:OerderedDict类型，value类型是Parameter. 登记在本模块的参数.
# -  _buffers:OerderedDict类型，value类型是torch.Tensor. 登记在本模块供持久化的缓存.
# -  _modules:OerderedDict类型，value类型是Module. 添加在本模块的子模块.
# -  _backward_hooks
# -  _forward_pre_hooks
# -  _forward_hooks
# 
# ### 重要函数：
# - forward:任何继承类都需要重写的函数
# - register_parameter
# - register_buffer
# - add_module
# - register_backward_hook
# - register_forkward_hook
# - register_forkward_pre_hook
# - parameter/named_parameter：递归yield本模块和所有子模块的所有参数
# - children/named_children：yield本模块的所有子模块
# - modules/named_modules：递归yield本模块和所有子模块

#%%
##4.1 个卷积层+1个最大池化+2个ReLU激活的神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1)#输入3个通道，输出16个通道，核为3，padding为1，计算可得卷积后图片尺寸不变
        self.pool2=nn.AdaptiveMaxPool2d(8)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        return F.relu(self.pool2(x))


#%%
input=torch.randn(20*3*64*64).reshape(20,3,64,64)


#%%
n=Net()
out=n(input)
out.shape


#%%
##4.2 apply(fn)
##对所有子模块（直属的子模块，不递归）施加fn的作用，一般用以初始化
def init(m):
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(0.1)
        m.bias.data.fill_(0.1)

n2=Net()
n2.apply(init)
n2._modules['conv1'].bias


#%%
##4.3 children
##children只迭代直接的子模块
for m in n2.children():
    print(m)


#%%
##4.4 named_children
for n,m in n2.named_children():
    print(n,'-->',m)


#%%
##4.5 modules
##递归所有的模块，包括子模块
for m in n2.modules():
    print(m)


#%%
##4.6 named_modules
for name,m in n2.named_modules():
    print(name,'-->',m)


#%%
##4.7 parameters
for p in n2.parameters():
    print(p.shape)


#%%
##4.8 named_parameters
for n,p in n2.named_parameters():
    print(n,'-->',p.data.shape)


#%%
##登记一个参数
n2.register_parameter('param1',nn.Parameter(torch.randn(3)))


#%%
for n,p in n2.named_parameters():
    print(n,'-->',p.data.shape)


#%%
n2._parameters['param1'].data


#%%
##4.9 register_backward_hook
def b_hook(m,grad_in,grad_out):
    print('backward hook called')

n2.register_backward_hook(b_hook)
out=n2(input)
type(out)


#%%
z=out.sum()
z.backward()


#%%
##4.10 登记一个buffer
n2.register_buffer('buf1',torch.randn(3))


#%%
##4.11 register_forward_hook
def f_hook(m,input,out):
    print('forward hook called')
    
n2.register_forward_hook(f_hook)    
out2=n2(input)


#%%
##4.12 register_pre_forward_hook
def pf_hook(m,input):
    print('pre forward_hook called')

n2.register_forward_pre_hook(pf_hook)
out2=n2(input)


#%%
##4.13 state_dict
##以字典形式返回模块内所有的参数和buf
##包括所有子类的参数和buf
n2._modules['conv1'].register_parameter('param2',nn.Parameter(torch.randn(2)))
d=n2.state_dict()
d.keys()


#%%
##4.14 转移或转型方法：to
linear=nn.Linear(2,2)
linear.weight


#%%
##参数的类型转换
linear.to(torch.double)
linear.weight


#%%
gpu1=torch.device('cuda:0')
linear.to(gpu1,dtype=torch.half,non_blocking=True)


#%%
linear.weight


#%%
cpu=torch.device('cpu')
linear.to(cpu)


#%%
linear.weight


#%%
## 4.14 zero_grad
##把所有参数的梯度清零（grad.detach_()+grad.zero_()）
##如果没有梯度，即为None，则不进行处理
n2._parameters['param1'].grad is None


#%%
n2._modules['conv1']._parameters['param2'].grad is None


#%%
n2.zero_grad()
print(n2._parameters['param1'],'\n',n2._modules['conv1']._parameters['param2'])

#%% [markdown]
# ## 5.Sequential容器
#%% [markdown]
# 顺序容器，按传入module的顺序，构建网络，此时默认以序号（从0开始）作为module的名称，内部按add_module进行添加.
# 
# 也可以用OrderedDict作为参数传入.内容同样用add_module添加.

#%%
s=nn.Sequential(
    nn.Conv2d(3,16,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(16,16,3,padding=1),
    nn.ReLU()
)
s.apply(init)
s._modules['0'].bias


#%%
for name,m in s.named_modules():
    print(name,'-->',m)


#%%
##使用OrderedDict构建
from collections import OrderedDict


#%%
s2=nn.Sequential(OrderedDict({
    'conv1':nn.Conv2d(3,16,3,padding=1),
    'relu1':nn.ReLU(),
    'conv2':nn.Conv2d(16,16,3,padding=1),
    'relu2':nn.ReLU()
    
}))

for name,m in s2.named_modules():
    print(name,'-->',m)

#%% [markdown]
# ## 6.ModuleList容器
#%% [markdown]
# 以列表的方式维持子模块.
# 
# - 初始化：None或者可迭代的推断.
# - 方法：
#     - append
#     - extend    

#%%
class Net2(nn.Module):
    def __init__(self,n):
        super(Net2,self).__init__()
        self.linears=nn.ModuleList([nn.Linear(i,i+1) for i in range(2,n)])
        
    def forward(self,x):
        for i ,l in enumerate(self.linears):
            x=self.linears[i](x)
            
        return x


#%%
net=Net2(4)
input=torch.randn(2,2)
out=net(input)
out


#%%
type(out)


#%%
for name,m in net.named_modules():
    print(name,'-->',m)


#%%
for name,p in net.named_parameters():
    print(name,'-->',p)

#%% [markdown]
# ## 7.ModuleDict容器类
# 
# 以字典的方式维持子模块.
# 
# - 初始化：以字典或者可迭代推断进行.
# - 方法：
#     - 字典添加
#     - pop/clear
#     - items/keys/values
#     - update

#%%
class Net3(nn.Module):
    def __init__(self,mapping=None):
        super(Net3,self).__init__()
        if mapping is not None and isinstance(mapping,dict):
            self.layers=nn.ModuleDict(mapping)
        
        ##注意moduleDict会自动排序，不是OrderedDict
        self.act=nn.ModuleDict({
            'relu':nn.ReLU(),
            'asigmoid':nn.Sigmoid()
        })
    
    def forward(self,x):
        x=self.layers['conv1'](x)
        x=self.act['relu'](x)
        x=self.layers['pool1'](x)
        
        return self.act['asigmoid'](x)


#%%
layers={'conv1':nn.Conv2d(3,16,3,padding=1),'pool1':nn.AdaptiveMaxPool2d(8)}
net=Net3(layers)
input=torch.randn(4,3,12,12)
out=net(input)
out.shape


#%%
for name,p in net.named_parameters():
    print(name,'-->',p.data.shape)


#%%
for name,m in net.named_modules():
    print(name,'-->',m)


#%%



