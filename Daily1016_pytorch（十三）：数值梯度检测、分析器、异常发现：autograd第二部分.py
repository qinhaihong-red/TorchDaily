
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1.数值梯度检测
# 
# ### 1.1 autograd.gradcheck

#%%
## 自定义sigmoid函数
class sig(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        out=1/(1+torch.exp(-x))
        return out
    
    @staticmethod
    def backward(ctx,grad_output):
        x,=ctx.saved_tensors
        return x*(1-x)*grad_output


#%%
x=torch.arange(2.,4.,requires_grad=True)
z=sig.apply(x)
z


#%%
z.backward(torch.tensor([1.,0.]),retain_graph=True)
x.grad


#%%
x.grad.zero_()
z.backward(torch.tensor([0.,1.]),retain_graph=True)
x.grad


#%%
##使用gradcheck进行检测
torch.autograd.gradcheck(sig.apply,(x,),eps=1e-03)

#%% [markdown]
# ### 1.2 autograd.gradgradcheck
#%% [markdown]
# ## 2. 分析器（Profiler）
# 
# autograd内部的分析器，可以分析模型内部不同计算操作的代价，不论是在CPU上，还是在GPU上.
# 
# 
# 为此同时实现了两种模型：使用profil只对CPU分析；使用emit_nvtx同时分析CPU和GPU.
#%% [markdown]
# ### 2.1 profile

#%%
x=torch.randn((2,2),requires_grad=True)
with torch.autograd.profiler.profile() as prof:##注意：profile()是一个context manager类型，因此可以使用 with...as...
    y=(x**2).sum()
    y.backward()


#%%
print(prof)

#%% [markdown]
# ### 2.2 emit_nvtx
#%% [markdown]
# ## 3. 异常发现（Anomaly Detection）
# 
# ### 3.1 detect_anomaly可以为autograd引擎开启异常发现，是一个上下文管理器.

#%%
##先看看tensor的clone方法
zc=x.clone()


#%%
zc


#%%
zc.requires_grad


#%%
zc.is_leaf


#%%
c=(zc*2).sum()
c.backward()
zc.grad


#%%
x.grad##对x的clone的梯度计算，最终会传递到x这里


#%%
class myFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return x.clone()
    
    @staticmethod
    def backward(ctx,grad_output):
        raise RuntimeError('some err in backward')
        return grad_output.clone()


#%%
def run_fn(x):
    out = myFunc.apply(x)
    return out.sum()


#%%
input = torch.randn((2,2),requires_grad=True)
out=run_fn(input)
out.backward()


#%%
with torch.autograd.detect_anomaly():
    input = torch.randn((2,2),requires_grad=True)
    out=run_fn(input)
    out.backward()##由于使用了detect_anomaly，因此会打印出相关的过程

#%% [markdown]
# ### 3.2 set_detect_anomaly(mode)
#%% [markdown]
# 既可以作为函数，也可以作为上下文管理器. 功能与detect_anomaly相同.
# 
# 如果mode为True，则开启异常发现；否则关闭.

#%%



