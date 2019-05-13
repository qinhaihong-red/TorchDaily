
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# #  BatchNorm
# 
# 批标准化要注意区分是否在training mode 和 evaluation mode.
# 
# 因为在traning阶段需要持续统计batch上的均值与方差，以便在推断阶段使用.
# 
# 
# 参考：https://pytorch.org/docs/0.4.0/nn.html#torch.nn.BatchNorm1d
# 
# 见：https://arxiv.org/abs/1502.03167
#%% [markdown]
# # 1. BatchNorm1d
# 
# ## 1.1 使用nn.BatchNorm1d

#%%
x = torch.arange(2,6,dtype=torch.float).reshape(2,1,2)
x


#%%
m=nn.BatchNorm1d(1,eps=0.,affine=False,track_running_stats=True)##参数或者是通道数，或者是长度
r=m(x)

print(r)

print(r.mean(),r.std())


#%%
m.__dict__

#%% [markdown]
# ## 1.2 手动进行标准化

#%%
mean=torch.mean(x)
std=torch.std(x)

print(mean,std)

n=(x-mean)/(std)


print(n)

print(n.mean(),n.std())

#%% [markdown]
# __The mean and standard-deviation are calculated per-dimension over the mini-batches__
# 
# 参考:https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d
#%% [markdown]
# # 2. BatchNorm2d

#%%
x=torch.arange(8,dtype=torch.float).reshape(2,2,1,2)
m=nn.BatchNorm2d(2,eps=0.,affine=False)##参数是通道数
x


#%%
m(x)


#%%
y=torch.tensor([0,1,4,5],dtype=torch.float32).reshape(2,-1)
y_mean=y.mean()
y_std=y.std()
print(y_mean,y_std)
y_norm=(y-y.mean())/y.std()
y_norm


#%%
y0=y[0,:]
print(y0)
y0_mean=torch.mean(y0)
y0_std=torch.std(y0)
print(y0_mean,y0_std)
y0_norm= (y0-y0_mean)/y0_std
print(y0_norm)

#%% [markdown]
# # 3. GroupNormal

#%%
##把通道分组然后进行标准化
x=torch.arange(24,dtype=torch.float).reshape(1,6,2,2)
m=nn.GroupNorm(3,6)##6个通道分为3组
x


#%%
m(x)

#%% [markdown]
# # 4. InstanceNorm1d
# 
# 
# 参考：https://pytorch.org/docs/0.4.0/nn.html#torch.nn.InstanceNorm1d
# 
# 见：https://arxiv.org/abs/1607.08022

#%%
x=torch.arange(1,13,dtype=torch.float).reshape(2,3,2)
m=nn.InstanceNorm1d(3)##参数是通道数
m(x)

#%% [markdown]
# # 5. InstanceNorm2d

#%%
x=torch.arange(32,dtype=torch.float).reshape(1,2,4,4)
m=nn.InstanceNorm2d(4)
m(x)

#%% [markdown]
# # 6. 层标准化LayerNormalization
# 
# 
# 参考：https://pytorch.org/docs/0.4.0/nn.html#torch.nn.LayerNorm
# 
# 见：https://arxiv.org/abs/1607.06450
#%% [markdown]
# # 7. LocalResponseNorm
# 
# 用于AlexNet的标准化方法.

#%%



