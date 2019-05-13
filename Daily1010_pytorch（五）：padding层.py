#%% [markdown]
# ### padding层就是给输入数据的边界做一定数量的扩充，以进行卷积和池化
# 有以下分类：
# - 镜像 padding
# - 复制 padding
# - 0 padding
# - 常数 padding

#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
#>一 镜像 padding
##>1.1 一维镜像padding
m1=nn.ReflectionPad1d(2)##left=right=2
m2=nn.ReflectionPad1d((2,1))##left=2,right=1
input=torch.arange(8,dtype=torch.float).reshape(1,2,4)
input


#%%
m1(input)


#%%
m2(input)


#%%
##>1.2 二维镜像padding
m1=nn.ReflectionPad2d(2)#left=right=top=bottom=2
m2=nn.ReflectionPad2d((1,1,2,0))#left=right=1,top=2,bottom=0
input=torch.arange(9,dtype=torch.float).reshape(1,1,3,3)
input


#%%
m1(input)


#%%
m2(input)


#%%
#>二 复制 padding
##>2.1 一维复制padding
m1=nn.ReplicationPad1d(2)
m2=nn.ReplicationPad1d((2,1))
input_1d=torch.arange(8,dtype=torch.float).reshape(1,2,4)
input_2d=torch.arange(9,dtype=torch.float).reshape(1,1,3,3)


#%%
m1(input_1d)


#%%
m2(input_1d)


#%%
##>2.2 二维复制padding
m1=nn.ReplicationPad2d(2)
m2=nn.ReflectionPad2d((1,1,2,0))
m1(input_2d)


#%%
m2(input_2d)


#%%
#>三 零padding
m1=nn.ZeroPad2d(2)
m2=nn.ZeroPad2d((1,1,2,0))
m1(input_2d)


#%%
m2(input_2d)


#%%
#>四 常数padding
##>4.1 一维常数padding
m1=nn.ConstantPad1d(2,1.1)
m2=nn.ConstantPad1d((2,1),1.2)
m1(input_1d)


#%%
m2(input_1d)


#%%
##>4.2 二维常数padding
m1=nn.ConstantPad2d(2,0.22)
m2=nn.ConstantPad2d((1,1,2,0),0.33)
m1(input_2d)


#%%
m2(input_2d)


#%%



