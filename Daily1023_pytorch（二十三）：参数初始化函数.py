
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tf
import torchvision.models as mod
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # torch.nn.init
#%% [markdown]
# ### torch.nn.init.calculate_gain(nonlinearity, param=None) : 为给定的非线性函数，返回推荐的增加值. 非线性函数可以是：
# - Linear/Identity	1
# - Conv{1,2,3}D      1
# - Sigmoid           1
# - Tanh             5/3
# - ReLU             sqrt(2)
# 
# 
# 第一列是非线性函数名，第二列是增加的值.

#%%
gain=nn.init.calculate_gain('relu')##注意参数小写
gain

#%% [markdown]
# ### torch.nn.init.uniform\_(tensor, a=0, b=1)：以均匀分布初始化输入张量.

#%%
X=torch.empty(2,3)
nn.init.uniform_(X)
X

#%% [markdown]
# ### torch.nn.init.normal\_(tensor, mean=0, std=1)：以正太分布初始化输入张量.

#%%
nn.init.normal_(X)

#%% [markdown]
# ### torch.nn.init.constant\_(tensor, val): 以常量初始化输入张量.

#%%
nn.init.constant_(X,2)
X

#%% [markdown]
# ### torch.nn.init.eye\_(tensor):以恒等矩阵初始化输入的二维张量. 在线性层保持输入的恒等性，在该层使得尽量多的输入保持恒等.

#%%
#即使非方阵，也可调用
nn.init.eye_(X)
X


#%%
X=X.reshape(3,2)
nn.init.eye_(X)
X

#%% [markdown]
# ### torch.nn.init.dirac_(tensor):以 Dirac-delta 函数对3、4、5维张量进行初始化. 保持在卷积层的恒等性，在该层使得尽量多的输入保持恒等.

#%%
X=torch.empty(2,3,4)
nn.init.dirac_(X)
X

#%% [markdown]
# ### torch.nn.init.xavier\_uniform\_(tensor, gain=1): 根据 “Understanding the difficulty of training deep feedforward neural networks”中描述的方法，使用均匀分布U(-a,a)为输入张量进行初始化.gain值是上面介绍的calculate_gain函数，是对a的缩放系数.

#%%
X=torch.empty(2,3)
nn.init.xavier_uniform_(X,gain=nn.init.calculate_gain('relu'))
X

#%% [markdown]
# ### torch.nn.init.xavier\_normal\_(tensor, gain=1)：根据如上描述的方法，使用正太分布N(0,std)为输入变量进行初始化. gain值如上述，是std的缩放系数.

#%%
nn.init.xavier_normal_(X,gain=nn.init.calculate_gain('sigmoid'))
X

#%% [markdown]
# ### torch.nn.init.kaiming\_uniform\_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'): 使用均匀分布U(-bound,bound)为输入张量进行初始化，系数影响bound值.
# 见文档： https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_uniform_

#%%
nn.init.kaiming_uniform_(X)
X

#%% [markdown]
# ### torch.nn.init.kaiming\_normal\_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')：使用正太分布为输入张量做初始化. 
# 
# 见文档：https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_normal_

#%%
nn.init.kaiming_normal_(X)
X

#%% [markdown]
# ### torch.nn.init.orthogonal\_(tensor, gain=1): 使用半正交矩阵为输入张量进行初始化.
# 
# 见:https://pytorch.org/docs/stable/nn.html#torch.nn.init.orthogonal_

#%%
nn.init.orthogonal_(X)
X

#%% [markdown]
# ### torch.nn.init.sparse\_(tensor, sparsity, std=0.01): 把输入的二维张量初始化为稀疏矩阵，非0元素将从N(0,std)中采样.
# 
# 见：https://pytorch.org/docs/stable/nn.html#torch.nn.init.sparse_

#%%
nn.init.sparse_(X,sparsity=0.5)
X


#%%
##########test#############
X=torch.randn(2,3,2,2,dtype=torch.float)
f=nn.Dropout2d()
f(X)


#%%
f(X)


#%%
##推断时，模型的权值是最终训练模型的1/2. 这是所谓的 权重比例推断原则.(weight scaling inference rule).
f.eval()
f(X)


#%%
f=nn.BatchNorm2d


#%%



