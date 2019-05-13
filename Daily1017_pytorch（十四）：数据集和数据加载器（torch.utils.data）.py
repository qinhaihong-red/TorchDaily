
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
import torch.utils.data as data

#%% [markdown]
# ### 1. Dataset
# 
# 抽象类，供继承. 其子类主要供DataLoader使用.
# 需要实现：
# - \_\_getitem\_\_
# - \_\_len\_\_
#%% [markdown]
# ### 2. TensorDataset
# 
# Dataset子类.
# - 所有的数据第一维（dim=0）尺寸必须相等
# - 按索引取数据时，从第一维开始取，返回tuple

#%%
x=torch.randn(2,2)


#%%
y=torch.arange(10).reshape(2,-1)
d=data.TensorDataset(x,y)
print(x,'\n',y)


#%%
print(len(d),'\n',d[0])

#%% [markdown]
# ### 3.ConcatDataset
# 
# Dataset子类.

#%%
y2=torch.arange(10,14).reshape(1,-1)##ContcatDataset不要求各dataset第一维大小一致
d2=data.TensorDataset(y2)
cd=data.ConcatDataset([d,d2])


#%%
cd[0]


#%%
cd[1]


#%%
cd[2]


#%%
len(cd)

#%% [markdown]
# ### 4.Subset
# 
# Dataset子类.
# 
# 用于选取Dataset的子集

#%%
s=data.Subset(cd,[1,2])
s[0]


#%%

s[1]

#%% [markdown]
# ### 5.DataLoader
#%% [markdown]
# 组合Dataset、采样器、以及单线程或多线程的迭代器，实现对数据的弹性加载.
# 
# 主要用在数据的加载与处理方面.
# 
# 见：https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#%% [markdown]
# ### 6. random_split 数据切分
# 
# 按给定长度随机切分数据.
#%% [markdown]
# ### 7. 采样器基类 Sampler
# 每个子类需要提供 __iter__ 和__len__ 方法.
#%% [markdown]
# ### 8.各采样器子类
# 
# - RandomSampler:随机采样，无放回（without replacement）
# 
# 
# - SequentialSampler：顺序采样，总是保持相同顺序
# 
# 
# - SubsetRandomSampler：从给定的索引中随机采样，无放回
# 
# 
# - WeightedRandomSampler：带权重的随机抽样，默认有放回
# 
# 
# - BactchSampler：批采样
# 
# 
# - DistributedSampler：分布式采样

#%%
## BatchSampler+SequentialSampler
list(torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(10)),batch_size=3,drop_last=False))


#%%
list(torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(10)),batch_size=3,drop_last=True))


#%%



