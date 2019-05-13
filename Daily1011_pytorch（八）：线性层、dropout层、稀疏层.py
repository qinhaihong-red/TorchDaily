
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 1.线性层
# 
# ## 1.1 Linear
# 
# 源码见：https://pytorch.org/docs/0.4.0/_modules/torch/nn/modules/linear.html#Linear
# 
# 
# 
# __要点：__
# 
# - 继承自Module
# 
# - 4个核心属性：in_features, out_features, weight, bias

#%%
x=torch.randn(128,20)
m=nn.Linear(20,10)##input_features,output_features,bias=True
y=m(x)
y.shape


#%%
##注意它有两个类型为parameter的参数：一个是weight，一个是bias
print(type(m.weight),'\n',type(m.bias))


#%%
m.weight.shape


#%%
m.bias.data


#%%
##初始化
m2=nn.Linear(5,3)
def init(m):
    m.bias.data.fill_(1.1)
    m.weight.data.fill_(0.5)
m2.apply(init)
m2.weight.data


#%%
for n,i in m2.named_parameters():
    print(n,'-->',i)


#%%
##>1.2 Bilinear
m=nn.Bilinear(2,3,4)
input1=torch.arange(8,dtype=torch.float).reshape(4,2)
input2=torch.arange(12,dtype=torch.float).reshape(4,3)
def init2(m):
    m.weight.data.fill_(1.0)
    m.bias.data.fill_(2.0)

m.apply(init2)
m.weight.data.shape


#%%
out=m(input1,input2)
out.shape

#%% [markdown]
# # 2. dropout层
# 
# 注意该层需要区分是否在training mode 与 evaluation mode.
# 
# 源码见：https://pytorch.org/docs/0.4.0/_modules/torch/nn/modules/dropout.html#Dropout
# 
# 论文参考：https://arxiv.org/abs/1207.0580
# 
# 
# 
# 
# ## 2.1 Dropout
#%% [markdown]
# 在训练过程中，随机把输入张量的一些元素以p概率置0，可以：
# - 有效正则化
# - 防止神经元的共适应
# 
# 
# 输入数据size任意.
# 
# p默认为0.5

#%%
m=nn.Dropout()
x=torch.randn(4,3)
y=torch.arange(12,dtype=torch.float).reshape(1,2,6)
y


#%%
m(y)

#%% [markdown]
# __除了置0以外，非0输出会以$\frac{1}{1-p}$的比例进行缩放__
#%% [markdown]
# ## 2.2 Dropout2d
# 
# 以概率p，置某个channel的所有数据为0.
# 
# 输入数据的size为(N,C,H,W).

#%%
m=nn.Dropout2d()
x=torch.arange(32,dtype=torch.float).reshape(1,2,4,4)
m(x)

#%% [markdown]
# __dropout2d的置0与缩放，针对的是整个channel__

#%%
##>2.3 alphadropout


#%%
#>三 稀疏层
##>3.1 Embedding


#%%
##>3.2 EmbeddingGag


#%%



