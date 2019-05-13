
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # 一 距离函数
# ## 1.1 余弦相似度

#%%
x1=torch.arange(1,9,dtype=torch.float).reshape(2,4)
x2=torch.arange(10,18,dtype=torch.float).reshape(2,4)
print(x1,'\n',x2)


#%%
m=nn.CosineSimilarity()
m(x1,x2)


#%%
numerator=torch.dot(x1[0,:],x2[0,:])


#%%
denominator=torch.sqrt(torch.sum(x1[0,:]**2))*torch.sqrt(torch.sum(x2[0,:]**2))


#%%
numerator/denominator

#%% [markdown]
# ## 1.2 空间距离

#%%
m=nn.PairwiseDistance()##默认为欧几里得距离，p=2
m(x1,x2)


#%%
type(x1.grad_fn)


#%%
x1.is_leaf


#%%
x1.requires_grad

#%% [markdown]
# # 二 损失函数
# ## 2.1 L1损失

#%%
y=torch.arange(8,dtype=torch.float).reshape(2,4)
t=y+1
t


#%%
loss_fn=nn.L1Loss()##默认使用elementwise_mean
loss=loss_fn(t,y)
loss##注意输出为一个标量


#%%
loss_fn=nn.L1Loss(reduction='sum')
loss=loss_fn(t,y)
loss##输出为一个标量

#%% [markdown]
# ## 2.2 均方误差MSELoss

#%%
loss_fn=nn.MSELoss()
t=y+2
t[1,1]=10
loss_fn(t,y)

#%% [markdown]
# __输出的总是标量，没有batch_size维__

#%%
target=torch.empty(3).random_(2)
target

#%% [markdown]
# ## 2.3 二分类交叉熵损失：BCELoss

#%%
E=nn.BCELoss()
m=nn.Sigmoid()
y=torch.tensor([ 0.8589,  1.0864, -1.0978],requires_grad=True)
target=torch.tensor([0., 0., 1.])
y


#%%
target


#%%
m(y)


#%%
loss=E(m(y),target)##对y进行激活之后，再算交叉熵
loss


#%%
loss.backward()


#%%
y.grad


#%%
##手动计算
r=-torch.log(m(y))
r2=-torch.log(1-m(y))
r3=r2[0]+r2[1]+r[2]
r3/3

#%% [markdown]
# ### 2.3.1 二分类交叉熵手动计算说明
# 
# $E(w)=-\sum_{n}^{N}(t_{n}\log y_{n}+(1-t_{n})\log (1-y_{n})) $
# 
# $\bar{E(w)}=\frac{E(w)}{N}$
# 
# - 对m(y)和1-m(y)分别取负对数，记为r,r2
# - 如果target[n]=1，则取r[n]，否则取r2[n]
# - 上述相加除N
#%% [markdown]
# ## 2.4 负对数似然损失NLLLoss
#%% [markdown]
# ### 2.4.1.1 一维示例

#%%
E=nn.NLLLoss()
m=nn.LogSoftmax(dim=1)##注意这里是LogSoftmax
y=torch.randn(3,5,dtype=torch.float,requires_grad=True)##N*C，3个样本，5个分类
target=torch.tensor([1,0,4])
y


#%%
m(y)


#%%
loss=E(m(y),target)


#%%
loss


#%%
loss.backward()


#%%
y.grad


#%%
##手动计算损失
r=m(y)[0,1]+m(y)[1,0]+m(y)[2,4]
-r/3

#%% [markdown]
# ### 2.4.1.2 多分类交叉熵手动计算说明
# $E(w)=-\sum_{n}^{N} \sum_{k}^{K} t_{nk}\log y_{nk}$
# 
# $\bar{E(w)}=\frac{E(w)}{N}$
# 
# 假设 target=torch.tensor([1,0,4])，因随机数每次生成可能不同：
# 
# - 第一个样本属于第1类，即$t_{1}=1而t_{k\ne1}=0$，因此只取下标为[0,1]的元素即可
# - 同理，第二、三个样本，只取[1,0]、[2,4]即可
# - 三者相加取负数除3即可
#%% [markdown]
# ## 2.4.2 二维示例

#%%
N,C=5,4 ##5个样本，4个分类
E=nn.NLLLoss()
X=torch.randn(N,16,10,10)##5个样本，16个通道，H*W=10x10
target=torch.empty(N,8,8).random_(0,C)
target


#%%
##注意卷积核的形状
m=nn.Conv2d(16,C,3)#输入16个通道，输出C=4个通道，核尺寸为3，则输出后尺寸:[(10-3)/1]+1=8
y=m(X)
y.shape


#%%
target=target.to(torch.long)
loss=E(y,target)
loss

#%% [markdown]
# ## 2.5 交叉熵损失CrossEntropyLoss
#%% [markdown]
# 这个损失函数相当于 LogSoftmax+NLLLoss 的混合，即对于输入数据，不必再进行LogSoftmax的变换，直接计算交叉熵损失即可。

#%%
E=nn.CrossEntropyLoss()
X=torch.randn(3,5)
target=torch.empty(3,dtype=torch.long).random_(5)
target


#%%
loss=E(X,target)
loss


#%%
X=torch.tensor([[-1.8861,  2.8642,  2.1634,  1.5570,  1.1928],
        [-0.2726, -0.3854,  0.7054, -1.1143,  0.3424],
        [-0.0046,  1.3266,  1.6912, -0.3976, -0.9664]])

target=torch.tensor([1,0,4])

E(X,target)

#%% [markdown]
# __在使用相同数据X的情况下，这个结果与2.4.1.1的输出结果一致__
#%% [markdown]
# ## 2.6 KL散度损失

#%%
import torch.nn.functional as F
E=nn.KLDivLoss()
log_prob1=F.log_softmax(torch.randn(5,10),dim=1)
prob2=F.softmax(torch.randn(5,10),dim=1)
E(log_prob1,prob2)

#%% [markdown]
# ## 2.7 带sigmoid激活的二分类交叉熵损失 BCEWithLogitsLoss
#%% [markdown]
# 与BCELoss相比，自动添加了sigmoid激活

#%%
E=nn.BCEWithLogitsLoss()
y=torch.tensor([ 0.8589,  1.0864, -1.0978])
target=torch.tensor([0., 0., 1.])
E(y,target)##注意变换的变量在前，标签在后，顺序不能乱


#%%
E2=nn.BCELoss()
m=nn.Sigmoid()
E2(m(y),target)

#%% [markdown]
# ## 2.8 其他损失
# smoothl1loss等，详见：https://pytorch.org/docs/stable/nn.html#smoothl1loss

#%%



