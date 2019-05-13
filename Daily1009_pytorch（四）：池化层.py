
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# - 池化层没有输入、输出通道的参数。就是说，池化前后通道不会发生变化。
# - 通常池化就是缩小图片的尺寸。
# - 池化层的stride默认情况下跟随kernel_size，不像卷积层那样是默认为1. 这点需要注意.

#%%
#>一 最大池化
##>1.1 一维池化
m=nn.MaxPool1d(3,stride=2,return_indices=True)##核为3
input=torch.randn(1,1,50)
out=m(input)
out[0]##由于return_indices为True，因此返回一个tuple：一个元素是池化后的值，第二个元素是最大池化的坐标


#%%
out[1]


#%%
input


#%%
##>1.2 二维池化
m=nn.MaxPool2d((3,2),stride=(2,1))
input=torch.randn(20,16,50,32)
out=m(input)
out.shape


#%%
##>1.3 三维池化
m=nn.MaxPool3d((3,2,2),stride=(2,1,2))
input=torch.randn(20,16,50,44,31)
out=m(input)
out.shape

#%% [markdown]
# - 逆池化相当于把缩小的图片尺寸恢复为原来的尺寸
# - 逆最大池化只能恢复最大的元素，其他的元素均为0

#%%
#>2. 逆最大池化
##>2.1 一维逆最大池化
pool=nn.MaxPool1d(2,stride=2,return_indices=True)
unpool=nn.MaxUnpool1d(2,stride=2)
input=torch.arange(1,9,dtype=torch.float).reshape(-1,8).unsqueeze(1)
input


#%%
out,indices=pool(input)
indices


#%%
unpool(out,indices)


#%%
input=torch.arange(1,10,dtype=torch.float).reshape(-1,9).unsqueeze(1)##注意input的size是9
out,indices=pool(input)


#%%
unpool(out,indices)##input的size是9，这里没有设置output_size，因此unpool后，大小与原先的并不一致


#%%
unpool(out,indices,output_size=input.size())##这里设置了output_size与input.size一样大，因此unpool后，尺寸与原先的一样


#%%
##>2.2 二维逆最大池化
pool=nn.MaxPool2d(2,stride=2,return_indices=True)
unpool=nn.MaxUnpool2d(2,stride=2)
input=torch.arange(1,17,dtype=torch.float).reshape(1,1,4,4)
input


#%%
out,indices=pool(input)
out


#%%
unpool(out,indices)##恢复为原先的形状，除最大元素外，其他元素为 0


#%%
#>3. 平均池化
##>3.1 二维平均池化
m=nn.AvgPool2d((3,2),stride=(2,1))
input=torch.randn(20,16,50,32)
out=m(input)
out.shape


#%%
#>4. LP范数池化
input=torch.arange(1,5,dtype=torch.float).reshape(1,1,4)
pool=nn.LPPool1d(2,2)##取2-范数，核为2，其他参数默认
out=pool(input)
out


#%%
#>5. 自适应最大池化

#%% [markdown]
# - 自适应最大池化只需要指定输出的目标尺寸即可，其他参数自动计算

#%%
##>5.1 一维自适应最大池化
input=torch.randn(1,64,8)
m=nn.AdaptiveMaxPool1d(5)#目标尺寸为5
out=m(input)
out.shape


#%%
##>5.2 二维自适应最大池化
input=torch.randn(1,34,8,9)
m=nn.AdaptiveMaxPool2d((5,7))#目标尺寸：H=5,W=7
out=m(input)
out.shape


#%%
m=nn.AdaptiveMaxPool2d(7)##目标输出为方阵
out=m(input)
out.shape


#%%
m=nn.AdaptiveAvgPool2d((None,7))##可以只指定一个维度的尺寸
out=m(input)
out.shape


#%%
#>6. 自适应平均池化
##>6.1 二维自适应平均池化
m=nn.AdaptiveAvgPool2d((5,7))
out=m(input)
out.shape


#%%
#############test###########
m=nn.Conv1d(4,4,2,stride=2)
input=torch.arange(16,dtype=torch.float).reshape(1,4,4)
input


#%%
out=m(input)
out.shape


#%%
out


#%%



