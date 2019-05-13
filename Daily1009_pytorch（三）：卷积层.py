
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
#>一 卷积层
##>1.1 一维卷积层
m=nn.Conv1d(16,33,3,stride=2)
input=torch.randn(20,16,50)
out=m(input)


#%%
out.shape


#%%
##>1.2 二维卷积层
input=torch.randn(1,3,5,5)
m=nn.Conv2d(3,2,3,stride=2,padding=1)##stride默认是1，padding默认是0
out=m(input)
out.shape


#%%
##>1.2.1 非正方的卷积核+不相等的步长+不等的步长+膨胀
##从kernel_size到dilation，参数既可以是int，又可以是tuple
m=nn.Conv2d(16,33,kernel_size=(3,5),stride=(2,1),padding=(4,2),dilation=(3,1))


#%%
input=torch.randn(20,16,50,100)##20个16通道的50*100(H*W)数据
out=m(input)
out.shape


#%%
##>1.3 三维卷积层
m=nn.Conv3d(16,33,(3,5,2),stride=(2,1,1),padding=(4,2,0))
input=torch.randn(20,16,10,50,100)##20个16通道的10x50x100(depth*height*width,DHW)数据
out=m(input)
out.shape


#%%
#>二 转置卷积
##>2.1 二维转置卷积
m=nn.ConvTranspose2d(16,33,(3,5),stride=(2,1),padding=(4,2))
input=torch.randn(20,16,50,100)
out=m(input)
out.shape


#%%
##>2.1.2 上下采样
input=torch.randn(1,16,12,12)
downsample=nn.Conv2d(16,16,3,stride=2,padding=1)##25x25
upsample=nn.ConvTranspose2d(16,16,3,stride=2,padding=1)
h=downsample(input)
h.shape


#%%
out=upsample(h,output_size=input.shape)##output_size参数用作推断output_padding，参考源码
out.shape


#%%
##>2.2 三维卷积
m=nn.Conv3d(16,33,(3,5,2),stride=(2,1,1),padding=(0,4,2))
input=torch.randn(20,16,10,50,100)##dhw:10x50x100
out=m(input)
out.shape


#%%
##############test#############################


