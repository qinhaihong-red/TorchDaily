
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 视觉层：采样与插值

#%%
##>1. 像素混淆 pixelshuffle
##把一个形状为[∗,(r^2)*C,H,W]的张量，重排为[∗,C,r*H,r*W]
input=torch.arange(144).reshape(1,9,4,4)
ps=nn.PixelShuffle(3)##r=3
out=ps(input)
out.shape


#%%
##>2. upsample上采样

#%% [markdown]
# 上采样，也称为图像插值，一般用来放大图片尺寸。
# 这里的采样模式（mode）如下：
# - nearest neighbor，即最邻近：复制邻近值进行放大，适用3d,4d tensor
# - linear，bilinear，即线性采样和双线性采样：适用于3d，4d tensor
# - trilinear，三线性采样：适用于5d tensor
# 
# __注意:nn.Upsample类已经废置，适用nn.functional.interpolate进行代替__

#%%
##>2.1 Upsample
input=torch.arange(1,5).reshape(1,1,2,2).float()
input


#%%
m=nn.Upsample(scale_factor=2,mode='nearest')
m(input)


#%%
##使用functional.interpolate进行代替
import torch.nn.functional as F
F.interpolate(input,scale_factor=2,mode='nearest')


#%%
F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=False)


#%%
F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=True)


#%%
input_3=torch.zeros(3,3).reshape(1,1,3,3).float()
input_3[:,:,:2,:2].copy_(input)
input_3


#%%
F.interpolate(input_3,scale_factor=2,mode='bilinear',align_corners=False)


#%%
F.interpolate(input_3,scale_factor=2,mode='bilinear',align_corners=True)


#%%
##>2.2 其他的 UpsamplingNearest2d 和 UpsamplingBilinear2d，都可以用functional.interpolate代替


