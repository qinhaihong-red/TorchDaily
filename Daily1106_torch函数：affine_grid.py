
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

#%% [markdown]
# ## 关于affine_grid的两点说明：
# 
# - 输入的$\theta$矩阵必须是Nx2x3的反向映射矩阵. 如果输入正向映射矩阵，它里面不会给你求逆.
# 
# 
# - 根据torch.Size参数确定变换后图像的HW，坐标的范围是[-1,1]，也就是说用[-1,-1]表示左上，[1,1]表示右下. 逆向变换后超出[-1,-1]范围的坐标，再grid_sample中使用padding模式填充.

#%%
input=torch.arange(16).reshape(1,1,4,4)
M=torch.tensor([[[1.,0.,1.],[0.,1.,0]]])

M.shape


#%%
input.shape


#%%
grid=F.affine_grid(M,input.size())
grid


#%%
grid.shape


#%%
N,C,H,W=input.shape
x=np.linspace(-1,1,H)
y=np.linspace(-1,1,W)

##借用np的meshgrid函数
x_mesh,y_mesh=np.meshgrid(x,y)
x=torch.from_numpy(x_mesh)
y=torch.from_numpy(y_mesh)


#%%
ones=x.new_ones(H*W)
##齐次坐标
grid_HC=torch.cat([x.reshape(1,-1),y.reshape(1,-1),ones.reshape(1,-1)],dim=0)
grid_HC.shape


#%%
grid_HC=grid_HC.reshape(1,3,16)
grid_TF=torch.bmm(M,grid_HC.float())
grid_TF=grid_TF.reshape(1,2,4,4)
grid_TF.shape


#%%
#交换坐标
grid_TF=grid_TF.permute((0,-1,1,2))
grid_TF.shape


#%%
grid_TF


#%%



