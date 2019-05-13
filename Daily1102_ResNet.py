
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # ResNet
# 
# ## 为什么需要更深的网络？
# 
# 
# 
# ## 更深的网络面临的挑战
# 
# - 问题1：梯度消失与梯度爆炸
# 
# - 解决1：对参数进行正则化的初始化以及对中间层的批正则化（Batch Normalization）
# 
# 
# - 问题2：学习退化的问题. 即随着网络深度的增加，准确率在下降. 这不是过拟合，否则随着复杂度的增加，准确率应该提高.
# 
# - 解决2：Resnet
# 
# 
# ## ResNet基本思想
# 
# 假设一个深层网络由两部分构成：一个浅层网络，加上一个至少不会使网络整体性能变差的网络.
# 
# 最简单的情况是后一部分的网络，以前面浅层网络的输出作为输入，并且输出保持不变，也就是引入一个恒等变换的网络.
# 
# 下面看看如何构造这样的网络：
# 
# 引入恒等映射x与F(x)，令 h(x)=F(x)+x 表示后一部分的网络，其中x是前面网络的输出.
# 
# 这样在极限情况下，F(x)趋近于0，h(x)=x是一个恒等变换，网络的整体性能不会变差. 因此有一个下限.
# 
# F(x)=h(x)-x，称之为残差.
# 
# 
# 由于残差存在下限，因此随着网络层次的加深，可能带来性能的提高，而不会变得更差（类似于拉格朗日对偶问题的解，构成原问题的下限）.

#%%
model=tv.models.resnet18(pretrained=True)
model


#%%
model._modules['layer4'][0]._modules['conv1'].weight.shape


#%%



