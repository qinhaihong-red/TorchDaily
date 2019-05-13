
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1.torchvision.datasets
# 
# - 所有datasets都是torch.utils.data.Dataset的子集，因此都有\_\_getitem\_\_和\_\_len\_\_方法的实现.
# 
# 
# - 可以传给torch.utils.DataLoader使用并行方法加载数据.
# 
# 
# - 所有datasets都有相似的API接口. 它们都有两个共同的参数：transfrom和target_transfrom，分别用作对输入和目标进行变换.
#%% [markdown]
# ### 1.1 MNIST

#%%
mnist=tv.datasets.MNIST(root='./data/mnist/',download=True)

##mnist是Dataset的子类，可以直接进行索引
##>第一个下标是图像，这里直接显示了
mnist[0][0]


#%%
##>第二个下标是类标
mnist[0][1]


#%%
##查看mnist训练集的长度
len(mnist)


#%%
##再看第二个
mnist[1][1]


#%%
mnist[1][0]


#%%
type(mnist[1][0])


#%%
mnist[1][0].size


#%%
mnist[1][0].show()


#%%
pix=np.array(mnist[1][0])##转换为numpy
pix.shape


#%%
plt.imshow(pix,cmap='gray');

#%% [markdown]
# ### 1.2 Fashion-MNIST

#%%
fm=tv.datasets.FashionMNIST(root='./data/fmnist/')
len(fm)


#%%
fm[0][0]


#%%
fm[0][1]

#%% [markdown]
# ### 1.3 CIFAR10

#%%
cf=tv.datasets.CIFAR10(root='./data/cifar10/')
len(cf)


#%%
cf[0][0]


#%%
cf[0][1]


#%%
pix=np.array(cf[0][0])
pix.shape


#%%
pix[0,0,0]

#%% [markdown]
# ### 1.4 加载本地图像 ImageFolder
# 
# 使用ImageFolder加载本地图像，图像在本地路径上的排布方式如下：
# 
# 
# > root/dog/xxx.png
# 
# > root/dog/xxy.png
# 
# > root/dog/xxz.png
# 
# 
# > root/cat/123.png
# 
# > root/cat/nsdf3.png
# 
# > root/cat/asd932_.png
# 

#%%
s=tv.datasets.ImageFolder(root='./data/images/')
len(s)


#%%
s[0][0]


#%%
s[1][0]

#%% [markdown]
# ### 1.5 加载本地数据文件 DatasetFolder
# 
# 本地文件的存放排布要求同ImageFolder.
# 
# 
# 可以指定文件的后缀.
#%% [markdown]
# ### 1.6 其他数据集：
# 
# - Imagenet-12
# 
# - EMNIST
# 
# - COCO
# 
# - Captions
# 
# - DETECTION
# 
# - LSUN
# 
# - STL10
# 
# - SVHN
# 
# - PhotoTour
# 
# 
# 
# 具体见torchvision文档:https://pytorch.org/docs/stable/torchvision/datasets.html.

#%%



#%%



