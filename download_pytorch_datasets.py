
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tf
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
fmnist=tv.datasets.FashionMNIST('./data/fnist/',download=True)


#%%
cifar=tv.datasets.CIFAR10('./data/cifar10/',download=True)


#%%
cifar[0][0].show()


#%%
model_alex=tv.models.alexnet(pretrained=True)


#%%
model_resnet18=tv.models.resnet18(pretrained=True)


#%%
model_squeezenet=tv.models.squeezenet1_0(pretrained=True)


#%%
model_densenet=tv.models.densenet161(pretrained=True)


#%%
model_inceptionv3=tv.models.inception_v3(pretrained=True)


#%%



