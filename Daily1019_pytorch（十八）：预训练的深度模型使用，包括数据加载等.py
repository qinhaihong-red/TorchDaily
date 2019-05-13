
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tf
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# #  0. 模型使用简单示例 torchvision.models
#%% [markdown]
# ### 0.1 标准化参数：所有预训练模型使用如下的标准化参数，按通道分，各通道的参数不同

#%%
normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

#%% [markdown]
# ### 0.2 数据加载：使用ImageFolder，注意目录布局应符合要求

#%%
train_loader = torch.utils.data.DataLoader(
        tv.datasets.ImageFolder('./data/images/', tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            normalize,
        ])),
        batch_size=2, shuffle=True)

#%% [markdown]
# ### 0.3 定义模型、误差函数、优化器
# 
# 对于模型应修改全连接层（默认输出1000各分类），使之符合实际需求.
# 
# 这里仅供展示，没有做出修改.

#%%
model=tv.models.vgg16(pretrained=True)
E=nn.CrossEntropyLoss()
opt=torch.optim.SGD(model.parameters(),0.1,momentum=0.9,weight_decay=1e-4)

#%% [markdown]
# ### 0.4 定义训练和验证函数进行迭代训练与验证
# 
# 注意应该是双循环，外部是对epoch进行迭代，内部是对mini-batch进行迭代

#%%
def train(model,train_loader,E,opt):
    model.train()
    for i ,(input,target) in enumerate(train_loader):
        print(i,'--',input.shape,'--',target)
        
        output=model(input)
        print(output.shape)
        loss=E(output,target)
        
        
        opt.zero_grad()
        loss.backward()
        opt.step()


#%%
train(model,train_loader,E,opt)

#%% [markdown]
# # 1.模型
# 
# 所有模型期望的输入都相同，即满足如下条件：
#  
# - 形状为（C,H,W）的三通道RGB小批图像，高和宽至少为224
# 
# - 图像数据为[0,1]之间的浮点数
# 
# - 使用 mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] 进行标准化.
# 
# 
# 可通过使用组合变换实现上述要求.
# 
# 有些模型具有不同的训练和评估行为，比如批标准化. 通过使用train()和eval()方法，可以在这两种模式之间切换.

#%%



