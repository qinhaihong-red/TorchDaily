
#%%
import numpy as np
import torch
import torch.nn as nn
import math

#%% [markdown]
# ## SPP 空间金字塔池化
# 
# - 未使用SPP，全连接层与最后卷积层直接对接，因此需要输入数据尺寸固定，使得经过固定的池化后形成固定的数据尺寸给全连接层
# 
# - 使用SPP，最后的卷积层先与SPP对接，SPP经过内部池化，统一了不同数据尺寸，再与全连接层进行对接

#%%
class SPPNet(nn.Module):
    def __init__(self,spp_structure=[8,4,1],num_class=2):
        super(SPPNet,self).__init__()
        self.num_class = num_class
        self.spp_structure=spp_structure
        self.features=nn.Sequential(
        nn.Conv2d(3,64,3,1,1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64,64*2,3,1,1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            
            nn.Conv2d(64*2,64*4,3,1,1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            
            nn.Conv2d(64*4,64*8,3,1,1),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            
            nn.Conv2d(64*8,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.classifier=nn.Sequential(
            nn.Linear(64*np.square(self.spp_structure).sum(),100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100,100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100,self.num_class)             
        )
        
    def init(self):
        pass
        
    def spp(self,x):
        H,W=x.shape[2],x.shape[3]
        batch_sz=x.shape[0]
        out=None
        for i,val in enumerate(self.spp_structure):
            kernel_h=math.ceil(H/val)
            kernel_w=math.ceil(W/val)
            padding_h=(kernel_h*val-H+1)//2
            padding_w=(kernel_w*val-W+1)//2

            pool=nn.MaxPool2d(kernel_size=(kernel_h,kernel_w),padding=(padding_h,padding_w))

            if i == 0:
                out=pool(x).reshape(batch_sz,-1)
            else:
                out=torch.cat((out,pool(x).reshape(batch_sz,-1)),dim=1)

        return out
        
    def forward(self,x):
        x=self.features(x)
        x=self.spp(x)
        x=self.classifier(x)

        return x
            


#%%
#device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=SPPNet()
data=torch.randn(3,3,520,256)
out=model(data)
out.shape


#%%
##########test##############
spp_structure=[8,4,1]
np.square(spp_structure)


#%%



