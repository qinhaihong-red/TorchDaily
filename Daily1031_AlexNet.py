
#%%
import numpy as np
import torch
import torch.nn as nn
import math
import cv2


#%%
class AlexNet(nn.Module):
    '''alexnet'''
    
    def __init__(self,init_weights=True,num_class=1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2),
            nn.Conv2d(64,192,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2),
            nn.Conv2d(192,384,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2)            
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,self.num_class)
        )
        
        if init_weights is True:
            _init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(n/2))
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
    
    def forward(self,x):
        out=self.features(x)
        out=out.reshape(out.shape[0],-1)
        out=self.classifier(out)
        return out


#%%
2/math.sqrt(168)


#%%
import cv2


#%%
cv2.__version__


#%%
svm=cv2.ml.SVM_create()

#%% [markdown]
# 非奇异方阵的逆阵：$A^{-1}$
# 
# 列满秩矩阵的逆阵：$L = (A^{T}A)^{-1}A^{T}$，又称为左逆，因为：$LA=I$
# 
# 行满秩矩阵的逆阵：$R = A^{T}(AA^{T})^{-1}$，又称为右逆，因为：$AR=I$
# 
# __Moore-Penrose条件：$A$是任意 $m \times n$ 矩阵，则$A$的逆阵$A^{\dagger}$定义为满足下列条件的矩阵__：
# 
# - $AA^{\dagger}A=A$
# 
# 
# - $A^{\dagger}AA^{\dagger}=A$
# 
# 
# - $AA^{\dagger}$为Hermitian矩阵，即：$AA^{\dagger}=(AA^{\dagger})^{H}$
# 
# 
# - $A^{\dagger}A$为Hermitian矩阵，即：$A^{\dagger}A=(A^{\dagger}A)^{H}$
# 
# 
# 非奇异方阵的逆阵、左右伪逆，都满足上述条件.
# 
# 对于秩缺矩阵$A$，可以通过SVD奇异值分解来求$A^{\dagger}$：
# 
# $A=U\Sigma V^{T}$，则：
# 
# $A^{\dagger}=V(\Sigma)^{-1}U^{T}$

#%%



