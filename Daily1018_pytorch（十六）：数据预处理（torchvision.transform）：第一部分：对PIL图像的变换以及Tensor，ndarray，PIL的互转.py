
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
# ## 1.图像变换组合器 transforms.compose 
# 
# 使用列表将不同的图像变换组合在一起，传入compose，形成组合对象.
#%% [markdown]
# ## 2.对PIL图像进行变换 Transforms on PIL Image
#%% [markdown]
# ### 2.1 中心裁剪 CenterCrop

#%%
img=Image.open('./data/images/dog/L3.png')
img.size


#%%
img


#%%
cc=tf.CenterCrop(size=100)
cc(img)

#%% [markdown]
# ### 2.2 随机改变图像的亮度、对比度和饱和度 tf.ColorJitter
#%% [markdown]
# 可以传入相关指标的参数，将影响随机改变的程度.
# 
# 详见文档.

#%%
cj=tf.ColorJitter(0.5,0.4,0.3,0.2)
cj(img)

#%% [markdown]
# ### 2.3 四角加中心裁剪 tf.FiveCrop
# 
# 参数为相应裁剪处尺寸.

#%%
fc=tf.FiveCrop(50)
img5=fc(img)


#%%
img5[0]


#%%
img5[1]


#%%
img5[2]


#%%
img5[3]


#%%
img5[4]

#%% [markdown]
# ### 2.4 对图像进行灰阶处理 tf.GrayScale
# 
# 参数为输出的通道数.

#%%
gs3=tf.Grayscale(3)
img3=gs3(img)
img3


#%%
pix3=np.array(img3)
pix3.shape


#%%
##选择3通道的灰阶输出，通道值r=g=b
pix3[0,0,0]


#%%
pix3[0,0,1]


#%%
pix3[0,0,2]


#%%
pix3[100,0,0]


#%%
pix3[100,0,1]


#%%
pix3[100,0,2]


#%%
gs1=tf.Grayscale(1)
img1=gs1(img)
img1


#%%
pix1=np.array(img1)


#%%
pix1.shape


#%%
pix1[0,0]


#%%
pix1[100,0]

#%% [markdown]
# ### 2.5 线性变换 LinearTransformation
#%% [markdown]
# 输入一个方阵，对图像做线性变换，比如白化（whitening）操作.
#%% [markdown]
# ### 2.6 填充 tf.Pad
# 
# 通过对图像指定边界填充大小、填充值、填充模式，对图像进行填充操作.
#%% [markdown]
# ### 2.7 随机仿射 tf.RandomAffine
# 
# 参数意义参看文档.

#%%
ra=tf.RandomAffine(20)
img_ra=ra(img)
img_ra

#%% [markdown]
# ### 2.8 以指定概率对图像施行指定变换 tf.RandomApply
# 
# 参数见文档.
#%% [markdown]
# ### 2.9 从给定的变换列表中随机选择一个对图像进行变换 tf.RandomChoice
#%% [markdown]
# ### 2.10 随机裁剪 tf.RandomCrop
# 
# 
# 可指定裁剪尺寸和填充等相关参数.

#%%
rc=tf.RandomCrop(50)
rc_img=rc(img)
rc_img

#%% [markdown]
# ### 2.11 随机灰阶化 RandomGrayscale
# 
# 以指定概率对图像进行随机灰阶化.

#%%
rg=tf.RandomGrayscale(0.8)
rg_img=rg(img)
rg_img

#%% [markdown]
# ### 2.12 随机水平反转 RandomHorizontalFlip

#%%
rhf=tf.RandomHorizontalFlip(0.7)
rhf_img=rhf(img)
rhf_img

#%% [markdown]
# ### 2.13 以给定的变换列表随机对图像进行变换 tf.RandomOrder
#%% [markdown]
# ### 2.14 随机缩放再裁剪tf.RandomResizedCrop

#%%
rrc=tf.RandomResizedCrop(100)
rrc_img=rrc(img)
rrc_img

#%% [markdown]
# ### 2.15 随机旋转 tf.RandomRotation

#%%
rr=tf.RandomRotation(40)
rr_img=rr(img)
rr_img

#%% [markdown]
# ### 2.16 随机垂直反转 tf.RandomVerticalFlip

#%%
rvf=tf.RandomVerticalFlip(0.8)
rvf_img=rvf(img)
rvf_img

#%% [markdown]
# ### 2.17 缩放 tf.Resize

#%%
import PIL
##   H , W . 高在前，宽在后.
rs1=tf.Resize((64,28),interpolation=PIL.Image.BILINEAR)
rs1_img=rs1(img)
rs1_img


#%%
##如果指定的尺寸是一个整数n，那么较小的边为n，较大的边为 (h/w)*n. 假设 h 是大边.

rs2=tf.Resize(76,interpolation=PIL.Image.BILINEAR)
rs2_img=rs2(img)
rs2_img

#%% [markdown]
# ### 2.18 正反四角加中心裁剪 tf.TenCrop
# 
# 反转前裁剪5幅，反转后再裁剪5幅.可选择水平或垂直反转.
#%% [markdown]
# ## 3. 对 torch.*Tensor 进行变换
#%% [markdown]
# ### 3.1 标准化 tf.Normalize
# 
# - 给定均值和标准差序列，其中序列顺序是通道的顺序，序列长度是通道数，对张量图像逐通道进行标准化.
# 
# 
# - 张量图像格式为(C,H,W)
# 
# 
# - 注意这个函数是in-place操作.直接对输入张量进行改变.
#%% [markdown]
# ## 4. 转换变换 Tensor，ndarray，PIL 之间的互转
# 
# 包括Tensor或ndarray转PIL:
# - CHW的Tensor转PIL
# - HWC的ndarray转PIL
# 
# 以及PIL或ndarray转Tensor:
# - 范围是[0,255]的PIL或ndarray(HWC)转范围[0.,1.0]的CHW的FloatTensor
#%% [markdown]
# ### 4.1 PIL/ndarray 转 Tensor ： tf.ToTensor
# 
# - 转之前:HWC, [0,255],  uint8
# 
# 
# - 转之后:CHW, [0.,1.],  float32

#%%
img


#%%
print(type(img),'\n',img.mode,'\n',img.size)


#%%
##0.从PIL转到ndarray
nd_img=np.array(img)
nd_img.shape ##这是 HWC 数据格式


#%%
##1.从PIL转到Tensor
tt=tf.ToTensor()
t1=tt(img)
t1.shape## 已转换为CHW 数据格式


#%%
t1.dtype## 使用float32类型


#%%
##2.从ndarray转到Tensor
##uint8-->float32
##HWC-->CHW
t2=tt(nd_img)
print('t2:dtype={}, shape={}\n nd_img:dtype={},shape={}'.format(t2.dtype,t2.shape,nd_img.dtype,nd_img.shape))


#%%
nd_img[0,0,0]


#%%
t2[0,0,0]

#%% [markdown]
# ### 4.2 Tensor或 ndarray 转到 PIL
# 

#%%
##1. Tensor-->PIL
torch.all(t2==t1)


#%%
##Tensor to PIL
tp=tf.ToPILImage('RGB')##如这里不指定mode，则会根据输入数据的类型进行推断而得出一个mode
img1=tp(t1)
img1


#%%
## ndarray to PIL
img2=tp(nd_img)
img2


#%%



