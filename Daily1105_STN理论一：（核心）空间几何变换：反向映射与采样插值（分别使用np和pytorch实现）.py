
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
import os
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# 参考：https://kevinzakka.github.io/2017/01/10/stn-part1/
# 
# 对图像的平移、旋转、拉伸等操作，统称为__空间几何变换__，它由两个阶段构成：
# 
# - 反向映射 
# 
# - 采样插值
# 
# 
# ## 一.反向映射
# 
# ### 1.使用仿射变换进行坐标变换
# 
# 假设仿射变换矩阵是：
# $T=\begin{bmatrix}
# 1&0&2\\
# 0&1&0
# \end{bmatrix}$
# 
# 坐标x=[1,1]的齐次坐标是：
# $x=\begin{bmatrix}
# 1\\
# 1\\
# 1
# \end{bmatrix}$
# 
# T对x的变换结果是:
# $\hat{x}=[3,1]$，实现了把坐标向右平移的效果.
# 
# 这是从原始图像角度考虑的变换，这种变换称之为前向映射（forward mapping）.
# 
# 实践中，通常使用的是反向映射（backward mapping），就是从变换后图像的角度来考虑：
# 
# 假设变换前坐标为（v,w），变换后为（x,y），则反向映射定义为：
# 
# __$p(x,y)=intp(T^{-1}(x,y))=intp(v,w)$__,  其中intp表示某种插值方式. 反向映射可以更方便的进行插值.
# 
# 这时$T$定义为方阵：
# $T=\begin{bmatrix}
# 1&0&2\\
# 0&1&0\\
# 0&0&1
# \end{bmatrix}$
# 
# 
# ### 2.反向映射的实现
# 先使用网格存储变换后图像的坐标，$grid[0,0]=\begin{bmatrix}
# 0\\
# 0
# \end{bmatrix}$，就是grid矩阵坐标(0,0)处，存储的内容是变换后图像的(0,0)坐标.
# 
# 然后通过反向映射，求出每个坐标对应的变换前的坐标，如：
# $grid[0,0]=T^{-1}*\begin{bmatrix}
# 0\\
# 0\\
# 1
# \end{bmatrix}$
# 
# 使用反向映射后的坐标，作为索引，对原图像进行采样，并使用相应的插值算法，得到该坐标处的像素值.
# 
# __grid[0,0]存储的内容，发生了3次变化.__
# 
# 
# 
# ## 2.双线性插值
# 
# 形式为$p(v,w)=\alpha v+\beta w+\gamma vw+\delta$ ，其中 $\alpha,\beta,\gamma,\delta $ 四个参数是需要通过插值计算得到的.
# 
# ### 权重的计算
# 
# 假设(x,y)四角处整数坐标为：
# 
# a:(x0,y0)，
# 
# b:(x1,y0)，
# 
# c:(x0,y1)，
# 
# d:(x1,y1).
# 
# 由于图像坐标是从左上角的(0,0)开始，因此坐标值(x,y)始终满足：x属于[x0,x1]，y属于[y0,y1]
# 
# 权重计算的技巧，就是始终用大的坐标值减去小的坐标值,即：
# 
# Wa=(x1-x)(y1-y)
# 
# Wb=(x-x0)(y1-y)
# 
# Wc=(x1-x)(y-y0)
# 
# Wd=(x-x0)(y-y0)

#%%
data_dir='./stn/data/'
size=(400,400)
batch_size=2


#%%
img1=Image.open(os.path.join(data_dir,'cat1.jpg'))
img2=Image.open(os.path.join(data_dir,'cat2.jpg'))
plt.imshow(img2);


#%%
type(img1)


#%%
##把图像裁剪为(400,400)，然后转为float类型的ndarray
##使用torchvision的变换实现比较容易

def swap_axis(x):
    assert isinstance(x,torch.Tensor)
    return x.permute((1,2,0))

def to_numpy(x):
    return x.numpy()

tf_compose=tv.transforms.Compose([tv.transforms.Resize(size),tv.transforms.ToTensor(),swap_axis,to_numpy])
np_img1=tf_compose(img1)
np_img2=tf_compose(img2)

##构造批数据
input_img=np.concatenate([np_img1,np_img2],axis=0).reshape(batch_size,size[0],size[1],-1)
input_img.shape


#%%
##构造变换矩阵
M=np.array([[1.,0.,-50.],[0.,1.,0.],[0,0,1]])
M_inv=np.linalg.inv(M)
##与批数据形式保持一致
M_inv=np.resize(M_inv,(batch_size,3,3))
M_inv


#%%
#反向映射演示：可以看到变换后[0,0]坐标对应的变换前坐标是是[50,0]
v=torch.tensor([0.,0.,1.])
T_inv=M_inv[0][:2,:]
np.matmul(T_inv,v)


#%%
##构造坐标网格
batch_size,H,W,C=input_img.shape
x=np.linspace(0,W-1,W)
y=np.linspace(0,H-1,H)
mesh_x,mesh_y=np.meshgrid(x,y)
ones=np.ones(np.prod(size))
grid=np.vstack([mesh_x.flatten(),mesh_y.flatten(),ones])


#%%
grid.shape


#%%
##与批数据保持一致
grid=np.resize(grid,(batch_size,grid.shape[0],grid.shape[1]))


#%%
grid.shape


#%%
##进行变换
grid_tf=np.matmul(M_inv,grid).reshape(2,3,size[0],size[1])
grid_tf=np.moveaxis(grid_tf,[0,1,2,3],[0,3,1,2])


#%%
grid_tf.shape


#%%
##使用torch的affine_grid和grid_sample实现相同的功能
##这里注意，对于平移的量，位于[0,2]之间，就是说如果平移一半，就是1 ，
## 0.5就是平移1/4.
T2=torch.tensor([[1.,0.,0.5],[0.,1.,0.]],dtype=torch.float)
T2=T2.expand(2,2,3)


#%%
grid_tf2=F.affine_grid(T2,torch.Size((2,3,400,400)))
image2_1=tv.transforms.functional.to_tensor(input_img[0])
image2_2=tv.transforms.functional.to_tensor(input_img[1])
image2=torch.stack((image2_1,image2_2),0)
out2=F.grid_sample(image2,grid_tf2)


#%%
out2_1=tv.transforms.functional.to_pil_image(out2[0])
out2_2=tv.transforms.functional.to_pil_image(out2[1])
out2_2


#%%
#第三个坐标是其次坐标，全部为1
grid_tf[:,:,:,2]


#%%
##分解出变换后x,y的坐标，其次坐标舍弃不用
x_tf=grid_tf[:,:,:,0]
y_tf=grid_tf[:,:,:,1]


#%%
x_tf.shape


#%%
##找到(x_tf,y_tf)的四角整数坐标
x0=np.floor(x_tf).astype(np.int64)
x1=x0+1
y0=np.floor(y_tf).astype(np.int64)
y1=y0+1


#%%
##整理4角坐标，使其符合标准
x0=np.clip(x0,0,W-1)
x1=np.clip(x1,0,W-1)
y0=np.clip(y0,0,H-1)
y1=np.clip(y1,0,H-1)


#%%
x1.shape


#%%
##通过索引，找到4角坐标对应的像素值
#inds=np.arange(batch_size).reshape(batch_size,None,None)
inds=np.arange(batch_size)
inds=inds[:,None,None]#None不可以reshape，但是可以进行索引


pixel_a=input_img[inds,y0,x0]
pixel_b=input_img[inds,y0,x1]
pixel_c=input_img[inds,y1,x0]
pixel_d=input_img[inds,y1,x1]


#%%
##计算权重
Wa=(x1-x_tf)*(y1-y_tf)
Wb=(x_tf-x0)*(y1-y_tf)
Wc=(x1-x_tf)*(y_tf-y0)
Wd=(x_tf-x0)*(y_tf-y0)


#%%
pixel_a.shape


#%%
Wa.shape


#%%
Wa=np.expand_dims(Wa,axis=3)
Wb=np.expand_dims(Wb,axis=3)
Wc=np.expand_dims(Wc,axis=3)
Wd=np.expand_dims(Wd,axis=3)


#%%
print(Wa.shape,' ',pixel_a.shape)
##前面的2是批，后面的3是通道数据


#%%
out=Wa*pixel_a+Wb*pixel_b+Wc*pixel_c+Wd*pixel_d


#%%
out.shape


#%%
plt.imshow(out[1]);


#%%
##把上述过程汇总为一个函数
##normalize表示原始坐标是否从[-1,1]采样，否则从[0,W]或[0,H]中采样.
def img_transform(M,input_img,normalize=False):
    batch_size,H,W,C=input_img.shape
    M=np.vstack((M,[0.,0.,1.]))
    M_inv=np.linalg.inv(M)
    M_inv=np.resize(M_inv,(batch_size,2,3))
    
    if normalize:
        x=np.linspace(-1,1,W)
        y=np.linspace(-1,1,H)
    else:
        x=np.linspace(0,W-1,W)
        y=np.linspace(0,H-1,H)
        
    mesh_x,mesh_y=np.meshgrid(x,y)
    ones=np.ones(np.prod(size))
    grid=np.vstack([mesh_x.flatten(),mesh_y.flatten(),ones])

    ##与批数据保持一致
    grid=np.resize(grid,(batch_size,3,H*W))

    ##进行变换
    grid_tf=np.matmul(M_inv,grid).reshape(2,2,H,W)
    grid_tf=np.moveaxis(grid_tf,1,-1)

    ##分解出变换后x,y的坐标
    x_tf=grid_tf[:,:,:,0:1].squeeze()
    y_tf=grid_tf[:,:,:,1:2].squeeze()
    
    ##使用normalize采样要恢复到原始范围
    if normalize:
        x_tf=((x_tf+1.)*W)*0.5
        y_tf=((y_tf+1.)*H)*0.5

    ##找到(x_tf,y_tf)的四角整数坐标
    x0=np.floor(x_tf).astype(np.int64)
    x1=x0+1
    y0=np.floor(y_tf).astype(np.int64)
    y1=y0+1

    ##整理4角坐标，使其符合标准
    x0=np.clip(x0,0,W-1)
    x1=np.clip(x1,0,W-1)
    y0=np.clip(y0,0,H-1)
    y1=np.clip(y1,0,H-1)

    ##通过索引，找到4角坐标对应的像素值
    #inds=np.arange(batch_size).reshape(batch_size,None,None)##这样不行
    inds=np.arange(batch_size)
    inds=inds[:,None,None]#None不可以作为reshape的参数，但是可以进行索引


    ##注意 H（对应y）在前，W (对应x)在后.
    pixel_a=input_img[inds,y0,x0]
    pixel_b=input_img[inds,y0,x1]
    pixel_c=input_img[inds,y1,x0] 
    pixel_d=input_img[inds,y1,x1]

    ##计算权重
    Wa=(x1-x_tf)*(y1-y_tf)
    Wb=(x_tf-x0)*(y1-y_tf)
    Wc=(x1-x_tf)*(y_tf-y0)
    Wd=(x_tf-x0)*(y_tf-y0)
    

    Wa=np.expand_dims(Wa,axis=3)
    Wb=np.expand_dims(Wb,axis=3)
    Wc=np.expand_dims(Wc,axis=3)
    Wd=np.expand_dims(Wd,axis=3)

    out=Wa*pixel_a+Wb*pixel_b+Wc*pixel_c+Wd*pixel_d

    return out

##使用pytorch的affine_grid和grid_sample实现空间几何变换
def img_transform_with_pytorch(M,input_img):
    batch_size,H,W,C=input_img.shape
    M=np.vstack((M,[0.,0.,1.]))
    M_inv=np.linalg.inv(M)
    M_inv=M_inv[:2,:]
    M_inv=np.resize(M_inv,(batch_size,2,3))
    M_inv = torch.tensor(M_inv,dtype=torch.float)
    
    grid=F.affine_grid(M_inv,torch.Size((batch_size,C,H,W)))
    ##这里把input_img转为tensor
    image2_1=tv.transforms.functional.to_tensor(input_img[0])
    image2_2=tv.transforms.functional.to_tensor(input_img[1])
    image2=torch.stack((image2_1,image2_2),0)
    out2=F.grid_sample(image2,grid)
    
    ##再把图像转为PIL
    out2_1=tv.transforms.functional.to_pil_image(out2[0])
    out2_2=tv.transforms.functional.to_pil_image(out2[1])
    return out2_2


#%%
##坐标向右平移155个像素
M=np.array([[1.,0,155],[0,1.,0.]])
out=img_transform(M,input_img)
plt.imshow(out[0]);


#%%
##使用torch内置的函数
##旋转90度
M=np.array([[0.01,1,0],[-1.,0.01,0.]])
out=img_transform_with_pytorch(M,input_img)
plt.imshow(out);


#%%
##缩放
M=np.array([[0.7,0,0],[0,0.7,0.]])
out=img_transform(M,input_img,True)
plt.imshow(out[0]);


