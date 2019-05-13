
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
# # 1. 通用变换 Generic Transform
#%% [markdown]
# ### 1.1 tf.Lambda
#%% [markdown]
# 使用一个自定义的lambda或函数作为变换.

#%%
img=Image.open('./data/images/dog/L3.png')
img


#%%
lab=tf.Lambda(lambda x:tf.CenterCrop(size=100)(x))
lab(img)

#%% [markdown]
# # 2. 函数变换 Functional Transforms
# 
# 以下均为函数，不是具备\_\_call\_\_属性的class，直接调用.
# 
# torchvision.transforms里面的类，基本上是对Functional Transforms的封装，就像torch.nn里面的类，封装调用了torch.nn.functional一样.
#%% [markdown]
# ## 2.1 亮度调节 tf.functional.adjust_brightness

#%%
tf.functional.adjust_brightness(img,3)

#%% [markdown]
# ## 2.2 对比度调节 tf.functional.adjust_contrast

#%%
tf.functional.adjust_contrast(img,3)

#%% [markdown]
# ## 2.3 Gamma调节 tf.functional.adjust_gamma
# 
# $ I_{out} = 255 \times gain \times(\frac{I_{in}}{255})^{\gamma}$

#%%
tf.functional.adjust_gamma(img,2)

#%% [markdown]
# ## 2.4 色调调节 tf.functional.adjust_hue

#%%
tf.functional.adjust_hue(img,0.3)

#%% [markdown]
# ## 2.5 饱和度调节 tf.functional.adjust_saturation

#%%
tf.functional.adjust_saturation(img,2)

#%% [markdown]
# ## 2.6 仿射变换 tf.functional.affine
#%% [markdown]
# 对图像施加仿射变换且保持中心不变.
#%% [markdown]
# ## 2.7 图像裁剪 tf.functional.crop
#%% [markdown]
# ## 2.8 四角加中心裁剪 tf.functional.five_crop
#%% [markdown]
# ## 2.9 水平反转 tf.functional.hflip
#%% [markdown]
# ## 2.10 标准化 tf.functional.normalize
#%% [markdown]
# ## 2.11 填充 tf.functional.pad
#%% [markdown]
# ## 2.12 缩放 tf.functional.resize
#%% [markdown]
# ## 2.13 裁剪后缩放 tf.functional.resized_crop
#%% [markdown]
# ## 2.14 旋转 tf.functional.rotate
#%% [markdown]
# ## 2.15 正反5次裁剪 tf.functional.ten_crop
#%% [markdown]
# ## 2.16 对图像灰阶化 tf.functional.to_grayscale
#%% [markdown]
# ## 2.17 把tensor或ndarray转为PIL tf.functional.to_pil_image
#%% [markdown]
# ## 2.18 把pil或ndarray转为tensor tf.functional.to_tensor
#%% [markdown]
# ## 2.19 垂直反转 tf.functional.vflip
#%% [markdown]
# # 3 工具库 torchvision.utils

#%%
import scipy.misc
##看下scipy中自带的图像
face=scipy.misc.face()
type(face)


#%%
face.shape


#%%
face_img=tf.ToPILImage()(face)
face_img


#%%
face_img.size


#%%
plt.imshow(face);

#%% [markdown]
# ## 3.1 制作图像网格 make_grid

#%%

tensor_img=tf.ToTensor()(img)##这个图片是L3.png，不是上面的face
tensor_img.shape


#%%
img_list=[tensor_img,tensor_img,tensor_img]
img_grid=tv.utils.make_grid(img_list,padding=100)
type(img_grid)


#%%
img_grid.shape


#%%
def show_img_grid(img_grid):
    img_np=img_grid.numpy()
    img_np=np.transpose(img_np,(1,2,0))##把CHW转为HWC
    plt.imshow(img_np,interpolation='nearest');


#%%
show_img_grid(img_grid)


#%%
show_img_grid(tv.utils.make_grid(img_list,padding=100,normalize=True))


#%%
show_img_grid(tv.utils.make_grid(img_list,padding=100,normalize=True,range=(0,0.5)))


#%%
show_img_grid(tv.utils.make_grid(img_list,padding=100,normalize=True,range=(0,0.5),scale_each=True))

#%% [markdown]
# ## 3.2 把图像tensor数据保存为图像格式 save_image
# 
# 会根据给定的后缀名，自动保存为相应的图片格式.

#%%
tv.utils.save_image(tensor_img,'./data/images/dog/dog_tensor.jpg')


#%%



