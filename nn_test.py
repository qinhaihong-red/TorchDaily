
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
net=nn.Sequential(nn.Linear(2,2),nn.Linear(2,2),nn.Conv2d(3,4,3))
print(net,'\n\n')
def init(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
    elif (type(m) == nn.Conv2d):
        m.bias.data.fill_(0.1)

net.apply(init)##使用apply进行初始化


#%%
for c in net.children():
    print(c)


#%%
for i,c in net.named_children():
    print(i,c)##默认加了一个编号


#%%
print(net.named_parameters)


#%%
for i,m in enumerate(net.modules()):
    print(i,'-->',m)


#%%
for i,m in net.named_modules():
    print(i,m)##与上面的结果差不多
    


#%%
for i,m in net.named_parameters():
    print(i,m)##逐层展示每层的参数：weight+bias
    


#%%
net.state_dict().keys()


#%%
model=nn.Sequential(
    nn.Conv2d(3,4,3),
    nn.ReLU(),
    nn.Conv2d(4,2,1),
    nn.ReLU()
)

for i,m in model.named_modules():
    print(i,'-->',m)


#%%
import collections
layers=collections.OrderedDict()
layers['第一层：conv2d_1']= nn.Conv2d(3,4,3)
layers['relu_1']=nn.ReLU()
layers['conv2d_2']=nn.Conv2d(4,2,1)
layers['relu_2']=nn.ReLU()
model=nn.Sequential(layers
)

for i,m in model.named_modules():
    print(i,'-->',m)


#%%
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.choices=nn.ModuleDict({
            'conv':nn.Conv2d(1,4,3,padding=1),
            'pool':nn.MaxPool2d(2,stride=1)
            ##'pool':nn.AdaptiveMaxPool2d(3)
        })
        self.act=nn.ModuleList([nn.ReLU(),nn.PReLU()])
        self.params=nn.ParameterDict({
            'param1':nn.Parameter(torch.randn(1,2)),
            'param2':nn.Parameter(torch.randn(2,2))
        })
        
    def forward(self,x,choice):
        x=self.choices[choice](x)
        return x


#%%
input=torch.arange(16,dtype=torch.float).reshape(1,1,4,4)
input


#%%
net=MyModule()
net(input,'conv')


#%%
net(input,'pool')


#%%
for n,p in net.named_parameters():
    print(n,'-->',p)


#%%
for p in net.parameters():
    print(p)

#%% [markdown]
# DFT变换对：
# 
# $f(a,b)=\frac{1}{N^2}\sum_{x=0}^{N-1} \sum_{y=0}^{N-1} F(x,y)e^{\frac{j2\pi}{N}(ax+by)}$
# 
# $F(x,y)= \sum_{a=0}^{N-1} \sum_{b=0}^{N-1}f(a,b)e^{\frac{-j2\pi}{N}(ax+by)}$
#%% [markdown]
# -  成谐波关系的复指数序列两两正交，即：
#     
#     复指数序列 $\{e^{jkw_{0}t}\}，对于\int_{t}^{t+T_{0}}e^{jmw_{0}t}{{e^\ast}^{jnw_{0}t}}dt = \int_{t}^{t+T_{0}}e^{jmw_{0}t}{e^{-jnw_{0}t}}dt=\int_{t}^{t+T_{0}}e^{j(m-n)w_{0}t}dt$
# 
#     若m=n,则上式为1，否则为0. 因此复指数序列构成了频域上的规范正交基.
#     
#          
# -  能量有限（收敛条件）的周期或非周期信号，可以表示为成谐波关系的复指数序列的线性组合或者积分.
#     
#     从内积空间和正交性来理解，傅里叶变换就是计算时域信号投影到频域（由谐波序列正交基构成）上的系数.
#     
#     
# - 考察DFT变换对：
# 
#     $f(a,b)=\frac{1}{N^2}\sum_{x=0}^{N-1} \sum_{y=0}^{N-1} F(x,y)e^{\frac{j2\pi}{N}(ax+by)}$：表示时域信号是频域上的线性组合
#     
#     $F(x,y)= \sum_{a=0}^{N-1} \sum_{b=0}^{N-1}f(a,b)e^{\frac{-j2\pi}{N}(ax+by)}$ ：计算时域信号投影到频域上的系数
#     
# 回想计算空间向量正交投影到子空间系数的方法：空间向量与子空间中基向量做内积。
# 
# 
# 根据复域上函数内积的定义：$<x(t),y(t)> = \int_{a}^{b}x(t)y^\ast(t)dt$
# 
# $e^{\frac{-j2\pi}{N}(ax+by)}$ 表示 $e^{\frac{j2\pi}{N}(ax+by)}$ 的共轭，因此$F(x,y)$就是$f(a,b)$在频域 $e^{\frac{j2\pi}{N}(ax+by)}$ 上的系数.

#%%
np.log(2)


#%%
np.log(2.72)


#%%
np.log(np.exp2(32))


#%%
np.exp2(32)


#%%
from skimage import io 


#%%
img_path='C:\\Users\\qinhaihong\\Pictures\\sin3.jpg'
img=io.imread(img_path,as_gray=True)


#%%
io.imshow(img);


#%%
img_sub=img[:,:,0]


#%%
plt.imshow(img_sub,cmap='gray');


#%%
dft = np.fft.fft2(img_sub)


#%%
(dft>0).sum()


#%%
dft[1,0]


#%%
m=np.zeros((64,64),dtype=np.complex)
m[0,4]=250
idft=np.fft.ifft2(m)
m2=(idft.real*255).astype(np.uint8)
plt.imshow(m2,cmap='gray');


#%%
idft.real[0:10,0:10]


#%%
m3==15


#%%



#%%
np.square(np.abs(np.complex(1,2)))


#%%
np.pi


#%%
np.angle(np.complex(1.732,1))*(180/np.pi)


#%%
import cv2
import os


#%%
img_path=os.getenv('IMG_FOLDER')


#%%
img_path=os.path.join(img_path,'sin3.jpg')


#%%
sin_image=cv2.imread(img_path,0)
plt.imshow(sin_image,cmap='gray')


#%%
sin_image.shape


#%%
sin_dft=np.fft.fft2(sin_image)
sin_dft_mag= (np.log(np.abs(sin_dft)+1)).astype(np.uint8)
plt.imshow(sin_dft_mag,cmap='gray');


#%%
sin_dft.dtype


#%%
sin_idft=np.fft.ifft2(sin_dft)
sin_idft_mag= ((np.abs(sin_idft))).astype(np.uint8)
plt.imshow(sin_idft_mag,cmap='gray');


#%%
sin_dft


#%%
sin_idft


#%%
def imshow_gray(img):
    plt.imshow(img,cmap='gray');


#%%
from cv_helper import get_img_path


#%%
lena=get_img_path('lena')
img_lena=cv2.imread(lena,0)
plt.imshow(img_lena,cmap='gray');


#%%
lena_dft=np.fft.fft2(img_lena)


#%%
lena_mag=np.log(np.abs(lena_dft)+1)


#%%
lena_mag


#%%
imshow_gray(lena_mag)


#%%
lena_mag_shift=np.fft.fftshift(lena_mag)
imshow_gray(lena_mag_shift)


#%%
3/np.sqrt(30)


#%%
out=None
out=cv2.normalize(np.abs(lena_dft),out,0,1,cv2.NORM_MINMAX)
out_uint8=(out*255).astype(np.uint8)
imshow_gray(np.fft.fftshift(out_uint8));


#%%
np.array(data)


#%%
ilena=np.fft.ifft2(lena_dft)
lena2=np.abs(ilena)
imshow_gray(lena2)


#%%
ilena.dtype


#%%
##with cv2 
lena32f=img_lena.astype(np.float32)
cv_dft_real = cv2.dft(lena32f,flags=cv2.DFT_REAL_OUTPUT)
cv_dft_complex=cv2.dft(lena32f,flags=cv2.DFT_COMPLEX_OUTPUT)


#%%
cv_dft_real.shape


#%%
cv_dft_complex.shape


#%%
cv_dft_real[0:3,0:3]


#%%
np.log(np.abs(cv_dft_real)+1)


#%%
cv_dft_complex[0:3,0:3,0]


#%%
cv_dft_complex.shape


#%%
cv_dft_mag=np.log(np.sqrt(np.square(cv_dft_complex[:,:,0])+np.square(cv_dft_complex[:,:,1]))+1)


#%%
cv_dft_mag.shape


#%%
cv_dft_mag


#%%
imshow_gray(cv_dft_mag)


#%%
cv_idft=cv2.idft(cv_dft)


#%%
out=cv2.normalize(cv_idft,None,1,0,cv2.NORM_MINMAX)
imshow_gray(out)


#%%
cv_idft


#%%
cv_idft_complex=cv2.idft(cv_dft,flags=cv2.DFT_COMPLEX_OUTPUT)
cv_idft_complex.shape


#%%
cv_idft_complex


#%%
np.abs(ilena[0,0])


#%%
out[0,0]*255


#%%
np.fft.fftshift


