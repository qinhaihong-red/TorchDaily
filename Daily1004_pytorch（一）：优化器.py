#%% [markdown]
# ## pytorch的参数优化阶段：
# - 通过backward计算梯度
# - 选定合适的损失函数
# - 选定合适的梯度下降算法，对损失函数使用算法迭代计算梯度
# 
# ## 传统梯度下降的缺陷：
# - 缺乏动量，导致陷入极小点或者鞍点
# - 学习率固定，导致在最优点附近震荡或者迭代速度过慢

#%%
import numpy as np
import torch


#%%
x=np.arange(5)
t=torch.from_numpy(x)
t


#%%
t[0]=1


#%%
x


#%%
a=t.numpy()


#%%
type(a)


#%%
l=x.tolist()
l


#%%
l[1]=30
l


#%%
x

#%% [markdown]
# ### 一. 使用torch.optim.SGD
# 
# 求 $f(x,y)=-(cos^{2}x+cos^{2}y)^{2}$的极小值

#%%
import torch.optim
x = torch.tensor([np.pi/3,np.pi/6],requires_grad=True)
opt=torch.optim.SGD([x],lr=0.1,momentum=0)##先不使用动量


#%%
for i in range(11):
    if i:
        opt.zero_grad()
        f.backward()
        opt.step()
    
    f = -((x.cos()**2).sum())**2
    print('i {}: x={},f={}'.format(i,np.round(x.tolist(),3),f))

#%% [markdown]
# ## 二. 使用torch.optim.Adam
# 
# Adama优化函数在SGD的基础上，综合动量和自适应的学习率，避免出现SGD的两个问题
# 
# 求Himmelblau函数的极小值
# 
# 

#%%
def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2


#%%
##画出himemelblau的函数图像
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns;sns.set()


#%%
x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
X,Y=np.meshgrid(x,y)
X.shape


#%%
x.shape


#%%
x


#%%
X


#%%
X.shape


#%%
Z=himmelblau([X,Y])
Z.shape


#%%
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]');


#%%
init=[0,0]
x=torch.tensor(init,requires_grad=True,dtype=torch.float)
opt=torch.optim.Adam([x])
f=himmelblau(x)
print('with init-vals:{}'.format(init))
for i in range(20000):
    opt.zero_grad()
    f.backward()
    opt.step()
    
    f=himmelblau(x)
    if i % 1000 == 0:
        print('i:{},f={},x={},x.grad={}'.format(i,np.round(f.tolist(),5),np.round(x.tolist(),3),np.round(x.grad.tolist(),3)))


#%%
init=[-1,0]
x=torch.tensor(init,requires_grad=True,dtype=torch.float)
opt=torch.optim.Adam([x])
f=himmelblau(x)
print('with init-vals:{}'.format(init))
for i in range(20000):
    opt.zero_grad()
    f.backward()
    opt.step()
    
    f=himmelblau(x)
    if i % 1000 == 0:
        print('i:{},f={},x={},x.grad={}'.format(i,np.round(f.tolist(),5),np.round(x.tolist(),3),np.round(x.grad.tolist(),3)))


#%%
init=[-4,0]
x=torch.tensor(init,requires_grad=True,dtype=torch.float)
opt=torch.optim.Adam([x])
f=himmelblau(x)
print('with init-vals:{}'.format(init))
for i in range(20000):
    opt.zero_grad()
    f.backward()
    opt.step()
    
    f=himmelblau(x)
    if i % 1000 == 0:
        print('i:{},f={},x={},x.grad={}'.format(i,np.round(f.tolist(),5),np.round(x.tolist(),3),np.round(x.grad.tolist(),3)))


#%%
init=[4,0]
x=torch.tensor(init,requires_grad=True,dtype=torch.float)
opt=torch.optim.Adam([x])
f=himmelblau(x)
print('with init-vals:{}'.format(init))
for i in range(20000):
    opt.zero_grad()
    f.backward()
    opt.step()
    
    f=himmelblau(x)
    if i % 1000 == 0:
        print('i:{},f={},x={},x.grad={}'.format(i,np.round(f.tolist(),5),np.round(x.tolist(),3),np.round(x.grad.tolist(),3)))


#%%
from torch.autograd import Variable
init=[0,0]
x=torch.tensor(init,dtype=torch.float)
x=Variable(x,requires_grad=True)
opt=torch.optim.Adam([x])
f=himmelblau(x)
print('with init-vals:{}'.format(init))
for i in range(20000):
    opt.zero_grad()
    f.backward()
    opt.step()
    
    f=himmelblau(x)
    if i % 1000 == 0:
        print('i:{},f={},x={},x.grad={}'.format(i,np.round(f.tolist(),5),np.round(x.tolist(),3),np.round(x.grad.tolist(),3)))


#%%



