
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ### 各种非线性激活：
# - ReLU及其各种变体
# - sigmoid和tanh等
# 
# 以上都是逐元素进行计算

#%%
##>1. ELU激活
x=np.linspace(-5,5,1000)
x=torch.tensor(x)
m=torch.nn.ELU()
y=m(x)


#%%
import seaborn as sns;sns.set()
plt.title('ELU');
plt.plot(x.numpy(),y.numpy());

#%% [markdown]
# $ ELU(x)=max(0,x)+min(0,α∗(exp(x)−1)) $
# 
# 相比ReLU，在小于0的部分，变为 $ e^{x-1} $，即[-1,0]之间的指数函数
# 
# 默认$\alpha$为1

#%%
X=torch.randn(3,4)
X


#%%
##对于多维数据来说，是逐元素进行处理
##下面只画出图像，不再示例多维数据
m=torch.nn.ELU()
m(X)


#%%
def plot_activationFunc(x,y,title=''):
    if title != '':
        plt.title(title)
    
    if type(x)==torch.Tensor and type(y)==torch.Tensor:
        plt.plot(x.numpy(),y.numpy());
    else:
        plt.plot(x,y);


#%%
##>2. HardShrink
m=torch.nn.Hardshrink()
plot_activationFunc(x,m(x),'HardShrink')

#%% [markdown]
# 在 0 的某个$[-\lambda,\lambda]$区间内，输出为0；其他区间输出为x
# 
# 默认$\lambda$为0.5

#%%
##>3. HardTanh
var={'min_val':-0.5,'max_val':0.5}
m=torch.nn.Hardtanh(**var)
plot_activationFunc(x,m(x),'Hardtanh')

#%% [markdown]
# - $x>max\_val,则x=max\_val$
# 
# - $x<min\_val,则x=min\_val$
# 
# - 在两者之间时，输出为x自身
# 
# 默认min_val，max_val为-1和1

#%%
##>4. LeakyReLU
m=torch.nn.LeakyReLU(0.1)
plot_activationFunc(x,m(x),'LeakyReLU')

#%% [markdown]
# - $x>0时，y=x$
# 
# - $x<0时，y=-slope*(x). 默认slope为0.01$

#%%
##>5. LogSigmoid
m=torch.nn.LogSigmoid()
plot_activationFunc(x,m(x),'LogSigmoid')

#%% [markdown]
# $LogSigmoid=log(\frac{1}{1+exp(-x)})$

#%%
##>6. PReLUm=torch.nn.PReLU()
x=x.to(torch.float)
y=m(x)
plot_activationFunc(x,y.data,'PReLU')

#%% [markdown]
# $PReLU(x)=max(0,x)+a∗min(0,x) $
# 即：
# - y=x,如果$x\ge0$
# - y=ax，如果x<0
# 
# a是个可学习的参数，可根据不同的通道数n，设置n个a，默认不设置，即只有1个参数
# 
# 详见:https://pytorch.org/docs/stable/nn.html#torch.nn.PReLU

#%%
##>7. ReLu
##这个不用说了


#%%
##>8. ReLu6
m=nn.ReLU6()
x8=torch.linspace(-8,8,1000)
y=m(x8)
plot_activationFunc(x8,y,'ReLU6')

#%% [markdown]
# $ReLU6(x)=min(max(0,x),6)$
# 
# 即到6之后，y=6

#%%
##>9. RReLu
m=nn.RReLU(0.1,0.3)
y=m(x)
plot_activationFunc(x,y,'RReLU')


#%%
##>10. SELU
m=nn.SELU()
y=m(x)
plot_activationFunc(x,y,'SELU')


#%%
##>11. sigmoid


#%%
##>12. softplus
m=nn.Softplus()
y=m(x)
plot_activationFunc(x,y,'softplus')

#%% [markdown]
# $Softplus(x)=\frac{1}{β}∗log(1+exp(β∗x))$
# 
# 平滑版的ReLU，$\beta$默认为1

#%%
##>13. softshrink
m=nn.Softshrink()
y=m(x)
plot_activationFunc(x,y,'Softshrink')


#%%
##>14. softsign
m=nn.Softsign()
y=m(x)
plot_activationFunc(x,y,'Softsign')


#%%
##>15. tanh


#%%
##>16. tanhshrink
m=nn.Tanhshrink()
y=m(x)
plot_activationFunc(x,y,'Tanhshrink')

#%% [markdown]
# $Tanhshrink(x)=x−Tanh(x)$

#%%
##>17. threshold
m=nn.Threshold(0,-1)
y=m(x)
plot_activationFunc(x,y,'Threshold')


#%%
##>18. softmin
m=nn.Softmin(dim=0)
x18=torch.arange(6,dtype=torch.float).reshape(2,3)
x18


#%%
m(x18)


#%%
m(x18)[:,0].sum()


#%%
##>19. softmax
m=nn.Softmax(dim=1)
x19=x18
m(x19)


#%%
m(x19)[0,:].sum()


#%%
1/(1+np.exp(1)+np.exp(2))


#%%
np.exp(3)/(np.exp(3)+np.exp(4)+np.exp(5))


#%%
##>20. softmax2d
m=nn.Softmax2d()
x20=torch.arange(32,dtype=torch.float).reshape(1,2,4,4)
x20


#%%
m(x20).round()


#%%
x20_=torch.randn(1,2,4,4)
x20_


#%%
m(x20_)


#%%
m(x20_)[0,0,:,:]+m(x20_)[0,1,:,:]

#%% [markdown]
# ### 这个函数是对不同通道同一位置的数据做softmax, 使得它们的和为1

#%%
##>21 LogSoftmax
m=nn.LogSoftmax(dim=1)
x21=torch.arange(3,dtype=torch.float).reshape(1,3)
m(x21)


#%%
##>22. AdaptiveLogSoftmaxWithLoss
##是一种对于有较大输出空间模型的softmax的近似
##详见：https://pytorch.org/docs/stable/nn.html#torch.nn.AdaptiveLogSoftmaxWithLoss


