#%% [markdown]
# ## 回归的线性模型
# 
# - 使用pytorch求解线性回归问题（超定方程的最小二乘解）
#     - 使用pytorch.gels（对应的是正规方程的解析闭式解）
#     - 使用pytorc.nn.Linear，相当于对gels的封装，例如不用为设计矩阵加1等
#     - 使用优化器. 当超定方程过大，无法一次加载，可使用基于梯度下降的优化算法进行迭代求数值解
#    
# 
# - 损失函数的分类
#     - L1
#     - L2
#     - SmoothL1
#  
# 
# - 数据的标准化
#     - 标准化的作用

#%%
import numpy as np
import torch

##1.1-多元线性回归
X = torch.tensor([[1,1,1,],[2,3,1],[3,5,1],[4,2,1],[5,4,1]],dtype=torch.float)
y = torch.tensor([-10,12,14,16,18],dtype=torch.float)
w,_=torch.gels(y,X)##这里的顺序不能错
w[:3]


#%%
##1.2-多变量线性回归
Y=torch.tensor([[-10,3],[12,14],[14,12],[16,16],[18,16]],dtype=torch.float)
W,_=torch.gels(Y,X)
W[:3,:]


#%%
##2.损失函数
E=torch.nn.MSELoss()
y=torch.arange(5,requires_grad=True,dtype=torch.float)
t=torch.ones(5)

##2.1-得到均方误差损失
loss=E(y,t)##注意这里的顺序不能错，t属于标签数据，y是预测数据，要对y相关的变量求梯度
loss


#%%
torch.mean((y-t)**2)


#%%
##2.2-L1 Loss
E_L1=torch.nn.L1Loss()
loss2=E_L1(y,t)
loss2


#%%
##2.3-SmoothL1 Loss
E_SL1=torch.nn.SmoothL1Loss()
loss3=E_SL1(y,t)
loss3


#%%
##3.使用优化器求解回归问题的解
X


#%%
y = torch.tensor([-10,12,14,16,18],dtype=torch.float)
y


#%%
w=torch.zeros(3,requires_grad=True,dtype=torch.float)
E=torch.nn.MSELoss()
opt=torch.optim.Adam([w])
loss=E(torch.mv(X,w),y)

for i in range(30000):
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    loss=E(torch.mv(X,w),y)
    
    if i%1000 == 0:
        print('第{}次：loss={},w={},grad_w={}'.format(i,np.round(loss.tolist(),3),  
                                                   np.round(w.tolist(),3),
                                                   np.round(w.grad.tolist(),3)))


#%%
##4.使用torch.nn.Linear求解线性回归问题
X=X[:,:2]
X


#%%
y=y.reshape(-1,1)
y.shape


#%%
##生成一个全连接层：2个输入，1个输出
##或者说是该层有1个神经元，这个神经元有2个突触

fc=torch.nn.Linear(2,1)##权值包含在这个fc中
E=torch.nn.MSELoss()
opt=torch.optim.Adam(fc.parameters())

pred=fc(X)
loss=E(fc(X),y)

w,b=fc.parameters()

for i in range(30000):
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    loss=E(fc(X),y)
    
    
    if i%1000 ==0:
        print('第{}次：loss={},w={},grad_w={},b={},grad_b={}'.format(i,np.round( loss.tolist() ,3)
                                                                  ,w,w.grad.tolist(),b,b.grad.tolist()))


#%%
v=torch.arange(1,6,3)
v.dtype


#%%
i=v.double()
i.dtype


#%%
x = torch.Tensor([.1])


#%%
w=torch.tensor([2,2,2],dtype=torch.float,requires_grad=True)
y=x*w
z=y.sum()
z.backward()


#%%
x.grad


#%%
w.grad


#%%
type(z)


#%%
type(x)


#%%
from torch.autograd import Variable as Var
xv=Var(x,requires_grad=True)
wv=Var(w)
yv=xv*wv
zv=yv.sum()
zv.backward(retain_graph=True)
xv.grad


#%%
xv.grad


#%%
type(zv)


#%%
zv.grad_fn


#%%
zv.grad


#%%
yv.grad_fn


#%%
yv.grad_fn.next_functions


#%%
zv.data


#%%
zv.grad_fn.next_functions


#%%
type(zv.grad)


#%%
xv.requires_grad


#%%
type(xv)


#%%
i=torch.tensor([1,2,3],dtype=torch.int)
i


#%%
i.type()


#%%
type(torch.FloatTensor)==type(torch.FloatTensor)


#%%
zv.backward()


#%%
xv.grad


#%%
yv.grad_fn


#%%
type(xv.size())


#%%



#%%



