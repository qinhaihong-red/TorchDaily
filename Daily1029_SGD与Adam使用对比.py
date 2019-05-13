
#%%
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

#%% [markdown]
# # 0.问题
# 
# 求 Ax=b 的最小二乘解，其中A是满秩矩阵.
# 
# 即 $min \|Ax-b\|^{2} $ 的范数逼近问题.
#%% [markdown]
# # 1.要点
# 
# - 解析解 vs 迭代数值解
# 
# 
# - SGD在不使用lr调度器的情况下，设置学习率非常重要. 较大的学习率导致无法进行学习（示例4.1中0.05就已经太大了）.
# 
# 
# - Adam应该保证足够的循环次数
# 
# 
# - 学习率调度器
# 
#%% [markdown]
# 
# 
# # 2.数据
# 
# 其中X表示上面的A. 
# 
# b不带截距.

#%%
X=torch.arange(1,7,dtype=torch.float).reshape(3,2)
X


#%%
##torch里面的dot，只能用于向量对向量的内积
##torch.dot(X,X.t())


#%%
b=torch.tensor([7,8,9],dtype=torch.float).reshape(3,-1)
b

#%% [markdown]
# # 3. 解析方法：使用torch的线性代数库解
# 
# ## 3.1 torch.gels

#%%
p,_=torch.gels(b,X)


#%%
p

#%% [markdown]
# ## 3.2 使用解析解进行手动计算验证
# 
# 即 $\hat{x} = (A^{T}A)^{-1}A^{T}b$

#%%
a1=torch.inverse(torch.mm(X.t(),X))


#%%
a2=torch.mv(X.t(),b.squeeze(1))


#%%
print(a1,'\n',a2)


#%%
torch.mv(a1,a2)

#%% [markdown]
# # 4.数值迭代方法：使用SGD

#%%
l=nn.Linear(2,1,bias=False)##不使用偏置


#%%
X.requires_grad_(True)


#%%
l

#%% [markdown]
# ## 4.1 lr=0.05

#%%
#l.weight.data.normal_(0,1)##参数初始化
l.weight.data=torch.tensor([[0.3,0.2]])
opt=optim.SGD(l.parameters(),momentum=0.9,lr=0.03,weight_decay=0.001)
#sheduler=optim.lr_scheduler.MultiStepLR(opt,[2],0.5)
sheduler=optim.lr_scheduler.ExponentialLR(opt,0.9)
E=nn.MSELoss()


#%%
n=torch.ones(3)
torch.norm(n)


#%%
print("before trainning:")
for i in l.parameters():
    print(i)
    
print()
for i in range(100):
    #if i%10 == 0:
        #print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)
    opt.zero_grad()
    #sheduler.zero_grad()
    #if i%10 == 0:
        #print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)
    loss=E(l(X),b)
    loss.backward()
    
   
    opt.step()
    if i%10 == 0:
        sheduler.step()
        print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad,'grad norm:',torch.norm(l.weight.grad))

print("\nafter trainning:")
for i in l.parameters():
    print(i)

print(opt.param_groups[0])

#%% [markdown]
# __4.1, lr=0.05 ：学习率过大，无法进行学习__
#%% [markdown]
# ## 4.2 lr=0.02

#%%
l.weight.data.normal_(0,1)##参数初始化
opt=optim.SGD(l.parameters(),lr=0.02)
E=nn.MSELoss()
loss=E(l(X),b)
for i in range(20000):
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    loss=E(l(X),b)
    if i%1000 == 0:
        print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)

for i in l.parameters():
    print(i)

#%% [markdown]
# __4.2，lr=0.02, 在3000次就已经基本收敛__
#%% [markdown]
# ## 4.3 lr=0.01
# 
# 

#%%
l.weight.data.normal_(0,1)##参数初始化
opt=optim.SGD(l.parameters(),lr=0.01)
E=nn.MSELoss()
loss=E(l(X),b)
for i in range(20000):
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    loss=E(l(X),b)
    if i%1000 == 0:
        print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)

for i in l.parameters():
    print(i)

#%% [markdown]
# __4.3,lr=0.01,收敛稍慢，在5000次以上收敛__
#%% [markdown]
# ## 4.4 动态学习率：使用lr调度器

#%%
l.weight.data.normal_(0,1)##参数初始化
opt=optim.SGD(l.parameters(),lr=0.04)

##学习率调度器
##小于40：lr为0.04
##40到1500：lr为0.2
##1500--：lr为0.01
scheduler=optim.lr_scheduler.MultiStepLR(opt,[40,1500],0.5)

E=nn.MSELoss()
loss=E(l(X),b)
for i in range(20000):
    opt.zero_grad()
    loss.backward()
    
    scheduler.step()
    opt.step()
    
    loss=E(l(X),b)
    if i%1000 == 0:
        print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)

for i in l.parameters():
    print(i)
    
print('\n')
opt.param_groups

#%% [markdown]
# __4.4,效果与4.3相比，收敛更快. 可以看到在第3000次的时候，精度比固定lr=00.1要好__
#%% [markdown]
# # 5 数值迭代法：Adam
# 
# Adam使用自适应学习率调节，可以预感到它的收敛率会比较慢.

#%%
l.weight.data.normal_(0,1)
opt=optim.Adam(l.parameters())
E=nn.MSELoss()
loss=E(l(X),b)
for i in range(20000):
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    loss=E(l(X),b)
    if i % 1000 == 0:
        print(i,':',' loss:',loss.data,' w:',l.weight.data,' grad:',l.weight.grad)

#%% [markdown]
# __5.Adam, 到达16000次以上，才收敛__

#%%



